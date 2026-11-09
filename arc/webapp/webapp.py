from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import traceback

import sys
from pathlib import Path

# So it works when run directly (python arc/webapp/webapp.py) from anywhere
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from alexnot.main_solver import solve
from alexnot.main_solver_batch import solve_batch, RED_CROSS, available_cpu_count
from flask import Response
import json
import queue
import threading
import time
from typing import Optional

################################################

# Create Flask app instance
app = Flask(__name__, static_folder='static', static_url_path='')
# Enable CORS for JS frontend integration
CORS(app)

# The Bayes posterior's tuning log: one line per solved level, with the raw ingredients (each top program's score,
# which output each program made per test, and whether that output was right), so the temperature + dummy score can
# be re-tuned offline without re-running the solver. Temporary: delete it (and this folder) once those are tuned.
BAYES_LOG_PATH: Path = Path(__file__).resolve().parent / "bayes_logs" / f"{time.strftime('%Y-%m-%d_%H%M%S')}.jsonl"
bayes_log_lock = threading.Lock()  # Single solves and batch levels can finish at the same time

def log_bayes_posterior(level_id: Optional[str], level_data: dict, bayes_posterior: Optional[dict]) -> None:
    """Append one level's posterior ingredients to the tuning log (skipped if solve() found no programs)."""
    if bayes_posterior is None:
        return
    tests: list = []
    for test_pair, outputs in zip(level_data["test"], bayes_posterior["tests"]):
        true_output: Optional[list] = test_pair.get("output")  # None if the level file has no answer
        tests.append([{
            "program_indices": output["program_indices"],
            "is_dummy": output["is_dummy"],
            "is_correct": (not output["is_dummy"]) and output["grid"] == true_output,
        } for output in outputs])
    line: dict = {
        "level_id": level_id,
        "temperature": bayes_posterior["temperature"],
        "dummy_score": bayes_posterior["dummy_score"],
        "program_scores": [program["score"] for program in bayes_posterior["programs"]],
        "program_is_dummy": [program["is_dummy"] for program in bayes_posterior["programs"]],
        "tests": tests,
    }
    with bayes_log_lock:
        BAYES_LOG_PATH.parent.mkdir(exist_ok=True)
        with open(BAYES_LOG_PATH, "a") as file:
            file.write(json.dumps(line) + "\n")

# One Front Page
@app.route('/')
def serve_index():
    return send_from_directory(app.static_folder or 'static','index.html')

# Serve up the Level json, too
@app.route('/levels/<path:filename>')
def serve_levels(filename):
    return send_from_directory('../levels',filename)

# POST in level data, feeds it into solver, get back solutions & CoT
@app.route('/api/solve', methods=['POST'])
def solve_level():
    try:
        
        # You need to give me a Level JSON, dingus.
        level_data = request.get_json()
        if not level_data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        # Solve! (bayes=True: the webapp also shows the approximate posterior, which never changes the answers)
        solution = solve(level_data, bayes=True)
        log_bayes_posterior(request.args.get("level_id"), level_data, solution.get("bayes_posterior"))
        return jsonify(solution)
    
    except Exception as e:
        # Print full traceback to console
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

# The "stop!" flag of the batch that's running right now (None if there isn't one).
# A new batch (or leaving the page, see /api/cancel_batch) sets the old flag, so the old batch's
# workers get killed, instead of two batches fighting over the same CPUs.
current_batch_stop_event: Optional[threading.Event] = None

# While nothing's finished, poke the page this often, to notice quickly if it's gone
HEARTBEAT_SECONDS: float = 1.0

@app.route('/api/solve_batch', methods=['POST'])
def solve_level_batch():
    global current_batch_stop_event
    try:

        # You need to give me some Level JSONs, dingus.
        levels = request.get_json()
        if not levels:
            return jsonify({"error": "No JSON data provided"}), 400

        # Cancel the old batch (if any), and remember this one's stop flag
        if current_batch_stop_event is not None:
            current_batch_stop_event.set()
        stop_event = threading.Event()
        current_batch_stop_event = stop_event

        # Solve 'em all!
        # submission = solve_batch(levels, stop_event=stop_event)
        # return jsonify(submission)
        # Instead of one big answer at the very end, STREAM each level back the moment it's solved.
        # The batch runs in a background thread, and drops each finished level into this queue
        finished_levels: queue.Queue = queue.Queue()

        def on_level_done(level_id: str, outcome: dict) -> None:
            # What the page needs per level: answers (red crosses if it timed out / crashed), plus the CoT
            reasoning_trace: str = outcome["reasoning_trace"]
            if "error" in outcome:
                reasoning_trace += "\n\n" + outcome["error"]  # The worker's traceback, if solve() itself blew up
            finished_levels.put({
                "level_id": level_id,
                "status": outcome["status"],
                "seconds": outcome["seconds"],
                "test_solutions": outcome["test_solutions"] if outcome["status"] == "ok"
                                  else [{"attempt_1": RED_CROSS, "attempt_2": RED_CROSS} for _ in levels[level_id]["test"]],
                "solved_all_demos": outcome["solved_all_demos"],
                "reasoning_trace": reasoning_trace,
                "bayes_posterior": outcome.get("bayes_posterior"),
            })
            log_bayes_posterior(level_id, levels[level_id], outcome.get("bayes_posterior"))

        def run_batch() -> None:
            # Solve 'em all, then put a None in the queue to say "that's everyone" (even if it crashed)
            try:
                solve_batch(levels, stop_event=stop_event, on_level_done=on_level_done, bayes=True)
            except Exception:
                traceback.print_exc()
            finally:
                finished_levels.put(None)

        threading.Thread(target=run_batch, daemon=True).start()

        def stream_levels():
            # First line: how many CPUs we're solving on (so the page's "done by" estimate isn't a guess)
            yield json.dumps({"workers": available_cpu_count()}) + "\n"
            try:
                while True:
                    # Wait for the next finished level...
                    try:
                        finished_level: Optional[dict] = finished_levels.get(timeout=HEARTBEAT_SECONDS)
                    except queue.Empty:
                        # ...nobody finished yet? Send a blank "heartbeat" line anyway. If the page was closed
                        # (or the stop button aborted the request), this write fails, and Flask closes this
                        # generator, which jumps straight to the `finally` below
                        yield "\n"
                        continue
                    if finished_level is None:
                        break  # Everyone's done!
                    yield json.dumps(finished_level) + "\n"
            finally:
                # Done, closed tab, or stop button: either way, kill whatever's still running
                stop_event.set()

        # One JSON object per line ("newline-delimited JSON"), which the page reads as it arrives
        return Response(stream_levels(), mimetype='application/x-ndjson')

    except Exception as e:
        # Print full traceback to console
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "status": "error"
        }), 500

# The page calls this when you leave it (or reload it), so the batch stops burning CPUs
@app.route('/api/cancel_batch', methods=['POST'])
def cancel_batch():
    if current_batch_stop_event is not None:
        current_batch_stop_event.set()
    return jsonify({"status": "ok"})

if __name__ == '__main__':

    # Pick the port with --port (default's still 9999), e.g. python webapp/webapp.py --port 8888
    import argparse
    argument_parser = argparse.ArgumentParser(description="AlexNot's visual tester")
    argument_parser.add_argument('--port', type=int, default=9999, help='Which port to serve on (default: 9999)')
    arguments = argument_parser.parse_args()

    # Run in debug mode for development
    app.run(debug=True, port=arguments.port)