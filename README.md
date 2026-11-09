# AlexNot

A pure-symbolic solver for ARC-AGI-2, that runs at $0.000035/task!

**Read the paper!** (⏱️ ~90 min reading, but it's full of illustrations & not academic bone-dry)

- Permalink (PDF): https://ncase.me/alexnot-paper
- Kaggle: (to come)

**See also:**

- AlexNot-ODE, a more minimalist version that induces real-science differential equations from noisy, sparse data: https://github.com/ncase/alexnot-ode
- The Kaggle notebook we submitted to the contest, which is also ensembled with NVARC (2025): https://www.kaggle.com/code/kittyhawkarc/alexnot-plus-nvarc

---

**Key documentation:**

 - [docs/DSL.md](docs/DSL.md): AlexNot's domain-specific language (DSL) for ARC-AGI tasks
 - [docs/MONKEYS.md](docs/MONKEYS.md): AlexNot's "monkeys" (code-proposing pattern-matching heuristics)
 - [arc/README.md](arc/README.md): the solver's own README, with more details on how to use it

**Structure of this repo:**

```
arc/
├── alexnot/        the solver (entry point: main_solver.py → solve(level_data))
├── webapp/         a little web app to watch the solver work, level by level
└── levels/         the puzzle datasets (JSON), plus one .txt list per dataset
docs/               DSL.md + MONKEYS.md
paper/
├── overleaf/       the paper's LaTeX, figures, references + the final PDF
└── code/           the code + data behind the paper's charts & experiments
tests/              the full-solver test, and BLAST_EM.py (runs many levels in parallel, Kaggle-style)
requirements.txt    Python dependencies
LICENSE             CC0
```

---

**How to run it** (tested with Python 3.13):

```bash
# Set up (from the repo root)
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# The webapp: pick a level, watch the solver solve it → http://127.0.0.1:9999/
# (runs from inside arc/; Ctrl+C to stop, then `cd ..` back to the repo root)
cd arc
../.venv/bin/python webapp/webapp.py

# Run the full solver on one level (pick the level in the "CHOOSE" section at the top of tests/test_arc2_solver.py)
.venv/bin/python -m tests.test_arc2_solver

# Blast many levels in parallel, Kaggle-style scoring. Results go in tests/blast_logs/<time>/
.venv/bin/python -m tests.BLAST_EM --datasets mini-arc
#   --datasets: any list(s) in arc/levels/*.txt, or 'all'  ·  -n N: N random levels per list
#   --workers N: parallel workers (default: all CPUs)  ·  --timeout SECONDS: per level (default: 120)
```

---

**Credits:**

Code & paper by [Nicky Case](https://ncase.me/) and Nora Danger.

All *our* code & paper & illustrations are given under the public domain [Creative Commons Zero (CC0 1.0)](https://creativecommons.org/publicdomain/zero/1.0/) waiver!

But as for other stuff that's in this repo, here's their links & licenses:

Level datasets (in `arc/levels/`):
- ARC-AGI-1 (`arc-1-train/`, `arc-1-eval/`): https://github.com/fchollet/ARC-AGI (Apache 2.0)
- ARC-AGI-2 (`arc-2-train/`, `arc-2-eval/`): https://github.com/arcprize/ARC-AGI-2 (Apache 2.0)
- ConceptARC (`concept-arc/`): https://github.com/victorvikram/ConceptARC (MIT)
- Mini-ARC (`mini-arc/`): https://github.com/KSB21ST/MINI-ARC (Apache 2.0)

Python libraries:
- func_timeout (stops a solve attempt at its time limit): https://github.com/kata198/func_timeout (LGPL-3.0)
- NumPy: https://numpy.org · SciPy: https://scipy.org
- Flask + Flask-CORS (the webapp): https://flask.palletsprojects.com · https://github.com/corydolphin/flask-cors
- Matplotlib (the paper's figures): https://matplotlib.org

Fonts (the paper's figures):
- Merriweather (SIL Open Font License): https://fonts.google.com/specimen/Merriweather
- Monaco (comes with macOS)
