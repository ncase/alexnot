# BLAST 'EM 💥 2026-11-09_152652_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: search_rounds_5 (arc/alexnot/config/search_rounds_5.py)
- Workers: 32 (we may use 32 of this machine's 128 CPUs)
- Wall time: 195s for 400 levels
- Solver time per level: median 6.3s, mean 11.3s, max 72.8s
- Phase medians per level: search parse 0.35s, search transform 5.96s
- Phase totals: search parse 206.6s, search transform 4307.1s, run on tests 6.5s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 68.00 | 17.0% | 0 | 0 |
| **TOTAL** | 400 | 68.00 | 17.0% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
