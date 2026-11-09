# BLAST 'EM 💥 2026-11-09_153018_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: search_rounds_15 (arc/alexnot/config/search_rounds_15.py)
- Workers: 32 (we may use 32 of this machine's 128 CPUs)
- Wall time: 301s for 400 levels
- Solver time per level: median 9.0s, mean 21.5s, max 66.9s
- Phase medians per level: search parse 0.34s, search transform 8.48s
- Phase totals: search parse 212.9s, search transform 8387.9s, run on tests 7.9s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 70.00 | 17.5% | 0 | 0 |
| **TOTAL** | 400 | 70.00 | 17.5% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
