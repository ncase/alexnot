# BLAST 'EM 💥 2026-11-08_203248_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: search_best_1 (arc/alexnot/config/search_best_1.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 170s for 400 levels
- Solver time per level: median 3.5s, mean 10.3s, max 66.1s
- Phase medians per level: search parse 0.34s, search transform 3.07s
- Phase totals: search parse 206.2s, search transform 3891.8s, run on tests 6.9s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 61.00 | 15.2% | 0 | 0 |
| **TOTAL** | 400 | 61.00 | 15.2% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
