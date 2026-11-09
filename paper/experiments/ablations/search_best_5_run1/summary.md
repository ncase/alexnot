# BLAST 'EM 💥 2026-11-08_203551_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: search_best_5 (arc/alexnot/config/search_best_5.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 254s for 400 levels
- Solver time per level: median 7.6s, mean 17.4s, max 92.8s
- Phase medians per level: search parse 0.34s, search transform 6.95s
- Phase totals: search parse 204.5s, search transform 6709.3s, run on tests 50.5s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 66.50 | 16.6% | 0 | 0 |
| **TOTAL** | 400 | 66.50 | 16.6% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
