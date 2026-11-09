# BLAST 'EM 💥 2026-11-08_204429_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: search_no_demo_islands (arc/alexnot/config/search_no_demo_islands.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 208s for 400 levels
- Solver time per level: median 4.7s, mean 13.6s, max 65.6s
- Phase medians per level: search parse 0.34s, search transform 4.35s
- Phase totals: search parse 204.4s, search transform 5240.0s, run on tests 8.0s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 65.50 | 16.4% | 0 | 0 |
| **TOTAL** | 400 | 65.50 | 16.4% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
