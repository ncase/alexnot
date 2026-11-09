# BLAST 'EM 💥 2026-11-08_204851_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: search_rounds_3 (arc/alexnot/config/search_rounds_3.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 108s for 400 levels
- Solver time per level: median 3.5s, mean 6.3s, max 61.6s
- Phase medians per level: search parse 0.34s, search transform 3.08s
- Phase totals: search parse 203.7s, search transform 2305.5s, run on tests 4.9s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 60.00 | 15.0% | 0 | 0 |
| **TOTAL** | 400 | 60.00 | 15.0% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
