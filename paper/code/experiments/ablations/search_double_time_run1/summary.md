# BLAST 'EM 💥 2026-11-08_205054_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: search_double_time (arc/alexnot/config/search_double_time.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 400s for 400 levels
- Solver time per level: median 11.0s, mean 26.0s, max 125.0s
- Phase medians per level: search parse 0.35s, search transform 7.18s
- Phase totals: search parse 180.5s, search transform 7746.4s, run on tests 6.6s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 74.00 | 18.5% | 0 | 40 |
| **TOTAL** | 400 | 74.00 | 18.5% | 0 | 40 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
