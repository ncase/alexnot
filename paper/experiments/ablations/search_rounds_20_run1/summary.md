# BLAST 'EM 💥 2026-11-08_213314_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: search_rounds_20 (arc/alexnot/config/search_rounds_20.py)
- Workers: 32 (we may use 32 of this machine's 128 CPUs)
- Wall time: 302s for 400 levels
- Solver time per level: median 8.7s, mean 21.0s, max 65.4s
- Phase medians per level: search parse 0.30s, search transform 8.27s
- Phase totals: search parse 181.8s, search transform 8120.5s, run on tests 7.1s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 69.00 | 17.2% | 0 | 2 |
| **TOTAL** | 400 | 69.00 | 17.2% | 0 | 2 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
