# BLAST 'EM 💥 2026-11-08_202328_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: compression_no_unique_cost (arc/alexnot/config/compression_no_unique_cost.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 274s for 400 levels
- Solver time per level: median 9.1s, mean 19.5s, max 64.9s
- Phase medians per level: search parse 0.34s, search transform 8.84s
- Phase totals: search parse 204.5s, search transform 7597.7s, run on tests 7.2s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 71.50 | 17.9% | 0 | 0 |
| **TOTAL** | 400 | 71.50 | 17.9% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
