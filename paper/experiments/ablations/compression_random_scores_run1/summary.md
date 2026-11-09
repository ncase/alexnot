# BLAST 'EM 💥 2026-11-08_221436_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: compression_random_scores (arc/alexnot/config/compression_random_scores.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 241s for 400 levels
- Solver time per level: median 11.1s, mean 16.5s, max 63.5s
- Phase medians per level: search parse 0.34s, search transform 10.75s
- Phase totals: search parse 206.5s, search transform 6367.6s, run on tests 6.0s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 19.00 | 4.8% | 0 | 0 |
| **TOTAL** | 400 | 19.00 | 4.8% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
