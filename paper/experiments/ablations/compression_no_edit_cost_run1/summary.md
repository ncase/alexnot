# BLAST 'EM 💥 2026-11-08_201846_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: compression_no_edit_cost (arc/alexnot/config/compression_no_edit_cost.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 269s for 400 levels
- Solver time per level: median 7.3s, mean 18.9s, max 66.6s
- Phase medians per level: search parse 0.34s, search transform 6.73s
- Phase totals: search parse 206.7s, search transform 7336.8s, run on tests 9.9s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 72.00 | 18.0% | 0 | 0 |
| **TOTAL** | 400 | 72.00 | 18.0% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
