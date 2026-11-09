# BLAST 'EM 💥 2026-11-08_204018_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: search_no_shape_diff_island (arc/alexnot/config/search_no_shape_diff_island.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 237s for 400 levels
- Solver time per level: median 6.8s, mean 16.3s, max 64.7s
- Phase medians per level: search parse 0.35s, search transform 6.23s
- Phase totals: search parse 206.2s, search transform 6292.5s, run on tests 6.6s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 69.00 | 17.2% | 0 | 0 |
| **TOTAL** | 400 | 69.00 | 17.2% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
