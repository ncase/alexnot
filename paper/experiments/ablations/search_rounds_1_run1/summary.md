# BLAST 'EM 💥 2026-11-08_204811_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: search_rounds_1 (arc/alexnot/config/search_rounds_1.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 27s for 400 levels
- Solver time per level: median 1.0s, mean 1.6s, max 14.4s
- Phase medians per level: search parse 0.32s, search transform 0.52s
- Phase totals: search parse 192.5s, search transform 454.3s, run on tests 3.7s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 40.50 | 10.1% | 0 | 0 |
| **TOTAL** | 400 | 40.50 | 10.1% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
