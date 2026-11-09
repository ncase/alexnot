# BLAST 'EM 💥 2026-11-08_201401_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: compression_pixels_and_length (arc/alexnot/config/compression_pixels_and_length.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 272s for 400 levels
- Solver time per level: median 7.1s, mean 19.0s, max 64.5s
- Phase medians per level: search parse 0.34s, search transform 6.74s
- Phase totals: search parse 205.5s, search transform 7383.2s, run on tests 7.7s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 68.50 | 17.1% | 0 | 0 |
| **TOTAL** | 400 | 68.50 | 17.1% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
