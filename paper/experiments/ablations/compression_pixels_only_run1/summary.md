# BLAST 'EM 💥 2026-11-08_200921_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: compression_pixels_only (arc/alexnot/config/compression_pixels_only.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 268s for 400 levels
- Solver time per level: median 7.6s, mean 18.5s, max 64.5s
- Phase medians per level: search parse 0.35s, search transform 7.10s
- Phase totals: search parse 205.8s, search transform 7148.2s, run on tests 33.1s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 64.50 | 16.1% | 0 | 0 |
| **TOTAL** | 400 | 64.50 | 16.1% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
