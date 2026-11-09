# BLAST 'EM 💥 2026-11-08_202815_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: compression_flat_token_prices (arc/alexnot/config/compression_flat_token_prices.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 259s for 400 levels
- Solver time per level: median 8.0s, mean 18.0s, max 64.1s
- Phase medians per level: search parse 0.34s, search transform 7.15s
- Phase totals: search parse 206.4s, search transform 6977.8s, run on tests 7.2s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 67.50 | 16.9% | 0 | 0 |
| **TOTAL** | 400 | 67.50 | 16.9% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
