# BLAST 'EM 💥 2026-11-08_205919_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: ablation_control (arc/alexnot/config/ablation_control.py)
- Workers: 32 (we may use 32 of this machine's 128 CPUs)
- Wall time: 283s for 400 levels
- Solver time per level: median 9.7s, mean 20.1s, max 68.5s
- Phase medians per level: search parse 0.39s, search transform 8.96s
- Phase totals: search parse 240.3s, search transform 7772.4s, run on tests 7.9s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 72.00 | 18.0% | 0 | 0 |
| **TOTAL** | 400 | 72.00 | 18.0% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
