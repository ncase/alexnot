# BLAST 'EM 💥 2026-11-08_211758_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: ablation_control (arc/alexnot/config/ablation_control.py)
- Workers: 32 (we may use 32 of this machine's 128 CPUs)
- Wall time: 266s for 400 levels
- Solver time per level: median 7.9s, mean 18.0s, max 70.4s
- Phase medians per level: search parse 0.31s, search transform 7.32s
- Phase totals: search parse 183.6s, search transform 6919.8s, run on tests 7.1s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 71.50 | 17.9% | 0 | 2 |
| **TOTAL** | 400 | 71.50 | 17.9% | 0 | 2 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
