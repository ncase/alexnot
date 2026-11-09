# BLAST 'EM 💥 2026-11-08_211325_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: ablation_control (arc/alexnot/config/ablation_control.py)
- Workers: 32 (we may use 32 of this machine's 128 CPUs)
- Wall time: 261s for 400 levels
- Solver time per level: median 7.9s, mean 18.0s, max 67.9s
- Phase medians per level: search parse 0.30s, search transform 7.49s
- Phase totals: search parse 187.0s, search transform 6922.0s, run on tests 6.5s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 71.00 | 17.8% | 0 | 2 |
| **TOTAL** | 400 | 71.00 | 17.8% | 0 | 2 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
