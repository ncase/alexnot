# BLAST 'EM 💥 2026-11-08_200348_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: ablation_control (arc/alexnot/config/ablation_control.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 263s for 400 levels
- Solver time per level: median 8.8s, mean 18.6s, max 66.1s
- Phase medians per level: search parse 0.35s, search transform 8.06s
- Phase totals: search parse 209.8s, search transform 7225.0s, run on tests 7.0s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 73.50 | 18.4% | 0 | 0 |
| **TOTAL** | 400 | 73.50 | 18.4% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
