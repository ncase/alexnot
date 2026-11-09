# BLAST 'EM 💥 2026-11-08_195908_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: ablation_control (arc/alexnot/config/ablation_control.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 267s for 400 levels
- Solver time per level: median 8.8s, mean 18.9s, max 65.4s
- Phase medians per level: search parse 0.35s, search transform 8.19s
- Phase totals: search parse 210.6s, search transform 7333.9s, run on tests 7.6s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 68.00 | 17.0% | 0 | 0 |
| **TOTAL** | 400 | 68.00 | 17.0% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
