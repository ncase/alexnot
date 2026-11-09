# BLAST 'EM 💥 2026-11-08_195314_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: ablation_control (arc/alexnot/config/ablation_control.py)
- Workers: 32 (we may use 32 of this machine's 32 CPUs)
- Wall time: 268s for 400 levels
- Solver time per level: median 8.8s, mean 18.8s, max 64.7s
- Phase medians per level: search parse 0.36s, search transform 8.11s
- Phase totals: search parse 211.3s, search transform 7301.5s, run on tests 7.0s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 70.50 | 17.6% | 0 | 0 |
| **TOTAL** | 400 | 70.50 | 17.6% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
