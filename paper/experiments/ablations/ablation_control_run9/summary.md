# BLAST 'EM 💥 2026-11-08_212238_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: ablation_control (arc/alexnot/config/ablation_control.py)
- Workers: 32 (we may use 32 of this machine's 128 CPUs)
- Wall time: 266s for 400 levels
- Solver time per level: median 8.8s, mean 18.5s, max 67.7s
- Phase medians per level: search parse 0.30s, search transform 8.12s
- Phase totals: search parse 182.8s, search transform 7227.9s, run on tests 7.2s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 67.50 | 16.9% | 0 | 0 |
| **TOTAL** | 400 | 67.50 | 16.9% | 0 | 0 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
