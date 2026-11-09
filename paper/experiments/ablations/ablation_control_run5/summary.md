# BLAST 'EM 💥 2026-11-08_210414_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: ablation_control (arc/alexnot/config/ablation_control.py)
- Workers: 32 (we may use 32 of this machine's 128 CPUs)
- Wall time: 271s for 400 levels
- Solver time per level: median 9.0s, mean 19.0s, max 66.6s
- Phase medians per level: search parse 0.34s, search transform 8.20s
- Phase totals: search parse 206.1s, search transform 7340.3s, run on tests 7.5s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 69.50 | 17.4% | 0 | 1 |
| **TOTAL** | 400 | 69.50 | 17.4% | 0 | 1 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
