# BLAST 'EM 💥 2026-11-08_212716_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: ablation_control (arc/alexnot/config/ablation_control.py)
- Workers: 32 (we may use 32 of this machine's 128 CPUs)
- Wall time: 262s for 400 levels
- Solver time per level: median 8.6s, mean 18.0s, max 65.1s
- Phase medians per level: search parse 0.30s, search transform 7.26s
- Phase totals: search parse 185.4s, search transform 6877.3s, run on tests 7.4s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 71.00 | 17.8% | 0 | 3 |
| **TOTAL** | 400 | 71.00 | 17.8% | 0 | 3 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
