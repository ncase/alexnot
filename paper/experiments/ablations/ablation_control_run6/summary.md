# BLAST 'EM 💥 2026-11-08_210857_arc-1-eval

- Random seed: -1 (PYTHONHASHSEED=None)
- Config: ablation_control (arc/alexnot/config/ablation_control.py)
- Workers: 32 (we may use 32 of this machine's 128 CPUs)
- Wall time: 255s for 400 levels
- Solver time per level: median 8.6s, mean 17.7s, max 65.3s
- Phase medians per level: search parse 0.30s, search transform 7.91s
- Phase totals: search parse 184.5s, search transform 6840.8s, run on tests 7.2s

| List | Levels | Score (pass@2) | % | Timeouts | Errors/crashes |
|---|---|---|---|---|---|
| arc-1-eval | 400 | 70.50 | 17.6% | 0 | 1 |
| **TOTAL** | 400 | 70.50 | 17.6% | 0 | 1 |

- **Caught solver crashes:** 0 levels had attempts crash
- **Precision:** not available yet: solve() doesn't report `confidence` (see the TODO in main_solver.py)
