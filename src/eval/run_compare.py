"""
DEPRECATED.

This file used to contain an older 2-method runner (LinUCB vs PD-LinUCB).
It drifted from the current codebase (nonexistent methods, incorrect pending-queue handling)
and could produce order-dependent or wrong results.

Use instead:
  python -m src.eval.run_compare_baselines --memmap_dir <...> --T 5000 --budget_ratio 0.7 --stop_at_budget --n_seeds 10
"""

raise SystemExit("Deprecated: use `python -m src.eval.run_compare_baselines ...`")
