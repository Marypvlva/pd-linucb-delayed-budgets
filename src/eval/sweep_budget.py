"""
DEPRECATED.

Old script: swept rho only for PD-LinUCB and did not compare against tuned CostNormUCB[sub].
It also reused the same env instance across multiple runs, which made the evaluation order-dependent.

Use instead:
  python -m src.eval.run_budget_sweep_baselines --memmap_dir <...> --T 5000 --stop_at_budget
"""

raise SystemExit("Deprecated: use `python -m src.eval.run_budget_sweep_baselines ...`")
