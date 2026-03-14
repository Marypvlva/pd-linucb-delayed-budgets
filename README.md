# pd-linucb-delayed-budgets (paper_cbwk_delays)

Prototype repository for a paper on contextual bandits with a budget constraint (BwK/CBwK) and delayed feedback.
The main focus is a reproducible large-scale empirical benchmark on Criteo Attribution with `numpy.memmap`,
a strict stop-at-budget feasible-set protocol, and comparisons between:

- Disjoint LinUCB
- Primal-Dual LinUCB (PD-LinUCB)
- CostNormUCB (ratio/sub)
- Context-free PD-BwK

## Important scientific notes

This repo implements a semi-synthetic simulator:

- Contexts `x_t` are sampled from the logged Criteo dataset.
- Rewards are generated from an offline-fitted arm-specific ridge model
  `mu_a(x)=clip(theta_a^T x,0,1)` with `r ~ Bernoulli(mu_a(x))`.
- The online budget controller uses arm-level mean costs `c(a)` computed from the `cost` field
  and normalized to mean approximately 1. This is not per-event counterfactual cost control.
- Delays:
  - For `r=1`, delays are sampled from an empirical pool of positive conversion delays.
  - For `r=0`, the environment uses censoring: delay equals `censor_steps = ceil(W/Δ)`.
  - In the default paper setting, `W/Δ = 5000` at `Δ=3600`, so `W=18,000,000` seconds
    (about 208.3 days).

## Paper artifacts

- `paper_artifacts/figures/` contains figures for Overleaf.
- `paper_artifacts/tables/` contains LaTeX tabulars such as `main_ci.tex`.
- `results/` can still be used as local scratch output, but the paper-facing scripts now write to `paper_artifacts/` by default.

## Environment setup

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel
python -m pip install -r requirements.txt
```

## 1) Preprocess Criteo into memmaps

```bash
python -m src.env.make_criteo_attrib_memmap_full \
  --inp data/raw/criteo_attrib/criteo_attribution_dataset.tsv.gz \
  --out_dir data/processed/criteo_full_k50_d64_real \
  --k_cap 50 --d_hash 64 \
  --delta_seconds 3600 \
  --censor_seconds $((5000*3600)) \
  --d_max 5000
```

Optional arm-wise ridge stats:

```bash
python -m src.env.compute_arm_ridge_stats_from_memmap \
  --memmap_dir data/processed/criteo_full_k50_d64_real \
  --lam 1.0 --out arm_ridge_stats.npz
```

## 2) Generate paper artifacts

Main comparison with mean±CI curves and the LaTeX summary table:

```bash
python -m src.eval.run_compare_baselines \
  --memmap_dir data/processed/criteo_full_k50_d64_real \
  --T 5000 --budget_ratio 0.7 --stop_at_budget \
  --n_seeds 10 --seed0 123 \
  --artifact_dir paper_artifacts \
  --tag T5000_rho0.7
```

Gamma sweep for `CostNormUCB[sub]`:

```bash
python -m src.eval.run_gamma_sweep \
  --memmap_dir data/processed/criteo_full_k50_d64_real \
  --T 5000 --budget_ratio 0.7 --stop_at_budget \
  --n_seeds 10 --seed0 123
```

Delay vs no-delay ablation:

```bash
python -m src.eval.run_delay_ablation \
  --memmap_dir data/processed/criteo_full_k50_d64_real \
  --T 5000 --budget_ratio 0.7 --stop_at_budget \
  --n_seeds 10 --seed0 123
```

Budget sweep over `rho=B/T` with per-rho `gamma*` selection:

```bash
python -m src.eval.run_budget_sweep_baselines \
  --memmap_dir data/processed/criteo_full_k50_d64_real \
  --T 5000 --stop_at_budget \
  --budgets 0.40,0.55,0.70,0.85 \
  --gammas 0,0.1,0.3,1,2,3,5,10 \
  --n_seeds 10 --seed0 123
```
