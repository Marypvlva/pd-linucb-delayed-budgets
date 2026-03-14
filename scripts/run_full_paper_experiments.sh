#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/run_full_paper_experiments.sh <memmap_dir> [artifact_dir]" >&2
  exit 2
fi

MEMMAP_DIR="$1"
ARTIFACT_DIR="${2:-paper_artifacts}"
REWARD_MODEL="${REWARD_MODEL:-linear_clip}"
POLICY_MODEL="${POLICY_MODEL:-linear}"

export MPLBACKEND="${MPLBACKEND:-Agg}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$(pwd)/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

if [[ -f "$MEMMAP_DIR/arm_logistic_stats.npz" ]]; then
  DIAG_MODELS="${DIAG_MODELS:-linear_clip,logistic}"
else
  DIAG_MODELS="${DIAG_MODELS:-linear_clip}"
fi

python -m src.eval.run_compare_baselines \
  --memmap_dir "$MEMMAP_DIR" \
  --context_split auto \
  --reward_model "$REWARD_MODEL" \
  --policy_model "$POLICY_MODEL" \
  --T 5000 --budget_ratio 0.7 --stop_at_budget \
  --n_seeds 10 --seed0 123 \
  --artifact_dir "$ARTIFACT_DIR" \
  --tag T5000_rho0.7

python -m src.eval.run_gamma_sweep \
  --memmap_dir "$MEMMAP_DIR" \
  --context_split auto \
  --reward_model "$REWARD_MODEL" \
  --policy_model "$POLICY_MODEL" \
  --T 5000 --budget_ratio 0.7 --stop_at_budget \
  --n_seeds 10 --seed0 123 \
  --out_dir "$ARTIFACT_DIR/figures/gamma_sweep_rho0.7_T5000"

python -m src.eval.run_delay_ablation \
  --memmap_dir "$MEMMAP_DIR" \
  --context_split auto \
  --reward_model "$REWARD_MODEL" \
  --policy_model "$POLICY_MODEL" \
  --T 5000 --budget_ratio 0.7 --stop_at_budget \
  --n_seeds 10 --seed0 123 \
  --artifact_dir "$ARTIFACT_DIR"

python -m src.eval.run_budget_sweep_baselines \
  --memmap_dir "$MEMMAP_DIR" \
  --context_split auto \
  --reward_model "$REWARD_MODEL" \
  --policy_model "$POLICY_MODEL" \
  --T 5000 --stop_at_budget \
  --budgets 0.40,0.55,0.70,0.85 \
  --gammas 0,0.1,0.3,1,2,3,5,10 \
  --n_seeds 10 --seed0 123 \
  --artifact_dir "$ARTIFACT_DIR"

python -m src.eval.run_simulator_diagnostics \
  --memmap_dir "$MEMMAP_DIR" \
  --eval_split auto \
  --reward_models "$DIAG_MODELS" \
  --artifact_dir "$ARTIFACT_DIR"
