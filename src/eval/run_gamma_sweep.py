# src/eval/run_gamma_sweep.py
from __future__ import annotations

import sys
from pathlib import Path
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from src.env.sim_bandit_env import SimBanditEnv
from src.algos.cost_normalized_ucb import CostNormalizedDisjointUCB
from src.algos.linucb_pd_delayed import PrimalDualLinUCB
from src.eval.runner_utils import mean_ci, run_contextual_delayed


@dataclass
class Row:
    gamma: float
    seed: int
    reward: float
    spent_ratio: float
    t_stop: int


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Empty rows for {path}")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_with_ci(x, y, yerr, title, xlabel, ylabel, out_path: Path, xscale: str = "symlog"):
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=3)
    if xscale:
        plt.xscale(xscale, linthresh=0.1)
    plt.grid(True, alpha=0.3)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap_dir", type=str, required=True)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--budget_ratio", type=float, default=0.7)
    ap.add_argument("--stop_at_budget", action="store_true")

    ap.add_argument("--gammas", type=str, default="0,0.1,0.3,1,2,3,5,10")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=123)
    ap.add_argument("--env_seed", type=int, default=0)
    ap.add_argument(
        "--context_split",
        type=str,
        default="auto",
        choices=["auto", "all", "train", "test"],
        help="Which context split to sample from. 'auto' uses test when split metadata exists.",
    )

    ap.add_argument("--alpha_cnu", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)

    ap.add_argument("--also_pd", action="store_true")
    ap.add_argument("--alpha_pd", type=float, default=1.5)
    ap.add_argument("--eta_pd", type=float, default=0.05)

    ap.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="If empty, uses paper_artifacts/figures/gamma_sweep_rho{rho}_T{T}",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else Path(
        f"paper_artifacts/figures/gamma_sweep_rho{args.budget_ratio}_T{args.T}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    gammas = [float(s) for s in args.gammas.split(",") if s.strip() != ""]
    if not gammas:
        raise RuntimeError("Empty --gammas list")

    rows_raw: list[Row] = []
    env_base = SimBanditEnv.from_memmap_dir(
        args.memmap_dir,
        seed=args.env_seed,
        ridge_lambda=1.0,
        cost_mode="lin",
        context_split=args.context_split,
    )

    for si in range(args.n_seeds):
        seed = int(args.seed0 + si)
        X_seq = env_base.sample_contexts(args.T, rng=np.random.default_rng(seed))

        for g in gammas:
            env = env_base.clone(seed=seed + 10_000 + int(round(g * 1000)))
            algo = CostNormalizedDisjointUCB(
                env.K, env.d,
                costs=env.costs,
                alpha=float(args.alpha_cnu),
                lam=float(args.lam),
                mode="sub",
                gamma=float(g),
                eps=1e-3,
                seed=seed + 20_000 + int(round(g * 1000)),
            )
            res = run_contextual_delayed(
                env, X_seq, algo,
                budget_ratio=args.budget_ratio,
                stop_at_budget=bool(args.stop_at_budget),
                is_primal_dual=False,
            )
            rows_raw.append(
                Row(
                    gamma=g,
                    seed=seed,
                    reward=float(res.reward),
                    spent_ratio=float(res.spent_ratio),
                    t_stop=int(res.t_stop),
                )
            )

        print(f"done seed {seed}")

    raw_path = out_dir / "gamma_sweep_raw.csv"
    write_csv(raw_path, [r.__dict__ for r in rows_raw])

    summary = []
    for g in sorted(set(r.gamma for r in rows_raw)):
        rr = [r for r in rows_raw if r.gamma == g]
        reward_m, reward_ci = mean_ci([x.reward for x in rr])
        spent_m, spent_ci = mean_ci([x.spent_ratio for x in rr])
        tstop_m, tstop_ci = mean_ci([float(x.t_stop) for x in rr])
        summary.append({
            "gamma": g,
            "n": len(rr),
            "reward_mean": reward_m,
            "reward_ci95": reward_ci,
            "spent_mean": spent_m,
            "spent_ci95": spent_ci,
            "tstop_mean": tstop_m,
            "tstop_ci95": tstop_ci,
        })

    sum_path = out_dir / "gamma_sweep_summary.csv"
    write_csv(sum_path, summary)

    x = [d["gamma"] for d in summary]
    reward = [d["reward_mean"] for d in summary]
    reward_ci = [d["reward_ci95"] for d in summary]
    spent = [d["spent_mean"] for d in summary]
    spent_ci = [d["spent_ci95"] for d in summary]

    plot_with_ci(
        x, reward, reward_ci,
        title="CostNormUCB[sub]: mean total reward vs gamma (95% CI)",
        xlabel="gamma (symlog axis)",
        ylabel="mean total reward",
        out_path=out_dir / "gamma_sweep_reward.png",
        xscale="symlog",
    )

    plot_with_ci(
        x, spent, spent_ci,
        title="CostNormUCB[sub]: mean spent/B vs gamma (95% CI)",
        xlabel="gamma (symlog axis)",
        ylabel="mean spent/B",
        out_path=out_dir / "gamma_sweep_spent.png",
        xscale="symlog",
    )

    if args.also_pd:
        pd_rows = []
        for si in range(args.n_seeds):
            seed = int(args.seed0 + si)
            X_seq = env_base.sample_contexts(args.T, rng=np.random.default_rng(seed))
            env = env_base.clone(seed=seed + 30_000)
            algo = PrimalDualLinUCB(
                env.K, env.d,
                costs=env.costs,
                alpha=float(args.alpha_pd),
                lam=float(args.lam),
                eta=float(args.eta_pd),
                seed=seed + 40_000,
            )
            res = run_contextual_delayed(
                env, X_seq, algo,
                budget_ratio=args.budget_ratio,
                stop_at_budget=bool(args.stop_at_budget),
                is_primal_dual=True,
            )
            pd_rows.append(res)

        pd_r = np.array([x.reward for x in pd_rows], dtype=float)
        pd_s = np.array([x.spent_ratio for x in pd_rows], dtype=float)
        pd_t = np.array([x.t_stop for x in pd_rows], dtype=float)

        print("\nPD-LinUCB baseline (same T,rho):")
        print("reward mean=", pd_r.mean(), "std=", pd_r.std(ddof=1))
        print("spent/B mean=", pd_s.mean(), "std=", pd_s.std(ddof=1))
        print("t_stop mean=", pd_t.mean(), "std=", pd_t.std(ddof=1))

    print("\nWrote:", raw_path)
    print("Wrote:", sum_path)
    print("Saved:", out_dir / "gamma_sweep_reward.png")
    print("Saved:", out_dir / "gamma_sweep_spent.png")


if __name__ == "__main__":
    main()
