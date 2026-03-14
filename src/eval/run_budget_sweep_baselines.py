# src/eval/run_budget_sweep_baselines.py
from __future__ import annotations

import sys
from pathlib import Path
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt

from src.env.sim_bandit_env import SimBanditEnv
from src.algos.linucb_pd_delayed import PrimalDualLinUCB
from src.algos.cost_normalized_ucb import CostNormalizedDisjointUCB
from src.eval.runner_utils import mean_ci, run_contextual_delayed


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Empty rows for {path}")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def write_budget_sweep_tabular(tex_path: Path, rows: list[dict]):
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l l r r r r r r}\n")
        f.write("\\toprule\n")
        f.write("$\\\\rho$ & Method & Reward (mean) & 95\\% CI & Spent$/B$ (mean) & 95\\% CI & $\\\\tau$ (mean) & 95\\% CI \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(
                f"{r['rho']:.2f} & {r['method']} & "
                f"{r['reward_mean']:.1f} & {r['reward_ci']:.1f} & "
                f"{r['spent_mean']:.6f} & {r['spent_ci']:.6f} & "
                f"{r['tau_mean']:.1f} & {r['tau_ci']:.1f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap_dir", type=str, required=True)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--stop_at_budget", action="store_true")
    ap.add_argument("--budgets", type=str, default="0.40,0.55,0.70,0.85")
    ap.add_argument("--gammas", type=str, default="0,0.1,0.3,1,2,3,5,10")
    ap.add_argument("--min_spent_ratio", type=float, default=0.99)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=123)
    ap.add_argument("--env_seed", type=int, default=0)
    ap.add_argument("--alpha_pd", type=float, default=1.5)
    ap.add_argument("--eta_pd", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alpha_cnu", type=float, default=1.0)
    ap.add_argument("--artifact_dir", type=str, default="paper_artifacts")
    args = ap.parse_args()

    budgets = [float(x) for x in args.budgets.split(",") if x.strip() != ""]
    gammas = [float(x) for x in args.gammas.split(",") if x.strip() != ""]
    if not budgets:
        raise RuntimeError("Empty --budgets")
    if not gammas:
        raise RuntimeError("Empty --gammas")

    artifact_dir = Path(args.artifact_dir)
    fig_dir = artifact_dir / "figures"
    tab_dir = artifact_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    env_base = SimBanditEnv.from_memmap_dir(args.memmap_dir, seed=args.env_seed, ridge_lambda=1.0, cost_mode="lin")
    seeds = [int(args.seed0 + i) for i in range(int(args.n_seeds))]
    X_by_seed = {seed: env_base.sample_contexts(args.T, rng=np.random.default_rng(seed)) for seed in seeds}

    raw = []

    for rho in budgets:
        for seed in seeds:
            X_seq = X_by_seed[seed]

            env_pd = env_base.clone(seed=seed + 10_000 + int(round(rho * 1000)))
            pd = PrimalDualLinUCB(
                env_base.K,
                env_base.d,
                costs=env_base.costs,
                alpha=float(args.alpha_pd),
                lam=float(args.lam),
                eta=float(args.eta_pd),
                seed=seed + 20_000 + int(round(rho * 1000)),
            )
            res_pd = run_contextual_delayed(
                env_pd,
                X_seq,
                pd,
                budget_ratio=float(rho),
                stop_at_budget=bool(args.stop_at_budget),
                is_primal_dual=True,
            )
            raw.append({
                "rho": float(rho),
                "seed": int(seed),
                "method": "PD-LinUCB",
                "gamma": "",
                "reward": float(res_pd.reward),
                "spent_ratio": float(res_pd.spent_ratio),
                "tau": int(res_pd.t_stop),
            })

            for g in gammas:
                env_cnu = env_base.clone(
                    seed=seed + 30_000 + int(round(rho * 1000)) + int(round(g * 1000))
                )
                cnu = CostNormalizedDisjointUCB(
                    env_base.K,
                    env_base.d,
                    costs=env_base.costs,
                    alpha=float(args.alpha_cnu),
                    lam=float(args.lam),
                    mode="sub",
                    gamma=float(g),
                    eps=1e-3,
                    seed=seed + 40_000 + int(round(rho * 1000)) + int(round(g * 1000)),
                )
                res_cnu = run_contextual_delayed(
                    env_cnu,
                    X_seq,
                    cnu,
                    budget_ratio=float(rho),
                    stop_at_budget=bool(args.stop_at_budget),
                    is_primal_dual=False,
                )
                raw.append({
                    "rho": float(rho),
                    "seed": int(seed),
                    "method": "CostNormUCB[sub]",
                    "gamma": float(g),
                    "reward": float(res_cnu.reward),
                    "spent_ratio": float(res_cnu.spent_ratio),
                    "tau": int(res_cnu.t_stop),
                })

        print(f"done rho={rho:.2f}")

    raw_path = tab_dir / "budget_sweep_raw.csv"
    write_csv(raw_path, raw)
    print("Wrote:", raw_path)

    final_rows_for_tex = []
    plot_rho = []
    plot_pd_reward = []
    plot_pd_reward_ci = []
    plot_best_reward = []
    plot_best_reward_ci = []
    plot_best_gamma = []
    plot_best_spent = []
    plot_best_spent_ci = []

    for rho in budgets:
        pd_rr = [r for r in raw if r["rho"] == rho and r["method"] == "PD-LinUCB"]
        pd_reward_m, pd_reward_ci = mean_ci([r["reward"] for r in pd_rr])
        pd_spent_m, pd_spent_ci = mean_ci([r["spent_ratio"] for r in pd_rr])
        pd_tau_m, pd_tau_ci = mean_ci([float(r["tau"]) for r in pd_rr])

        final_rows_for_tex.append({
            "rho": rho,
            "method": "PD-LinUCB",
            "reward_mean": pd_reward_m,
            "reward_ci": pd_reward_ci,
            "spent_mean": pd_spent_m,
            "spent_ci": pd_spent_ci,
            "tau_mean": pd_tau_m,
            "tau_ci": pd_tau_ci,
        })

        gamma_summ = []
        for g in gammas:
            rr = [
                r for r in raw
                if r["rho"] == rho and r["method"] == "CostNormUCB[sub]" and float(r["gamma"]) == float(g)
            ]
            r_m, r_ci = mean_ci([x["reward"] for x in rr])
            s_m, s_ci = mean_ci([x["spent_ratio"] for x in rr])
            t_m, t_ci = mean_ci([float(x["tau"]) for x in rr])
            gamma_summ.append({
                "gamma": g,
                "reward_mean": r_m,
                "reward_ci": r_ci,
                "spent_mean": s_m,
                "spent_ci": s_ci,
                "tau_mean": t_m,
                "tau_ci": t_ci,
            })

        feasible = [d for d in gamma_summ if d["spent_mean"] >= float(args.min_spent_ratio)]
        if feasible:
            best = max(feasible, key=lambda d: d["reward_mean"])
        else:
            best = max(gamma_summ, key=lambda d: (d["spent_mean"], d["reward_mean"]))

        best_label = f"CostNormUCB[sub] ($\\gamma^\\star={best['gamma']}$)"
        final_rows_for_tex.append({
            "rho": rho,
            "method": best_label,
            "reward_mean": best["reward_mean"],
            "reward_ci": best["reward_ci"],
            "spent_mean": best["spent_mean"],
            "spent_ci": best["spent_ci"],
            "tau_mean": best["tau_mean"],
            "tau_ci": best["tau_ci"],
        })

        plot_rho.append(rho)
        plot_pd_reward.append(pd_reward_m)
        plot_pd_reward_ci.append(pd_reward_ci)
        plot_best_reward.append(best["reward_mean"])
        plot_best_reward_ci.append(best["reward_ci"])
        plot_best_gamma.append(best["gamma"])
        plot_best_spent.append(best["spent_mean"])
        plot_best_spent_ci.append(best["spent_ci"])

    tex_path = tab_dir / "budget_sweep.tex"
    write_budget_sweep_tabular(tex_path, final_rows_for_tex)
    print("Wrote:", tex_path)

    summary_path = tab_dir / "budget_sweep_summary.csv"
    write_csv(summary_path, final_rows_for_tex)
    print("Wrote:", summary_path)

    plt.figure()
    plt.errorbar(plot_rho, plot_pd_reward, yerr=plot_pd_reward_ci, marker="o", capsize=3, label="PD-LinUCB")
    plt.errorbar(
        plot_rho,
        plot_best_reward,
        yerr=plot_best_reward_ci,
        marker="s",
        capsize=3,
        label="Best CostNormUCB[sub]",
    )
    plt.xlabel(r"budget ratio $\rho=B/T$")
    plt.ylabel("total reward (mean ± 95% CI)")
    plt.title("Budget sweep: reward vs rho")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "budget_sweep_reward.png", dpi=220)
    plt.close()

    plt.figure()
    plt.plot(plot_rho, plot_best_gamma, marker="o")
    plt.xlabel(r"budget ratio $\rho=B/T$")
    plt.ylabel(r"selected $\gamma^\star$")
    plt.title(r"Budget sweep: $\gamma^\star(\rho)$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "budget_sweep_gamma_star.png", dpi=220)
    plt.close()

    plt.figure()
    plt.errorbar(plot_rho, plot_best_spent, yerr=plot_best_spent_ci, marker="o", capsize=3)
    plt.xlabel(r"budget ratio $\rho=B/T$")
    plt.ylabel("spent/B (mean ± 95% CI)")
    plt.title("Budget sweep: best baseline budget utilization")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "budget_sweep_spent.png", dpi=220)
    plt.close()

    print("Saved plots to:", fig_dir)


if __name__ == "__main__":
    main()
