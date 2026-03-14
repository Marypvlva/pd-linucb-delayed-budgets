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

import src.eval.mpl_setup  # noqa: F401
import matplotlib.pyplot as plt

from src.algos.cost_normalized_ucb import CostNormalizedDisjointUCB
from src.algos.linucb_pd_delayed import PrimalDualLinUCB
from src.algos.logistic_ucb_delayed import CostNormalizedDisjointLogisticUCB, PrimalDualLogisticUCB
from src.env.sim_bandit_env import SimBanditEnv
from src.eval.runner_utils import mean_ci, run_contextual_delayed


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Empty rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_budget_sweep_tabular(tex_path: Path, rows: list[dict]):
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{l l r r r r r r}\n")
        f.write("\\toprule\n")
        f.write(
            "$\\\\rho$ & Method & Reward (mean) & 95\\% CI & Spent$/B$ (mean) & "
            "95\\% CI & $\\\\tau$ (mean) & 95\\% CI \\\\\n"
        )
        f.write("\\midrule\n")
        for row in rows:
            f.write(
                f"{row['rho']:.2f} & {row['method']} & "
                f"{row['reward_mean']:.1f} & {row['reward_ci']:.1f} & "
                f"{row['spent_mean']:.6f} & {row['spent_ci']:.6f} & "
                f"{row['tau_mean']:.1f} & {row['tau_ci']:.1f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def make_pd_algo(env: SimBanditEnv, args, seed: int):
    if args.policy_model == "logistic":
        return PrimalDualLogisticUCB(
            env.K,
            env.d,
            costs=env.costs,
            alpha=float(args.alpha_pd),
            lam=float(args.lam),
            eta=float(args.eta_pd),
            seed=seed,
        )
    return PrimalDualLinUCB(
        env.K,
        env.d,
        costs=env.costs,
        alpha=float(args.alpha_pd),
        lam=float(args.lam),
        eta=float(args.eta_pd),
        seed=seed,
    )


def make_costnorm_algo(env: SimBanditEnv, args, gamma: float, seed: int):
    if args.policy_model == "logistic":
        return CostNormalizedDisjointLogisticUCB(
            env.K,
            env.d,
            costs=env.costs,
            alpha=float(args.alpha_cnu),
            lam=float(args.lam),
            mode="sub",
            gamma=float(gamma),
            eps=1e-3,
            seed=seed,
        )
    return CostNormalizedDisjointUCB(
        env.K,
        env.d,
        costs=env.costs,
        alpha=float(args.alpha_cnu),
        lam=float(args.lam),
        mode="sub",
        gamma=float(gamma),
        eps=1e-3,
        seed=seed,
    )


def resolve_tune_split(args, env_eval_base: SimBanditEnv) -> str:
    if args.tune_context_split != "auto":
        return str(args.tune_context_split)
    eval_effective = str(env_eval_base.meta.get("context_split", args.context_split))
    if eval_effective == "test" and int(env_eval_base.meta.get("n_train", 0)) > 0:
        return "train"
    return eval_effective


def choose_best_gamma(gamma_summ: list[dict], min_spent_ratio: float) -> dict:
    feasible = [row for row in gamma_summ if row["spent_mean"] >= float(min_spent_ratio)]
    if feasible:
        return max(feasible, key=lambda row: row["reward_mean"])
    return max(gamma_summ, key=lambda row: (row["spent_mean"], row["reward_mean"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap_dir", type=str, required=True)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--stop_at_budget", action="store_true")
    ap.add_argument("--budgets", type=str, default="0.40,0.55,0.70,0.85")
    ap.add_argument("--gammas", type=str, default="0,0.1,0.3,1,2,3,5,10")
    ap.add_argument("--min_spent_ratio", type=float, default=0.99)
    ap.add_argument("--n_seeds", type=int, default=10, help="Number of held-out evaluation seeds.")
    ap.add_argument("--seed0", type=int, default=123, help="First held-out evaluation seed.")
    ap.add_argument("--n_tune_seeds", type=int, default=10, help="Number of tuning seeds.")
    ap.add_argument("--tune_seed0", type=int, default=10123, help="First tuning seed.")
    ap.add_argument("--env_seed", type=int, default=0)
    ap.add_argument(
        "--context_split",
        type=str,
        default="auto",
        choices=["auto", "all", "train", "test"],
        help="Which held-out context split to evaluate on. 'auto' uses test when available.",
    )
    ap.add_argument(
        "--tune_context_split",
        type=str,
        default="auto",
        choices=["auto", "all", "train", "test"],
        help="Which context split to tune gamma on. 'auto' uses train when eval split is test.",
    )
    ap.add_argument(
        "--reward_model",
        type=str,
        default="auto",
        choices=["auto", "linear_clip", "logistic"],
        help="Reward model used by the simulator.",
    )
    ap.add_argument(
        "--policy_model",
        type=str,
        default="linear",
        choices=["linear", "logistic"],
        help="Contextual learner family for PD and CostNorm baselines.",
    )
    ap.add_argument("--alpha_pd", type=float, default=1.5)
    ap.add_argument("--eta_pd", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alpha_cnu", type=float, default=1.0)
    ap.add_argument("--artifact_dir", type=str, default="paper_artifacts")
    args = ap.parse_args()

    budgets = [float(x) for x in args.budgets.split(",") if x.strip()]
    gammas = [float(x) for x in args.gammas.split(",") if x.strip()]
    if not budgets:
        raise RuntimeError("Empty --budgets")
    if not gammas:
        raise RuntimeError("Empty --gammas")

    artifact_dir = Path(args.artifact_dir)
    fig_dir = artifact_dir / "figures"
    tab_dir = artifact_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    env_eval_base = SimBanditEnv.from_memmap_dir(
        args.memmap_dir,
        seed=args.env_seed,
        ridge_lambda=1.0,
        cost_mode="lin",
        context_split=args.context_split,
        reward_model=args.reward_model,
    )
    tune_context_split = resolve_tune_split(args, env_eval_base)
    env_tune_base = SimBanditEnv.from_memmap_dir(
        args.memmap_dir,
        seed=args.env_seed + 1,
        ridge_lambda=1.0,
        cost_mode="lin",
        context_split=tune_context_split,
        reward_model=args.reward_model,
    )

    eval_seeds = [int(args.seed0 + i) for i in range(int(args.n_seeds))]
    tune_seeds = [int(args.tune_seed0 + i) for i in range(int(args.n_tune_seeds))]
    X_eval_by_seed = {seed: env_eval_base.sample_contexts(args.T, rng=np.random.default_rng(seed)) for seed in eval_seeds}
    X_tune_by_seed = {seed: env_tune_base.sample_contexts(args.T, rng=np.random.default_rng(seed)) for seed in tune_seeds}

    pd_name = "PD-LogisticUCB" if args.policy_model == "logistic" else "PD-LinUCB"
    cnu_name = "CostNormLogisticUCB[sub]" if args.policy_model == "logistic" else "CostNormUCB[sub]"

    tuning_raw: list[dict] = []
    tuning_summary: list[dict] = []
    eval_raw: list[dict] = []
    final_rows_for_tex: list[dict] = []
    plot_rho = []
    plot_pd_reward = []
    plot_pd_reward_ci = []
    plot_best_reward = []
    plot_best_reward_ci = []
    plot_best_gamma = []
    plot_best_spent = []
    plot_best_spent_ci = []

    for rho in budgets:
        gamma_summ = []
        for gamma in gammas:
            rewards = []
            spent_ratios = []
            taus = []
            for seed in tune_seeds:
                X_seq = X_tune_by_seed[seed]
                env = env_tune_base.clone(seed=seed + 10_000 + int(round(rho * 1000)) + int(round(gamma * 1000)))
                algo = make_costnorm_algo(
                    env,
                    args,
                    gamma=gamma,
                    seed=seed + 20_000 + int(round(rho * 1000)) + int(round(gamma * 1000)),
                )
                res = run_contextual_delayed(
                    env,
                    X_seq,
                    algo,
                    budget_ratio=float(rho),
                    stop_at_budget=bool(args.stop_at_budget),
                    is_primal_dual=False,
                )
                tuning_raw.append({
                    "rho": float(rho),
                    "seed": int(seed),
                    "context_split": tune_context_split,
                    "gamma": float(gamma),
                    "reward": float(res.reward),
                    "spent_ratio": float(res.spent_ratio),
                    "tau": int(res.t_stop),
                })
                rewards.append(float(res.reward))
                spent_ratios.append(float(res.spent_ratio))
                taus.append(float(res.t_stop))

            reward_m, reward_ci = mean_ci(rewards)
            spent_m, spent_ci = mean_ci(spent_ratios)
            tau_m, tau_ci = mean_ci(taus)
            gamma_summ.append({
                "rho": float(rho),
                "gamma": float(gamma),
                "reward_mean": reward_m,
                "reward_ci": reward_ci,
                "spent_mean": spent_m,
                "spent_ci": spent_ci,
                "tau_mean": tau_m,
                "tau_ci": tau_ci,
                "context_split": tune_context_split,
            })

        best = choose_best_gamma(gamma_summ, min_spent_ratio=float(args.min_spent_ratio))
        for row in gamma_summ:
            tuning_summary.append({
                **row,
                "selected": int(float(row["gamma"]) == float(best["gamma"])),
            })

        pd_rewards = []
        pd_spent = []
        pd_taus = []
        best_rewards = []
        best_spent = []
        best_taus = []

        for seed in eval_seeds:
            X_seq = X_eval_by_seed[seed]

            env_pd = env_eval_base.clone(seed=seed + 30_000 + int(round(rho * 1000)))
            pd = make_pd_algo(env_pd, args, seed=seed + 40_000 + int(round(rho * 1000)))
            res_pd = run_contextual_delayed(
                env_pd,
                X_seq,
                pd,
                budget_ratio=float(rho),
                stop_at_budget=bool(args.stop_at_budget),
                is_primal_dual=True,
            )
            eval_raw.append({
                "rho": float(rho),
                "seed": int(seed),
                "context_split": str(env_eval_base.meta.get("context_split", args.context_split)),
                "method": pd_name,
                "gamma": "",
                "reward": float(res_pd.reward),
                "spent_ratio": float(res_pd.spent_ratio),
                "tau": int(res_pd.t_stop),
            })
            pd_rewards.append(float(res_pd.reward))
            pd_spent.append(float(res_pd.spent_ratio))
            pd_taus.append(float(res_pd.t_stop))

            env_cnu = env_eval_base.clone(
                seed=seed + 50_000 + int(round(rho * 1000)) + int(round(best["gamma"] * 1000))
            )
            cnu = make_costnorm_algo(
                env_cnu,
                args,
                gamma=float(best["gamma"]),
                seed=seed + 60_000 + int(round(rho * 1000)) + int(round(best["gamma"] * 1000)),
            )
            res_cnu = run_contextual_delayed(
                env_cnu,
                X_seq,
                cnu,
                budget_ratio=float(rho),
                stop_at_budget=bool(args.stop_at_budget),
                is_primal_dual=False,
            )
            eval_raw.append({
                "rho": float(rho),
                "seed": int(seed),
                "context_split": str(env_eval_base.meta.get("context_split", args.context_split)),
                "method": cnu_name,
                "gamma": float(best["gamma"]),
                "reward": float(res_cnu.reward),
                "spent_ratio": float(res_cnu.spent_ratio),
                "tau": int(res_cnu.t_stop),
            })
            best_rewards.append(float(res_cnu.reward))
            best_spent.append(float(res_cnu.spent_ratio))
            best_taus.append(float(res_cnu.t_stop))

        pd_reward_m, pd_reward_ci = mean_ci(pd_rewards)
        pd_spent_m, pd_spent_ci = mean_ci(pd_spent)
        pd_tau_m, pd_tau_ci = mean_ci(pd_taus)
        best_reward_m, best_reward_ci = mean_ci(best_rewards)
        best_spent_m, best_spent_ci = mean_ci(best_spent)
        best_tau_m, best_tau_ci = mean_ci(best_taus)

        final_rows_for_tex.append({
            "rho": float(rho),
            "method": pd_name,
            "reward_mean": pd_reward_m,
            "reward_ci": pd_reward_ci,
            "spent_mean": pd_spent_m,
            "spent_ci": pd_spent_ci,
            "tau_mean": pd_tau_m,
            "tau_ci": pd_tau_ci,
        })
        final_rows_for_tex.append({
            "rho": float(rho),
            "method": f"{cnu_name} ($\\gamma^\\star={best['gamma']}$)",
            "reward_mean": best_reward_m,
            "reward_ci": best_reward_ci,
            "spent_mean": best_spent_m,
            "spent_ci": best_spent_ci,
            "tau_mean": best_tau_m,
            "tau_ci": best_tau_ci,
        })

        plot_rho.append(float(rho))
        plot_pd_reward.append(pd_reward_m)
        plot_pd_reward_ci.append(pd_reward_ci)
        plot_best_reward.append(best_reward_m)
        plot_best_reward_ci.append(best_reward_ci)
        plot_best_gamma.append(float(best["gamma"]))
        plot_best_spent.append(best_spent_m)
        plot_best_spent_ci.append(best_spent_ci)

        print(
            f"done rho={rho:.2f}  gamma*={best['gamma']}  "
            f"tune_split={tune_context_split}  eval_split={env_eval_base.meta.get('context_split')}"
        )

    write_csv(tab_dir / "budget_sweep_tuning_raw.csv", tuning_raw)
    write_csv(tab_dir / "budget_sweep_tuning_summary.csv", tuning_summary)
    write_csv(tab_dir / "budget_sweep_eval_raw.csv", eval_raw)
    write_csv(tab_dir / "budget_sweep_summary.csv", final_rows_for_tex)
    write_budget_sweep_tabular(tab_dir / "budget_sweep.tex", final_rows_for_tex)

    plt.figure()
    plt.errorbar(plot_rho, plot_pd_reward, yerr=plot_pd_reward_ci, marker="o", capsize=3, label=pd_name)
    plt.errorbar(plot_rho, plot_best_reward, yerr=plot_best_reward_ci, marker="s", capsize=3, label=f"Best {cnu_name}")
    plt.xlabel(r"budget ratio $\rho=B/T$")
    plt.ylabel("total reward (mean ± 95% CI)")
    plt.title("Budget sweep: held-out reward vs rho")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "budget_sweep_reward.png", dpi=220)
    plt.close()

    plt.figure()
    plt.plot(plot_rho, plot_best_gamma, marker="o")
    plt.xlabel(r"budget ratio $\rho=B/T$")
    plt.ylabel(r"selected $\gamma^\star$")
    plt.title(f"Budget sweep: gamma tuned on {tune_context_split}, evaluated held-out")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "budget_sweep_gamma_star.png", dpi=220)
    plt.close()

    plt.figure()
    plt.errorbar(plot_rho, plot_best_spent, yerr=plot_best_spent_ci, marker="o", capsize=3)
    plt.xlabel(r"budget ratio $\rho=B/T$")
    plt.ylabel("spent/B (mean ± 95% CI)")
    plt.title("Budget sweep: held-out budget utilization of tuned baseline")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "budget_sweep_spent.png", dpi=220)
    plt.close()

    print("Wrote:", tab_dir / "budget_sweep_tuning_raw.csv")
    print("Wrote:", tab_dir / "budget_sweep_tuning_summary.csv")
    print("Wrote:", tab_dir / "budget_sweep_eval_raw.csv")
    print("Wrote:", tab_dir / "budget_sweep_summary.csv")
    print("Wrote:", tab_dir / "budget_sweep.tex")
    print("Saved plots to:", fig_dir)


if __name__ == "__main__":
    main()
