# src/eval/run_compare_baselines.py
from __future__ import annotations

# ---- allow running as a script from repo root (python src/eval/run_compare_baselines.py) ----
import sys
from pathlib import Path
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt

from src.env.sim_bandit_env import SimBanditEnv
from src.algos.linucb_pd_delayed import DisjointLinUCB, PrimalDualLinUCB
from src.algos.cost_normalized_ucb import CostNormalizedDisjointUCB
from src.algos.context_free_bwk import ContextFreePrimalDualBwK
from src.eval.runner_utils import (
    RunResult,
    mean_ci,
    run_context_free_pd_delayed,
    run_contextual_delayed,
    t_crit_975,
)


def append_csv(path: str, row: dict):
    p = Path(path)
    newfile = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if newfile:
            w.writeheader()
        w.writerow(row)


def write_main_ci_tabular(tex_path: Path, rows: list[dict]):
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\toprule\n")
        f.write("Method & Reward (mean) & 95\\% CI & Spent$/B$ (mean) & 95\\% CI & $\\\\tau$ (mean) & 95\\% CI \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(
                f"{r['method']} & "
                f"{r['reward_mean']:.1f} & {r['reward_ci']:.1f} & "
                f"{r['spent_mean']:.6f} & {r['spent_ci']:.6f} & "
                f"{r['tau_mean']:.1f} & {r['tau_ci']:.1f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def plot_mean_ci(ax, x: np.ndarray, curves: np.ndarray, label: str, color=None):
    n_runs, _ = curves.shape
    mean = curves.mean(axis=0)
    if n_runs > 1:
        std = curves.std(axis=0, ddof=1)
        ci = t_crit_975(n_runs - 1) * std / sqrt(n_runs)
    else:
        ci = np.zeros_like(mean)

    ax.plot(x, mean, label=label, color=color, linewidth=1.8)
    ax.fill_between(x, mean - ci, mean + ci, alpha=0.18, color=color)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap_dir", type=str, required=True)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--budget_ratio", type=float, default=0.7)
    ap.add_argument("--stop_at_budget", action="store_true")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=123)
    ap.add_argument("--env_seed", type=int, default=0, help="Seed for building env_base")

    # LinUCB / PD-LinUCB
    ap.add_argument("--alpha_lin", type=float, default=1.0)
    ap.add_argument("--alpha_pd", type=float, default=1.5)
    ap.add_argument("--eta_pd", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0)

    # Cost-normalized UCB
    ap.add_argument("--alpha_cnu", type=float, default=1.0)
    ap.add_argument("--mode_cnu", type=str, default="ratio", choices=["ratio", "sub"])
    ap.add_argument("--gamma_cnu", type=float, default=1.0)
    ap.add_argument("--eps_cnu", type=float, default=1e-3)

    # Context-free PD-BwK
    ap.add_argument("--alpha_cf", type=float, default=1.0)
    ap.add_argument("--eta_cf", type=float, default=0.05)

    # output
    ap.add_argument("--artifact_dir", type=str, default="paper_artifacts")
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--log_csv", type=str, default="")
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    fig_dir = artifact_dir / "figures"
    tab_dir = artifact_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    env_base = SimBanditEnv.from_memmap_dir(args.memmap_dir, seed=args.env_seed, ridge_lambda=1.0, cost_mode="lin")
    seeds = [int(args.seed0 + i) for i in range(int(args.n_seeds))]

    metrics = {
        "lin": {"reward": [], "spent": [], "tau": [], "cum_r": [], "cum_c": []},
        "pd": {"reward": [], "spent": [], "tau": [], "cum_r": [], "cum_c": []},
        "cnu": {"reward": [], "spent": [], "tau": [], "cum_r": [], "cum_c": []},
        "cf": {"reward": [], "spent": [], "tau": [], "cum_r": [], "cum_c": []},
    }
    example: dict[str, RunResult] = {}

    for si, seed in enumerate(seeds):
        rng_ctx = np.random.default_rng(seed)
        X_seq = env_base.sample_contexts(args.T, rng=rng_ctx)

        env_lin = env_base.clone(seed=seed + 10_001)
        env_pd = env_base.clone(seed=seed + 10_002)
        env_cnu = env_base.clone(seed=seed + 10_003)
        env_cf = env_base.clone(seed=seed + 10_004)

        lin = DisjointLinUCB(env_base.K, env_base.d, alpha=args.alpha_lin, lam=args.lam, seed=seed + 101)
        pd = PrimalDualLinUCB(
            env_base.K,
            env_base.d,
            costs=env_base.costs,
            alpha=args.alpha_pd,
            lam=args.lam,
            eta=args.eta_pd,
            seed=seed + 102,
        )
        cnu = CostNormalizedDisjointUCB(
            env_base.K,
            env_base.d,
            costs=env_base.costs,
            alpha=args.alpha_cnu,
            lam=args.lam,
            mode=args.mode_cnu,
            gamma=args.gamma_cnu,
            eps=args.eps_cnu,
            seed=seed + 103,
        )
        cf = ContextFreePrimalDualBwK(
            env_base.K,
            costs=env_base.costs,
            alpha=args.alpha_cf,
            eta=args.eta_cf,
            seed=seed + 104,
        )

        res_lin = run_contextual_delayed(
            env_lin, X_seq, lin, args.budget_ratio, bool(args.stop_at_budget), is_primal_dual=False
        )
        res_pd = run_contextual_delayed(
            env_pd, X_seq, pd, args.budget_ratio, bool(args.stop_at_budget), is_primal_dual=True
        )
        res_cnu = run_contextual_delayed(
            env_cnu, X_seq, cnu, args.budget_ratio, bool(args.stop_at_budget), is_primal_dual=False
        )
        res_cf = run_context_free_pd_delayed(env_cf, X_seq, cf, args.budget_ratio, bool(args.stop_at_budget))

        for key, rr in [("lin", res_lin), ("pd", res_pd), ("cnu", res_cnu), ("cf", res_cf)]:
            metrics[key]["reward"].append(rr.reward)
            metrics[key]["spent"].append(rr.spent_ratio)
            metrics[key]["tau"].append(rr.t_stop)
            if not args.no_plots:
                metrics[key]["cum_r"].append(rr.cum_r)
                metrics[key]["cum_c"].append(rr.cum_c)

        if si == 0:
            example = {"lin": res_lin, "pd": res_pd, "cnu": res_cnu, "cf": res_cf}

        if args.log_csv:
            append_csv(
                args.log_csv,
                {
                    "seed": seed,
                    "T": args.T,
                    "budget_ratio": args.budget_ratio,
                    "stop_at_budget": int(args.stop_at_budget),
                    "alpha_lin": args.alpha_lin,
                    "alpha_pd": args.alpha_pd,
                    "eta_pd": args.eta_pd,
                    "alpha_cnu": args.alpha_cnu,
                    "mode_cnu": args.mode_cnu,
                    "gamma_cnu": args.gamma_cnu,
                    "eps_cnu": args.eps_cnu,
                    "alpha_cf": args.alpha_cf,
                    "eta_cf": args.eta_cf,
                    "lin_reward": res_lin.reward,
                    "lin_spent_ratio": res_lin.spent_ratio,
                    "lin_tau": res_lin.t_stop,
                    "pd_reward": res_pd.reward,
                    "pd_spent_ratio": res_pd.spent_ratio,
                    "pd_tau": res_pd.t_stop,
                    "cnu_reward": res_cnu.reward,
                    "cnu_spent_ratio": res_cnu.spent_ratio,
                    "cnu_tau": res_cnu.t_stop,
                    "cf_reward": res_cf.reward,
                    "cf_spent_ratio": res_cf.spent_ratio,
                    "cf_tau": res_cf.t_stop,
                },
            )

        print(
            f"done seed={seed}  lin={res_lin.reward:.1f}  pd={res_pd.reward:.1f}  "
            f"cnu={res_cnu.reward:.1f}  cf={res_cf.reward:.1f}"
        )

    def summarize_method(key: str, name: str):
        r_m, r_ci = mean_ci(metrics[key]["reward"])
        s_m, s_ci = mean_ci(metrics[key]["spent"])
        t_m, t_ci = mean_ci(metrics[key]["tau"])
        return {
            "method": name,
            "reward_mean": r_m,
            "reward_ci": r_ci,
            "spent_mean": s_m,
            "spent_ci": s_ci,
            "tau_mean": t_m,
            "tau_ci": t_ci,
        }

    rows = [
        summarize_method("lin", "LinUCB"),
        summarize_method("pd", "PD-LinUCB"),
        summarize_method("cnu", f"CostNormUCB[{args.mode_cnu}]"),
        summarize_method("cf", "CF-PD-BwK"),
    ]

    main_ci_path = tab_dir / "main_ci.tex"
    write_main_ci_tabular(main_ci_path, rows)
    print("Wrote:", main_ci_path)

    tables_dir = Path("tables")
    tables_dir.mkdir(parents=True, exist_ok=True)
    alias_path = tables_dir / "main_ci.tex"
    alias_path.write_text(
        "% This file is generated/overwritten by:\n"
        "%   python -m src.eval.run_compare_baselines --memmap_dir ... --T 5000 --budget_ratio 0.7 --stop_at_budget --n_seeds 10\n"
        "%\n"
        "% Tabular-only indirection:\n"
        "\\input{paper_artifacts/tables/main_ci.tex}\n",
        encoding="utf-8",
    )

    if args.no_plots:
        return

    def save_fig(basename: str, dpi: int = 200):
        plt.savefig(fig_dir / f"{basename}.png", dpi=dpi)
        if args.tag:
            plt.savefig(fig_dir / f"{basename}_{args.tag}.png", dpi=dpi)

    x = np.arange(1, args.T + 1, dtype=np.float64)
    B = float(args.budget_ratio) * float(args.T)
    ideal = (x / float(args.T)) * float(B)

    # ---------------- Figure 1: cumulative reward ----------------
    plt.figure()
    plt.plot(x, example["lin"].cum_r, label="LinUCB")
    plt.plot(x, example["pd"].cum_r, label="PD-LinUCB")
    plt.plot(x, example["cnu"].cum_r, label=f"CostNormUCB[{args.mode_cnu}]")
    plt.plot(x, example["cf"].cum_r, label="CF-PD-BwK")
    plt.title("Cumulative reward (example run, stop-at-budget)")
    plt.xlabel("t")
    plt.ylabel("reward")
    plt.xlim(1, args.T)
    plt.legend()
    plt.tight_layout()
    save_fig("baselines_cum_reward_full4_arm", dpi=220)
    plt.close()

    plt.figure()
    plt.plot(x, example["lin"].cum_c, label="LinUCB")
    plt.plot(x, example["pd"].cum_c, label="PD-LinUCB")
    plt.plot(x, example["cnu"].cum_c, label=f"CostNormUCB[{args.mode_cnu}]")
    plt.plot(x, example["cf"].cum_c, label="CF-PD-BwK")
    plt.axhline(B, linestyle="--", label="Budget B")
    plt.title("Cumulative cost (example run, stop-at-budget)")
    plt.xlabel("t")
    plt.ylabel("spent")
    plt.xlim(1, args.T)
    plt.ylim(0.0, 1.05 * B)
    plt.legend()
    plt.tight_layout()
    save_fig("baselines_cum_cost_full4_arm", dpi=220)
    plt.close()

    plt.figure()
    plt.plot(x, example["lin"].cum_c, label="LinUCB")
    plt.plot(x, example["pd"].cum_c, label="PD-LinUCB")
    plt.plot(x, example["cnu"].cum_c, label=f"CostNormUCB[{args.mode_cnu}]")
    plt.plot(x, example["cf"].cum_c, label="CF-PD-BwK")
    plt.plot(x, ideal, linestyle=":", label="Ideal spend (t/T)·B")
    plt.axhline(B, linestyle="--", label="Budget B")
    plt.title("Cumulative spend vs budget schedule (example run)")
    plt.xlabel("t")
    plt.ylabel("spent")
    plt.xlim(1, args.T)
    plt.ylim(0.0, 1.05 * B)
    plt.legend()
    plt.tight_layout()
    save_fig("baselines_spent_schedule", dpi=220)
    plt.close()

    def plot_running_avg_cost(res: RunResult, label: str):
        t_stop = int(res.t_stop)
        if t_stop <= 0:
            return
        t_axis = np.arange(1, t_stop + 1, dtype=np.float64)
        cbar = res.cum_c[:t_stop] / t_axis
        plt.plot(t_axis, cbar, label=label)

    plt.figure()
    plot_running_avg_cost(example["lin"], "LinUCB")
    plot_running_avg_cost(example["pd"], "PD-LinUCB")
    plot_running_avg_cost(example["cnu"], f"CostNormUCB[{args.mode_cnu}]")
    plot_running_avg_cost(example["cf"], "CF-PD-BwK")
    plt.axhline(float(args.budget_ratio), linestyle="--", label=r"$\rho=B/T$")
    plt.title("Running-average cost per step (example run)")
    plt.xlabel("t (until stop)")
    plt.ylabel(r"$\bar c_t = \mathrm{spent}_t / t$")
    plt.legend(ncol=2)
    plt.tight_layout()
    save_fig("avg_cost_per_step", dpi=240)
    plt.close()

    def stack(key: str, field: str) -> np.ndarray:
        return np.stack(metrics[key][field], axis=0).astype(np.float64)

    lin_r = stack("lin", "cum_r")
    pd_r = stack("pd", "cum_r")
    cnu_r = stack("cnu", "cum_r")
    cf_r = stack("cf", "cum_r")

    lin_c = stack("lin", "cum_c")
    pd_c = stack("pd", "cum_c")
    cnu_c = stack("cnu", "cum_c")
    cf_c = stack("cf", "cum_c")

    plt.figure()
    ax = plt.gca()
    plot_mean_ci(ax, x, lin_r, "LinUCB")
    plot_mean_ci(ax, x, pd_r, "PD-LinUCB")
    plot_mean_ci(ax, x, cnu_r, f"CostNormUCB[{args.mode_cnu}]")
    plot_mean_ci(ax, x, cf_r, "CF-PD-BwK")
    ax.set_title("Cumulative reward (mean ± 95% CI over seeds)")
    ax.set_xlabel("t")
    ax.set_ylabel("reward")
    ax.set_xlim(1, args.T)
    ax.legend()
    plt.tight_layout()
    save_fig("baselines_cum_reward_mean_ci", dpi=240)
    plt.close()

    plt.figure()
    ax = plt.gca()
    plot_mean_ci(ax, x, lin_c, "LinUCB")
    plot_mean_ci(ax, x, pd_c, "PD-LinUCB")
    plot_mean_ci(ax, x, cnu_c, f"CostNormUCB[{args.mode_cnu}]")
    plot_mean_ci(ax, x, cf_c, "CF-PD-BwK")
    ax.axhline(B, linestyle="--", label="Budget B", color="black", linewidth=1.0)
    ax.set_title("Cumulative cost (mean ± 95% CI over seeds)")
    ax.set_xlabel("t")
    ax.set_ylabel("spent")
    ax.set_xlim(1, args.T)
    ax.set_ylim(0.0, 1.05 * B)
    ax.legend()
    plt.tight_layout()
    save_fig("baselines_cum_cost_mean_ci", dpi=240)
    plt.close()

    print("Saved figures to:", fig_dir)


if __name__ == "__main__":
    main()
