# src/eval/run_compare_baselines.py
from __future__ import annotations

# ---- allow running as a script from repo root (python src/eval/run_compare_baselines.py) ----
import sys
from pathlib import Path
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]  # .../paper_cbwk_delays
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv

import numpy as np
import matplotlib.pyplot as plt

from src.env.sim_bandit_env import SimBanditEnv
from src.algos.linucb_pd_delayed import DisjointLinUCB, PrimalDualLinUCB
from src.algos.cost_normalized_ucb import CostNormalizedDisjointUCB
from src.algos.context_free_bwk import ContextFreePrimalDualBwK


def append_csv(path: str, row: dict):
    p = Path(path)
    newfile = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if newfile:
            w.writeheader()
        w.writerow(row)


def run_contextual_algo(
    env: SimBanditEnv,
    X_seq: np.ndarray,
    algo,
    budget_ratio: float,
    stop_at_budget: bool,
    is_primal_dual: bool,
):
    """
    Contextual runner with:
      - OTF design update: update_design() right after action is committed
      - reward-only delayed updates: update_reward() on feedback arrival
      - primal-dual uses dynamic b_t computed from remaining budget
    """
    T = len(X_seq)
    B = float(budget_ratio) * T

    pending: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    cum_r = np.zeros(T, dtype=np.float64)
    cum_c = np.zeros(T, dtype=np.float64)

    total_r = 0.0
    total_c = 0.0
    t_stop = T

    for t in range(T):
        # apply arrived reward feedback
        for (a_upd, x_upd, r_upd) in pending.pop(t, []):
            algo.update_reward(a_upd, x_upd, r_upd)

        x = X_seq[t]
        a = algo.select(x)
        c = float(env.costs[a])

        # stop-at-budget check BEFORE any side effects
        if stop_at_budget and (total_c + c > B):
            t_stop = t
            cum_r[t:] = total_r
            cum_c[t:] = total_c
            break

        # commit action: primal-dual dual update + design update
        if is_primal_dual:
            H = T - t  # remaining steps including current
            b_t = (B - total_c) / max(H, 1)
            algo.update_dual(c, b_t)

        algo.update_design(a, x)

        # environment step
        r, _, dly = env.step(x, a)

        total_c += c
        total_r += r
        cum_r[t] = total_r
        cum_c[t] = total_c

        # schedule reward update
        dly = int(dly)
        if dly <= 0:
            algo.update_reward(a, x, r)
        else:
            t_due = t + dly
            if t_due < T:
                pending.setdefault(t_due, []).append((a, x, r))
            # else: feedback arrives after horizon -> not observed in this run

    return {
        "cum_r": cum_r,
        "cum_c": cum_c,
        "reward": float(cum_r[t_stop - 1]) if t_stop > 0 else 0.0,
        "cost": float(cum_c[t_stop - 1]) if t_stop > 0 else 0.0,
        "t_stop": int(t_stop),
    }


def run_context_free_pd(
    env: SimBanditEnv,
    X_seq: np.ndarray,
    algo: ContextFreePrimalDualBwK,
    budget_ratio: float,
    stop_at_budget: bool,
):
    """
    Context-free runner:
      - delayed reward updates via pending
      - dynamic b_t from remaining budget
    """
    T = len(X_seq)
    B = float(budget_ratio) * T

    pending: dict[int, list[tuple[int, float]]] = {}
    cum_r = np.zeros(T, dtype=np.float64)
    cum_c = np.zeros(T, dtype=np.float64)

    total_r = 0.0
    total_c = 0.0
    t_stop = T

    for t in range(T):
        for (a_upd, r_upd) in pending.pop(t, []):
            algo.update(a_upd, r_upd)

        a = algo.select(t)
        c = float(env.costs[a])

        if stop_at_budget and (total_c + c > B):
            t_stop = t
            cum_r[t:] = total_r
            cum_c[t:] = total_c
            break

        # dual update with remaining-budget target
        H = T - t
        b_t = (B - total_c) / max(H, 1)
        algo.update_dual(c, b_t)

        x = X_seq[t]
        r, _, dly = env.step(x, a)

        total_c += c
        total_r += r
        cum_r[t] = total_r
        cum_c[t] = total_c

        dly = int(dly)
        if dly <= 0:
            algo.update(a, r)
        else:
            t_due = t + dly
            if t_due < T:
                pending.setdefault(t_due, []).append((a, r))

    return {
        "cum_r": cum_r,
        "cum_c": cum_c,
        "reward": float(cum_r[t_stop - 1]) if t_stop > 0 else 0.0,
        "cost": float(cum_c[t_stop - 1]) if t_stop > 0 else 0.0,
        "t_stop": int(t_stop),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="data/processed/criteo_attrib_k50_d64_delayed.npz")
    ap.add_argument("--memmap_dir", type=str, default="", help="Load env from memmaps dir instead of --npz")
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--budget_ratio", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--stop_at_budget", action="store_true")

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
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--log_csv", type=str, default="")
    args = ap.parse_args()

    # Load env
    if args.memmap_dir:
        env = SimBanditEnv.from_memmap_dir(args.memmap_dir, seed=args.seed, ridge_lambda=1.0, cost_mode="lin")
        dataset_id = args.memmap_dir
    else:
        raise RuntimeError("Use --memmap_dir for this pipeline (real costs/delays).")

    X_seq = env.sample_contexts(args.T)

    B = float(args.budget_ratio) * float(args.T)

    # instantiate methods
    lin = DisjointLinUCB(env.K, env.d, alpha=args.alpha_lin, lam=args.lam, seed=args.seed + 1)

    pd = PrimalDualLinUCB(
        env.K, env.d,
        costs=env.costs,
        alpha=args.alpha_pd,
        lam=args.lam,
        eta=args.eta_pd,
        seed=args.seed + 2
    )

    cnu = CostNormalizedDisjointUCB(
        env.K, env.d,
        costs=env.costs,
        alpha=args.alpha_cnu,
        lam=args.lam,
        mode=args.mode_cnu,
        gamma=args.gamma_cnu,
        eps=args.eps_cnu,
        seed=args.seed + 3
    )

    cf = ContextFreePrimalDualBwK(
        env.K,
        costs=env.costs,
        alpha=args.alpha_cf,
        eta=args.eta_cf,
        seed=args.seed + 4
    )

    # run methods
    res_lin = run_contextual_algo(env, X_seq, lin, args.budget_ratio, args.stop_at_budget, is_primal_dual=False)
    res_pd  = run_contextual_algo(env, X_seq, pd,  args.budget_ratio, args.stop_at_budget, is_primal_dual=True)
    res_cnu = run_contextual_algo(env, X_seq, cnu, args.budget_ratio, args.stop_at_budget, is_primal_dual=False)
    res_cf  = run_context_free_pd(env, X_seq, cf,  args.budget_ratio, args.stop_at_budget)

    def line(name, rr):
        return f"{name:18s} reward {rr['reward']:.1f}  cost {rr['cost']:.2f}  t_stop {rr['t_stop']}"

    print("=== Summary (4 methods) ===")
    print("Dataset:", dataset_id)
    print(f"B={B:.1f} (budget_ratio={args.budget_ratio}), stop_at_budget={bool(args.stop_at_budget)}, T={args.T}")
    print(line("LinUCB", res_lin))
    print(line("PD-LinUCB", res_pd))
    print(line(f"CostNormUCB[{args.mode_cnu}]", res_cnu))
    print(line("CF-PD-BwK", res_cf))

    # log csv
    if args.log_csv:
        row = {
            "dataset": dataset_id,
            "T": args.T,
            "budget_ratio": args.budget_ratio,
            "B": B,
            "stop_at_budget": int(args.stop_at_budget),
            "seed": args.seed,

            "alpha_lin": args.alpha_lin,
            "alpha_pd": args.alpha_pd,
            "eta_pd": args.eta_pd,
            "alpha_cnu": args.alpha_cnu,
            "mode_cnu": args.mode_cnu,
            "gamma_cnu": args.gamma_cnu,
            "eps_cnu": args.eps_cnu,
            "alpha_cf": args.alpha_cf,
            "eta_cf": args.eta_cf,

            "lin_reward": res_lin["reward"], "lin_cost": res_lin["cost"], "lin_t_stop": res_lin["t_stop"],
            "pd_reward": res_pd["reward"], "pd_cost": res_pd["cost"], "pd_t_stop": res_pd["t_stop"],
            "cnu_reward": res_cnu["reward"], "cnu_cost": res_cnu["cost"], "cnu_t_stop": res_cnu["t_stop"],
            "cf_reward": res_cf["reward"], "cf_cost": res_cf["cost"], "cf_t_stop": res_cf["t_stop"],
        }

        eps = 1e-9
        row["lin_spent_ratio"] = row["lin_cost"] / row["B"]
        row["pd_spent_ratio"] = row["pd_cost"] / row["B"]
        row["cnu_spent_ratio"] = row["cnu_cost"] / row["B"]
        row["cf_spent_ratio"] = row["cf_cost"] / row["B"]

        row["lin_reward_per_cost"] = row["lin_reward"] / max(row["lin_cost"], eps)
        row["pd_reward_per_cost"] = row["pd_reward"] / max(row["pd_cost"], eps)
        row["cnu_reward_per_cost"] = row["cnu_reward"] / max(row["cnu_cost"], eps)
        row["cf_reward_per_cost"] = row["cf_reward"] / max(row["cf_cost"], eps)

        append_csv(args.log_csv, row)

    if args.no_plots:
        return

    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    def save_base_and_tag(basename: str, dpi: int = 200):
        """
        Always save basename.png.
        If --tag provided, also save basename_{tag}.png (so article can reference a fixed tag name).
        """
        plt.savefig(outdir / f"{basename}.png", dpi=dpi)
        if args.tag:
            plt.savefig(outdir / f"{basename}_{args.tag}.png", dpi=dpi)

    x = np.arange(1, args.T + 1, dtype=np.float64)  # 1..T
    ideal = (x / float(args.T)) * float(B)          # (t/T)*B

    # ---------------- Figure 1: cumulative reward ----------------
    plt.figure()
    plt.plot(x, res_lin["cum_r"], label="LinUCB")
    plt.plot(x, res_pd["cum_r"], label="PD-LinUCB")
    plt.plot(x, res_cnu["cum_r"], label=f"CostNormUCB[{args.mode_cnu}]")
    plt.plot(x, res_cf["cum_r"], label="CF-PD-BwK")
    plt.title("Cumulative reward (stop-at-budget)")
    plt.xlabel("t")
    plt.ylabel("reward")
    plt.xlim(1, args.T)
    plt.legend()
    plt.tight_layout()
    save_base_and_tag("baselines_cum_reward_full4_arm", dpi=200)
    plt.close()

    # ---------------- Figure 2: cumulative cost ----------------
    plt.figure()
    plt.plot(x, res_lin["cum_c"], label="LinUCB")
    plt.plot(x, res_pd["cum_c"], label="PD-LinUCB")
    plt.plot(x, res_cnu["cum_c"], label=f"CostNormUCB[{args.mode_cnu}]")
    plt.plot(x, res_cf["cum_c"], label="CF-PD-BwK")
    plt.axhline(B, linestyle="--", label="Budget B")
    plt.title("Cumulative cost (stop-at-budget)")
    plt.xlabel("t")
    plt.ylabel("spent")
    plt.xlim(1, args.T)
    plt.ylim(0.0, 1.05 * B)
    plt.legend()
    plt.tight_layout()
    save_base_and_tag("baselines_cum_cost_full4_arm", dpi=200)
    plt.close()

    # ---------------- Figure 3: cumulative spend + ideal schedule ----------------
    plt.figure()
    plt.plot(x, res_lin["cum_c"], label="LinUCB")
    plt.plot(x, res_pd["cum_c"], label="PD-LinUCB")
    plt.plot(x, res_cnu["cum_c"], label=f"CostNormUCB[{args.mode_cnu}]")
    plt.plot(x, res_cf["cum_c"], label="CF-PD-BwK")
    plt.plot(x, ideal, linestyle=":", label="Ideal spend (t/T)·B")
    plt.axhline(B, linestyle="--", label="Budget B")
    plt.title("Cumulative spend vs budget schedule (stop-at-budget)")
    plt.xlabel("t")
    plt.ylabel("spent")
    plt.xlim(1, args.T)
    plt.ylim(0.0, 1.05 * B)
    plt.legend()
    plt.tight_layout()
    save_base_and_tag("baselines_spent_schedule", dpi=200)
    plt.close()

    # ---------------- Figure 4: running-average cost per step ----------------
    def plot_running_avg_cost(res, label: str):
        t_stop = int(res["t_stop"])
        if t_stop <= 0:
            return
        t_axis = np.arange(1, t_stop + 1, dtype=np.float64)
        cbar = res["cum_c"][:t_stop] / t_axis
        plt.plot(t_axis, cbar, label=label)

    plt.figure()
    plot_running_avg_cost(res_lin, "LinUCB")
    plot_running_avg_cost(res_pd, "PD-LinUCB")
    plot_running_avg_cost(res_cnu, f"CostNormUCB[{args.mode_cnu}]")
    plot_running_avg_cost(res_cf, "CF-PD-BwK")
    plt.axhline(float(args.budget_ratio), linestyle="--", label=r"$\rho=B/T$")
    plt.title("Running-average cost per step (stop-at-budget)")
    plt.xlabel("t (until stop-at-budget)")
    plt.ylabel(r"$\bar c_t = \mathrm{spent}_t / t$")
    plt.legend(ncol=2)
    plt.tight_layout()
    # base: avg_cost_per_step.png; tagged: avg_cost_per_step_rho0.7.png (if --tag rho0.7)
    save_base_and_tag("avg_cost_per_step", dpi=220)
    plt.close()


if __name__ == "__main__":
    main()