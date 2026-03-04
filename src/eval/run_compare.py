# src/eval/run_compare.py
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.env.sim_bandit_env import SimBanditEnv
from src.algos.linucb_pd_delayed import DisjointLinUCB, PrimalDualLinUCB


def run_linucb(
    env: SimBanditEnv,
    X_seq: np.ndarray,
    alpha: float,
    lam: float,
    seed: int,
    budget_ratio: float | None,
    stop_at_budget: bool,
):
    T = len(X_seq)
    B = None if budget_ratio is None else float(budget_ratio) * T

    algo = DisjointLinUCB(env.K, env.d, alpha=alpha, lam=lam, seed=seed)

    pending: dict[int, list[tuple[int, np.ndarray, float]]] = {}  # t_due -> [(a,x,r)]
    cum_r = np.zeros(T, dtype=np.float64)
    cum_c = np.zeros(T, dtype=np.float64)

    total_r = 0.0
    total_c = 0.0

    for t in range(T):
        # 1) apply arrived feedback: ONLY label update
        if t in pending:
            for (a_upd, x_upd, r_upd) in pending[t]:
                algo.update_reward(a_upd, x_upd, r_upd)

        x = X_seq[t]

        # 2) choose action
        a = algo.select(x)

        # 3) know cost immediately (arm-dependent)
        c = float(env.costs[a])

        # 4) stop-at-budget before executing action
        if stop_at_budget and (B is not None) and (total_c + c > B):
            cum_r[t:] = total_r
            cum_c[t:] = total_c
            break

        # 5) execute action: immediate design update
        algo.observe(a, x)

        # 6) environment generates reward + delay
        r, _, dly = env.step(x, a)

        total_c += c
        total_r += r

        # 7) schedule label update (or apply immediately if dly==0)
        if dly >= 0:
            t_due = t + int(dly)
            if t_due < T:
                pending.setdefault(t_due, []).append((a, x, r))
            elif int(dly) == 0:
                # redundant but harmless; keep logic explicit
                algo.update_reward(a, x, r)
        else:
            # backward-compat if env still uses -1
            algo.update_reward(a, x, r)

        cum_r[t] = total_r
        cum_c[t] = total_c

    return cum_r, cum_c


def run_pd_linucb(
    env: SimBanditEnv,
    X_seq: np.ndarray,
    alpha: float,
    lam: float,
    eta: float,
    budget_ratio: float,
    seed: int,
    stop_at_budget: bool,
):
    T = len(X_seq)
    B = float(budget_ratio) * T

    algo = PrimalDualLinUCB(
        env.K,
        env.d,
        costs=env.costs,
        alpha=alpha,
        lam=lam,
        eta=eta,
        budget_per_step=(B / T),  # fallback; но ниже используем динамический b_t
        seed=seed,
    )

    pending: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    cum_r = np.zeros(T, dtype=np.float64)
    cum_c = np.zeros(T, dtype=np.float64)
    dual_hist = np.zeros(T, dtype=np.float64)

    total_r = 0.0
    total_c = 0.0

    for t in range(T):
        # 1) apply arrived feedback
        if t in pending:
            for (a_upd, x_upd, r_upd) in pending[t]:
                algo.update_reward(a_upd, x_upd, r_upd)

        x = X_seq[t]

        # 2) dynamic remaining-budget target:
        #    remaining steps incl current = (T - t)
        rem_steps = float(T - t)
        b_t = (B - total_c) / rem_steps

        # 3) choose action (dual updated inside)
        a, c = algo.select_and_spend(x, b_t=b_t)

        # 4) stop-at-budget before executing action
        if stop_at_budget and (total_c + c > B):
            cum_r[t:] = total_r
            cum_c[t:] = total_c
            dual_hist[t:] = algo.dual
            break

        # 5) immediate design update
        algo.observe(a, x)

        # 6) env step
        r, _, dly = env.step(x, a)

        total_c += c
        total_r += r

        # 7) schedule label update
        if dly >= 0:
            t_due = t + int(dly)
            if t_due < T:
                pending.setdefault(t_due, []).append((a, x, r))
            elif int(dly) == 0:
                algo.update_reward(a, x, r)
        else:
            algo.update_reward(a, x, r)

        cum_r[t] = total_r
        cum_c[t] = total_c
        dual_hist[t] = algo.dual

    return cum_r, cum_c, dual_hist


def append_csv(path: str, row: dict):
    p = Path(path)
    newfile = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if newfile:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, default="data/processed/obd_feedback_delayed.npz")
    ap.add_argument(
        "--memmap_dir",
        type=str,
        default="",
        help="If set, load env from memmaps (X.npy/D.npy/meta_and_stats.npz) instead of --npz",
    )
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--budget_ratio", type=float, default=0.30)

    ap.add_argument("--alpha", type=float, default=1.0, help="Fallback alpha if alpha_lin/alpha_pd not set")
    ap.add_argument("--lam", type=float, default=1.0)

    ap.add_argument("--alpha_lin", type=float, default=None)
    ap.add_argument("--alpha_pd", type=float, default=None)

    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=123)

    ap.add_argument("--stop_at_budget", action="store_true")

    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--log_csv", type=str, default="")

    args = ap.parse_args()

    alpha_lin = float(args.alpha if args.alpha_lin is None else args.alpha_lin)
    alpha_pd = float(args.alpha if args.alpha_pd is None else args.alpha_pd)

    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    if args.memmap_dir:
        env = SimBanditEnv.from_memmap_dir(args.memmap_dir, seed=args.seed, ridge_lambda=1.0, cost_mode="lin")
    else:
        env = SimBanditEnv.from_npz(args.npz, seed=args.seed, ridge_lambda=1.0, cost_mode="lin")

    X_seq = env.sample_contexts(args.T)

    r1, c1 = run_linucb(
        env,
        X_seq,
        alpha=alpha_lin,
        lam=args.lam,
        seed=args.seed + 1,
        budget_ratio=args.budget_ratio,
        stop_at_budget=args.stop_at_budget,
    )
    r2, c2, dual = run_pd_linucb(
        env,
        X_seq,
        alpha=alpha_pd,
        lam=args.lam,
        eta=args.eta,
        budget_ratio=args.budget_ratio,
        seed=args.seed + 2,
        stop_at_budget=args.stop_at_budget,
    )

    T = args.T
    B = float(args.budget_ratio) * T
    v1 = np.maximum(0.0, c1 - B)
    v2 = np.maximum(0.0, c2 - B)

    print("=== Summary ===")
    print("Budget B =", B, "stop_at_budget =", bool(args.stop_at_budget))
    print(f"LinUCB(alpha={alpha_lin}):    reward {float(r1[-1])} cost {float(c1[-1])} violation {float(v1[-1])}")
    print(f"PD-LinUCB(alpha={alpha_pd}, eta={args.eta}): reward {float(r2[-1])} cost {float(c2[-1])} violation {float(v2[-1])}")

    if args.log_csv:
        append_csv(
            args.log_csv,
            {
                "memmap_dir": args.memmap_dir,
                "npz": args.npz,
                "T": args.T,
                "budget_ratio": args.budget_ratio,
                "B": B,
                "stop_at_budget": int(args.stop_at_budget),
                "seed": args.seed,
                "alpha_lin": alpha_lin,
                "alpha_pd": alpha_pd,
                "lam": args.lam,
                "eta": args.eta,
                "lin_reward": float(r1[-1]),
                "lin_cost": float(c1[-1]),
                "lin_violation": float(v1[-1]),
                "pd_reward": float(r2[-1]),
                "pd_cost": float(c2[-1]),
                "pd_violation": float(v2[-1]),
            },
        )

    if args.no_plots:
        return

    suf = f"_{args.tag}" if args.tag else ""

    plt.figure()
    plt.plot(r1, label=f"LinUCB (alpha={alpha_lin})")
    plt.plot(r2, label=f"PD-LinUCB (alpha={alpha_pd}, eta={args.eta})")
    plt.title("Cumulative reward")
    plt.xlabel("t")
    plt.ylabel("sum reward")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"compare_cum_reward{suf}.png", dpi=160)

    plt.figure()
    plt.plot(c1, label="LinUCB")
    plt.plot(c2, label="PD-LinUCB")
    plt.axhline(B, linestyle="--", label="Budget B")
    plt.title("Cumulative cost vs budget")
    plt.xlabel("t")
    plt.ylabel("sum cost")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"compare_cum_cost{suf}.png", dpi=160)

    plt.figure()
    plt.plot(v1, label="LinUCB violation")
    plt.plot(v2, label="PD-LinUCB violation")
    plt.title("Budget violation")
    plt.xlabel("t")
    plt.ylabel("max(0, cost-B)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / f"compare_violation{suf}.png", dpi=160)

    plt.figure()
    plt.plot(dual)
    plt.title("Dual variable lambda(t) (PD-LinUCB)")
    plt.xlabel("t")
    plt.ylabel("lambda")
    plt.tight_layout()
    plt.savefig(outdir / f"pd_lambda{suf}.png", dpi=160)

    print("Saved plots to:", outdir)


if __name__ == "__main__":
    main()