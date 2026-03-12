# src/eval/run_gamma_sweep.py
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.env.sim_bandit_env import SimBanditEnv
from src.algos.linucb_pd_delayed import PrimalDualLinUCB
from src.algos.cost_normalized_ucb import CostNormalizedDisjointUCB


def t_crit_975(df: int) -> float:
    # 95% CI => t_{0.975, df}
    try:
        import scipy.stats as st  # optional
        return float(st.t.ppf(0.975, df))
    except Exception:
        # fallback ~ Normal(0,1)
        return 1.96


@dataclass
class Row:
    gamma: float
    seed: int
    reward: float
    cost: float
    spent_ratio: float
    t_stop: int


def feasible_mask(costs: np.ndarray, remaining_budget: float) -> np.ndarray:
    rem = float(remaining_budget)
    return costs.astype(np.float64) <= (rem + 1e-12)


def run_contextual_algo_costnorm_sub(
    env: SimBanditEnv,
    X_seq: np.ndarray,
    algo: CostNormalizedDisjointUCB,
    budget_ratio: float,
    stop_at_budget: bool,
):
    """
    Correct runner for CostNormUCB[sub] with delayed feedback:
      - apply pending reward updates
      - choose a_t among feasible arms if stop_at_budget
      - update_design immediately after committing action
      - update_reward on feedback arrival (pending queue)
    """
    T = len(X_seq)
    B = float(budget_ratio) * T
    costs = env.costs

    pending: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    total_r = 0.0
    total_c = 0.0
    t_stop = T

    for t in range(T):
        for (a_upd, x_upd, r_upd) in pending.pop(t, []):
            algo.update_reward(a_upd, x_upd, r_upd)

        x = X_seq[t]

        if stop_at_budget:
            feas = feasible_mask(costs, B - total_c)
            if not bool(np.any(feas)):
                t_stop = t
                break
            a = int(algo.select_feasible(x, feas))
        else:
            a = int(algo.select(x))

        c = float(costs[a])

        # commit design
        algo.update_design(a, x)

        # environment step
        r, _, dly = env.step(x, a)

        total_c += c
        total_r += float(r)

        dly = int(dly)
        if dly <= 0:
            algo.update_reward(a, x, float(r))
        else:
            t_due = t + dly
            if t_due < T:
                pending.setdefault(t_due, []).append((a, x, float(r)))

    return total_r, total_c, t_stop


def run_pd_once(
    memmap_dir: str,
    T: int,
    budget_ratio: float,
    alpha_pd: float,
    lam: float,
    eta_pd: float,
    seed: int,
    stop_at_budget: bool,
):
    """
    PD-LinUCB baseline with the *same* semantics as run_compare_baselines:
      - choose a_t among feasible arms if stop_at_budget
      - dual update: b_t = (B - spent) / (T - t)
      - update_design immediately, update_reward on feedback arrival
    """
    env = SimBanditEnv.from_memmap_dir(memmap_dir, seed=seed, ridge_lambda=1.0, cost_mode="lin")
    X_seq = env.sample_contexts(T)

    B = float(budget_ratio) * T
    costs = env.costs

    algo = PrimalDualLinUCB(
        env.K, env.d,
        costs=env.costs,
        alpha=float(alpha_pd),
        lam=float(lam),
        eta=float(eta_pd),
        seed=seed + 777,
    )

    pending: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    total_r = 0.0
    total_c = 0.0
    t_stop = T

    for t in range(T):
        for (a_upd, x_upd, r_upd) in pending.pop(t, []):
            algo.update_reward(a_upd, x_upd, r_upd)

        x = X_seq[t]

        if stop_at_budget:
            feas = feasible_mask(costs, B - total_c)
            if not bool(np.any(feas)):
                t_stop = t
                break
            a = int(algo.select_feasible(x, feas))
        else:
            a = int(algo.select(x))

        c = float(costs[a])

        # dual update BEFORE spending
        H = T - t
        b_t = (B - total_c) / max(H, 1)
        algo.update_dual(c, b_t)

        # commit design
        algo.update_design(a, x)

        r, _, dly = env.step(x, a)

        total_c += c
        total_r += float(r)

        dly = int(dly)
        if dly <= 0:
            algo.update_reward(a, x, float(r))
        else:
            t_due = t + dly
            if t_due < T:
                pending.setdefault(t_due, []).append((a, x, float(r)))

    spent_ratio = total_c / max(B, 1e-12)
    return total_r, total_c, spent_ratio, t_stop


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Empty rows for {path}")
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[Row]):
    gammas = sorted(set(r.gamma for r in rows))
    out = []
    for g in gammas:
        rr = [r for r in rows if r.gamma == g]
        n = len(rr)
        df = max(n - 1, 1)
        tc = t_crit_975(df)

        def mean_ci(vals):
            vals = np.asarray(vals, dtype=np.float64)
            m = float(vals.mean())
            s = float(vals.std(ddof=1)) if n > 1 else 0.0
            ci = tc * s / sqrt(n) if n > 1 else 0.0
            return m, ci

        reward_m, reward_ci = mean_ci([r.reward for r in rr])
        spent_m, spent_ci = mean_ci([r.spent_ratio for r in rr])
        tstop_m, tstop_ci = mean_ci([r.t_stop for r in rr])

        out.append({
            "gamma": g,
            "n": n,
            "reward_mean": reward_m,
            "reward_ci95": reward_ci,
            "spent_mean": spent_m,
            "spent_ci95": spent_ci,
            "tstop_mean": tstop_m,
            "tstop_ci95": tstop_ci,
        })
    return out


def plot_with_ci(x, y, yerr, title, xlabel, ylabel, out_path: Path, xscale: str = "symlog"):
    plt.figure()
    plt.errorbar(x, y, yerr=yerr, marker="o", linewidth=1.5, capsize=3)
    if xscale:
        plt.xscale(xscale, linthresh=0.1)  # symlog allows gamma=0
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
    ap.add_argument("--T", type=int, default=8000)
    ap.add_argument("--budget_ratio", type=float, default=0.7)
    ap.add_argument("--stop_at_budget", action="store_true")

    ap.add_argument("--gammas", type=str, default="0,0.1,0.3,1,2,3,5,10")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=123)

    ap.add_argument("--alpha_cnu", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1.0)

    ap.add_argument("--also_pd", action="store_true")
    ap.add_argument("--alpha_pd", type=float, default=1.5)
    ap.add_argument("--eta_pd", type=float, default=0.05)

    ap.add_argument("--out_dir", type=str, default="results/gamma_sweep")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gammas = [float(s) for s in args.gammas.split(",") if s.strip() != ""]
    rows_raw: list[Row] = []

    for si in range(args.n_seeds):
        seed = int(args.seed0 + si)

        env = SimBanditEnv.from_memmap_dir(args.memmap_dir, seed=seed, ridge_lambda=1.0, cost_mode="lin")
        X_seq = env.sample_contexts(args.T)
        B = float(args.budget_ratio) * args.T

        for g in gammas:
            algo = CostNormalizedDisjointUCB(
                env.K, env.d,
                costs=env.costs,
                alpha=float(args.alpha_cnu),
                lam=float(args.lam),
                mode="sub",
                gamma=float(g),
                eps=1e-3,
                seed=seed + 1000 + int(100 * g),
            )

            total_r, total_c, t_stop = run_contextual_algo_costnorm_sub(
                env, X_seq, algo,
                budget_ratio=args.budget_ratio,
                stop_at_budget=bool(args.stop_at_budget),
            )
            spent_ratio = float(total_c / max(B, 1e-12))
            rows_raw.append(Row(gamma=g, seed=seed, reward=float(total_r), cost=float(total_c),
                                spent_ratio=spent_ratio, t_stop=int(t_stop)))

        print(f"done seed {seed}")

    raw_path = out_dir / "gamma_sweep_raw.csv"
    write_csv(raw_path, [r.__dict__ for r in rows_raw])

    summary = summarize(rows_raw)
    sum_path = out_dir / "gamma_sweep_summary.csv"
    write_csv(sum_path, summary)

    x = [d["gamma"] for d in summary]
    reward = [d["reward_mean"] for d in summary]
    reward_ci = [d["reward_ci95"] for d in summary]
    spent = [d["spent_mean"] for d in summary]
    spent_ci = [d["spent_ci95"] for d in summary]

    plot_with_ci(
        x, reward, reward_ci,
        title="CNU[sub]: mean total reward vs gamma (95% CI)",
        xlabel="gamma",
        ylabel="mean total reward",
        out_path=out_dir / "gamma_sweep_reward.png",
        xscale="symlog",
    )

    plot_with_ci(
        x, spent, spent_ci,
        title="CNU[sub]: mean spent/B vs gamma (95% CI)",
        xlabel="gamma",
        ylabel="mean spent/B",
        out_path=out_dir / "gamma_sweep_spent.png",
        xscale="symlog",
    )

    if args.also_pd:
        pd_rows = []
        for si in range(args.n_seeds):
            seed = int(args.seed0 + si)
            r, c, sratio, tstop = run_pd_once(
                args.memmap_dir, args.T, args.budget_ratio,
                args.alpha_pd, args.lam, args.eta_pd,
                seed, bool(args.stop_at_budget)
            )
            pd_rows.append((r, sratio, tstop))

        pd_r = np.array([x[0] for x in pd_rows], dtype=float)
        pd_s = np.array([x[1] for x in pd_rows], dtype=float)
        pd_t = np.array([x[2] for x in pd_rows], dtype=float)

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