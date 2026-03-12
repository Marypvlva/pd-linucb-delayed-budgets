# src/eval/sweep_budget.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from src.env.sim_bandit_env import SimBanditEnv
from src.algos.linucb_pd_delayed import PrimalDualLinUCB


def feasible_mask(costs: np.ndarray, remaining_budget: float) -> np.ndarray:
    rem = float(remaining_budget)
    return costs.astype(np.float64) <= (rem + 1e-12)


def run_pd_stop_at_budget(env: SimBanditEnv, X_seq: np.ndarray, alpha: float, lam: float, eta: float,
                          budget_ratio: float, seed: int, stop_at_budget: bool = True):
    """
    PD-LinUCB with delayed feedback + stop-at-budget (feasible-set selection):
      - choose a_t among feasible arms under remaining budget
      - stop only if feasible set is empty
      - dual update uses b_t = (B - spent) / (T - t)
      - design-now, reward-later via pending queue
    """
    T = len(X_seq)
    B = float(budget_ratio) * T
    costs = env.costs

    algo = PrimalDualLinUCB(
        env.K, env.d,
        costs=env.costs,
        alpha=float(alpha),
        lam=float(lam),
        eta=float(eta),
        seed=int(seed),
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

        # dual update before spending
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

    viol = max(0.0, total_c - B)  # should be 0 (up to float eps)
    return total_r, total_c, viol, t_stop, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap_dir", type=str, required=True)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--budgets", type=str, default="0.40,0.55,0.70,0.85")
    ap.add_argument("--alpha", type=float, default=1.5)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--eta", type=float, default=0.05)
    ap.add_argument("--out_dir", type=str, default="results/sweep_budget")
    args = ap.parse_args()

    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    budgets = [float(x) for x in args.budgets.split(",") if x.strip() != ""]

    env = SimBanditEnv.from_memmap_dir(args.memmap_dir, seed=args.seed, ridge_lambda=1.0, cost_mode="lin")
    X_seq = env.sample_contexts(args.T)

    rows = []
    for br in budgets:
        r, c, v, t_stop, B = run_pd_stop_at_budget(
            env, X_seq,
            alpha=args.alpha, lam=args.lam, eta=args.eta,
            budget_ratio=br, seed=args.seed + 10,
            stop_at_budget=True
        )
        r_per_step = r / max(1, t_stop)
        rows.append((br, r, r_per_step, c, v, t_stop, B))
        print(
            f"budget_ratio={br:.2f}  "
            f"reward={r:.1f}  reward/step={r_per_step:.4f}  "
            f"cost={c:.1f}/{B:.1f}  violation={v:.6f}  t_stop={t_stop}"
        )

    rows = np.array(rows, dtype=float)
    # cols: 0 br, 1 reward, 2 reward_per_step, 3 cost, 4 viol, 5 t_stop, 6 B

    # 1) total reward vs budget_ratio
    plt.figure()
    plt.plot(rows[:, 0], rows[:, 1], marker="o")
    plt.xlabel("budget_ratio (B/T)")
    plt.ylabel("total reward (until stop)")
    plt.title("PD-LinUCB (stop-at-budget): total reward vs budget")
    plt.tight_layout()
    plt.savefig(outdir / "sweep_reward_vs_budget_stop.png", dpi=180)
    plt.close()

    # 2) reward per step vs budget_ratio
    plt.figure()
    plt.plot(rows[:, 0], rows[:, 2], marker="o")
    plt.xlabel("budget_ratio (B/T)")
    plt.ylabel("reward per step")
    plt.title("PD-LinUCB (stop-at-budget): reward/step vs budget")
    plt.tight_layout()
    plt.savefig(outdir / "sweep_reward_per_step_vs_budget_stop.png", dpi=180)
    plt.close()

    # 3) spent cost vs budget_ratio
    plt.figure()
    plt.plot(rows[:, 0], rows[:, 3], marker="o", label="spent cost")
    plt.plot(rows[:, 0], rows[:, 6], marker="x", label="budget B")
    plt.xlabel("budget_ratio (B/T)")
    plt.ylabel("cost")
    plt.title("PD-LinUCB (stop-at-budget): spent cost vs budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "sweep_cost_vs_budget_stop.png", dpi=180)
    plt.close()

    # 4) violation vs budget_ratio
    plt.figure()
    plt.plot(rows[:, 0], rows[:, 4], marker="o")
    plt.xlabel("budget_ratio (B/T)")
    plt.ylabel("violation")
    plt.title("PD-LinUCB (stop-at-budget): budget violation (should be ~0)")
    plt.tight_layout()
    plt.savefig(outdir / "sweep_violation_vs_budget_stop.png", dpi=180)
    plt.close()

    # 5) stop time vs budget_ratio
    plt.figure()
    plt.plot(rows[:, 0], rows[:, 5], marker="o")
    plt.xlabel("budget_ratio (B/T)")
    plt.ylabel("t_stop")
    plt.title("PD-LinUCB (stop-at-budget): stopping time vs budget")
    plt.tight_layout()
    plt.savefig(outdir / "sweep_tstop_vs_budget_stop.png", dpi=180)
    plt.close()

    print("Saved sweep plots to:", outdir)


if __name__ == "__main__":
    main()