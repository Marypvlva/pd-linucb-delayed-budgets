# src/eval/sweep_budget.py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.env.sim_bandit_env import SimBanditEnv
from src.algos.linucb_pd_delayed import PrimalDualLinUCB


def run_pd_stop_at_budget(env, X_seq, alpha, lam, eta, budget_ratio, seed):
    """
    PD-LinUCB with delayed feedback + BwK stop-at-budget:
    stop the process when the next action would exceed B.
    """
    T = len(X_seq)
    B = float(budget_ratio) * T
    bps = B / T

    algo = PrimalDualLinUCB(
        env.K, env.d,
        costs=env.costs,
        alpha=alpha,
        lam=lam,
        eta=eta,
        budget_per_step=bps,
        seed=seed,
    )

    pending = {}
    total_r = 0.0
    total_c = 0.0
    t_stop = T  # фактическое время остановки (<=T)

    for t in range(T):
        # apply arrived feedback
        if t in pending:
            for (a, x, r) in pending[t]:
                algo.update(a, x, r)

        x = X_seq[t]
        a, c = algo.select_and_spend(x)

        # STOP-AT-BUDGET (BwK): если следующий шаг превышает бюджет, останавливаемся
        if total_c + c > B:
            t_stop = t
            break

        r, _, dly = env.step(x, a)

        total_c += c
        if dly >= 0:
            t_due = t + dly
            if t_due < T:
                pending.setdefault(t_due, []).append((a, x, r))
        else:
            algo.update(a, x, r)

        total_r += r

    viol = max(0.0, total_c - B)  # при stop-at-budget должно быть 0 (в пределах float)
    return total_r, total_c, viol, t_stop, B


def main():
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)

    npz = "data/processed/obd_feedback_delayed.npz"
    T = 5000
    seed = 123

    # параметры алгоритма (можешь менять)
    alpha = 1.5
    lam = 1.0
    eta = 0.05

    budgets = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]

    env = SimBanditEnv.from_npz(npz, seed=seed, ridge_lambda=1.0, cost_mode="lin")
    X_seq = env.sample_contexts(T)

    rows = []
    for br in budgets:
        r, c, v, t_stop, B = run_pd_stop_at_budget(
            env, X_seq, alpha=alpha, lam=lam, eta=eta, budget_ratio=br, seed=seed + 10
        )
        # Нормируем reward на шаг, чтобы сравнение было честнее при разном t_stop
        r_per_step = r / max(1, t_stop)

        rows.append((br, r, r_per_step, c, v, t_stop, B))
        print(
            f"budget_ratio={br:.2f}  "
            f"reward={r:.1f}  reward/step={r_per_step:.4f}  "
            f"cost={c:.1f}/{B:.1f}  violation={v:.1f}  t_stop={t_stop}"
        )

    rows = np.array(rows, dtype=float)
    # columns:
    # 0 br, 1 reward, 2 reward_per_step, 3 cost, 4 viol, 5 t_stop, 6 B

    # 1) total reward vs budget_ratio
    plt.figure()
    plt.plot(rows[:, 0], rows[:, 1], marker="o")
    plt.xlabel("budget_ratio (B/T)")
    plt.ylabel("total reward (until stop)")
    plt.title("PD-LinUCB (stop-at-budget): total reward vs budget")
    plt.tight_layout()
    plt.savefig(outdir / "sweep_reward_vs_budget_stop.png", dpi=160)

    # 2) reward per step vs budget_ratio (важнее для BwK, т.к. горизонты разные)
    plt.figure()
    plt.plot(rows[:, 0], rows[:, 2], marker="o")
    plt.xlabel("budget_ratio (B/T)")
    plt.ylabel("reward per step")
    plt.title("PD-LinUCB (stop-at-budget): reward/step vs budget")
    plt.tight_layout()
    plt.savefig(outdir / "sweep_reward_per_step_vs_budget_stop.png", dpi=160)

    # 3) spent cost vs budget_ratio (должно быть близко к B)
    plt.figure()
    plt.plot(rows[:, 0], rows[:, 3], marker="o", label="spent cost")
    plt.plot(rows[:, 0], rows[:, 6], marker="x", label="budget B")
    plt.xlabel("budget_ratio (B/T)")
    plt.ylabel("cost")
    plt.title("PD-LinUCB (stop-at-budget): spent cost vs budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "sweep_cost_vs_budget_stop.png", dpi=160)

    # 4) violation vs budget_ratio (должно быть ~0)
    plt.figure()
    plt.plot(rows[:, 0], rows[:, 4], marker="o")
    plt.xlabel("budget_ratio (B/T)")
    plt.ylabel("violation")
    plt.title("PD-LinUCB (stop-at-budget): violation vs budget")
    plt.tight_layout()
    plt.savefig(outdir / "sweep_violation_vs_budget_stop.png", dpi=160)

    # 5) stop time vs budget_ratio
    plt.figure()
    plt.plot(rows[:, 0], rows[:, 5], marker="o")
    plt.xlabel("budget_ratio (B/T)")
    plt.ylabel("t_stop")
    plt.title("PD-LinUCB (stop-at-budget): stopping time vs budget")
    plt.tight_layout()
    plt.savefig(outdir / "sweep_tstop_vs_budget_stop.png", dpi=160)

    print("Saved sweep plots to:", outdir)


if __name__ == "__main__":
    main()
