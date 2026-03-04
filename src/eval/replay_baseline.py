import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("data/processed/obd_feedback.npz")
OUTDIR = Path("results")
OUTDIR.mkdir(parents=True, exist_ok=True)

def main():
    z = np.load(DATA)
    R = z["R"].astype(float)
    C = z["C"][:, 0].astype(float)

    T = len(R)

    # бюджет: например 30% от горизонта при cost=1
    B = 0.3 * T

    cum_r = np.cumsum(R)
    cum_c = np.cumsum(C)
    viol = np.maximum(0.0, cum_c - B)

    print("T=", T, "Budget B=", B)
    print("Total reward:", float(cum_r[-1]))
    print("Total cost:", float(cum_c[-1]), "Violation:", float(viol[-1]))

    # 1) reward
    plt.figure()
    plt.plot(cum_r)
    plt.title("Replay baseline: cumulative reward")
    plt.xlabel("t")
    plt.ylabel("sum reward")
    plt.tight_layout()
    plt.savefig(OUTDIR / "replay_cum_reward.png", dpi=160)

    # 2) cost vs budget
    plt.figure()
    plt.plot(cum_c)
    plt.axhline(B, linestyle="--")
    plt.title("Replay baseline: cumulative cost vs budget")
    plt.xlabel("t")
    plt.ylabel("sum cost")
    plt.tight_layout()
    plt.savefig(OUTDIR / "replay_cum_cost.png", dpi=160)

    # 3) violation
    plt.figure()
    plt.plot(viol)
    plt.title("Replay baseline: budget violation")
    plt.xlabel("t")
    plt.ylabel("max(0, cost-B)")
    plt.tight_layout()
    plt.savefig(OUTDIR / "replay_violation.png", dpi=160)

    print("Saved plots to:", OUTDIR)

if __name__ == "__main__":
    main()
