import numpy as np
from pathlib import Path

from obp.dataset import OpenBanditDataset

OUT_PATH = Path("data/processed/obd_feedback.npz")


def main():
    # 1) Load the Open Bandit Dataset (OBD) via OBP.
    # These parameters are commonly available:
    dataset = OpenBanditDataset(behavior_policy="random", campaign="all")

    bandit_feedback = dataset.obtain_batch_bandit_feedback()

    # 2) Extract arrays.
    X = bandit_feedback["context"].astype(np.float32)          # (n, d)
    A = bandit_feedback["action"].astype(np.int64)             # (n,)
    R = bandit_feedback["reward"].astype(np.float32)           # (n,)

    # pscore is useful for off-policy evaluation, but optional here.
    pscore = bandit_feedback.get("pscore", None)
    if pscore is not None:
        pscore = np.asarray(pscore, dtype=np.float32)

    n = X.shape[0]

    # 3) Cost model for the MVP: one resource, cost=1 for every round.
    C = np.ones((n, 1), dtype=np.float32)

    # 4) No delays for now: -1 means "instant / delays unused".
    D = -np.ones(n, dtype=np.int64)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if pscore is None:
        np.savez(OUT_PATH, X=X, A=A, R=R, C=C, D=D)
    else:
        np.savez(OUT_PATH, X=X, A=A, R=R, C=C, D=D, pscore=pscore)

    print("Saved:", OUT_PATH)
    print("Shapes:",
          "X", X.shape,
          "A", A.shape,
          "R", R.shape,
          "C", C.shape,
          "D", D.shape,
          "pscore", None if pscore is None else pscore.shape)


if __name__ == "__main__":
    main()
