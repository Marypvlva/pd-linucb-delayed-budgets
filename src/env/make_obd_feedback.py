import numpy as np
from pathlib import Path

from obp.dataset import OpenBanditDataset

OUT_PATH = Path("data/processed/obd_feedback.npz")


def main():
    # 1) Загружаем Open Bandit Dataset (OBD) через OBP
    # Часто работают такие параметры:
    dataset = OpenBanditDataset(behavior_policy="random", campaign="all")

    bandit_feedback = dataset.obtain_batch_bandit_feedback()

    # 2) Достаём массивы
    X = bandit_feedback["context"].astype(np.float32)          # (n, d)
    A = bandit_feedback["action"].astype(np.int64)             # (n,)
    R = bandit_feedback["reward"].astype(np.float32)           # (n,)

    # pscore полезен для off-policy, но не обязателен сейчас
    pscore = bandit_feedback.get("pscore", None)
    if pscore is not None:
        pscore = np.asarray(pscore, dtype=np.float32)

    n = X.shape[0]

    # 3) Стоимость (пока MVP): один ресурс, cost=1 за любой шаг
    C = np.ones((n, 1), dtype=np.float32)

    # 4) Задержки (пока нет): -1 означает "мгновенно/не используем delays"
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
