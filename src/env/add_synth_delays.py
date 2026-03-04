import numpy as np
from pathlib import Path

INP = Path("data/processed/obd_feedback.npz")
OUT = Path("data/processed/obd_feedback_delayed.npz")

def main():
    z = np.load(INP)
    X, A, R, C = z["X"], z["A"], z["R"], z["C"]

    T = len(R)

    # геометрические задержки в шагах, обрезаем максимумом 48
    rng = np.random.default_rng(42)
    delays = rng.geometric(p=0.2, size=T) - 1  # 0,1,2,...
    delays = np.clip(delays, 0, 48).astype(np.int64)

    # можно сделать: если reward=0, delay=-1 (не было события)
    D = delays
    D = np.where(R > 0, D, -1).astype(np.int64)

    # сохраняем всё как было + новые задержки
    payload = dict(X=X, A=A, R=R, C=C, D=D)
    for k in z.files:
        if k not in payload:
            payload[k] = z[k]
    np.savez(OUT, **payload)

    print("Saved:", OUT)
    print("Delay stats:",
          "fraction delayed events:", float((D >= 0).mean()),
          "mean delay (events):", float(D[D >= 0].mean()) if (D >= 0).any() else None)

if __name__ == "__main__":
    main()
