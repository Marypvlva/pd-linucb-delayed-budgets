import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap_dir", type=str, required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--out", type=str, default="arm_ridge_stats.npz")
    args = ap.parse_args()

    ddir = Path(args.memmap_dir)
    X = np.load(ddir / "X.npy", mmap_mode="r")  # (n,d)
    A = np.load(ddir / "A.npy", mmap_mode="r")  # (n,)
    R = np.load(ddir / "R.npy", mmap_mode="r")  # (n,)

    n, d = X.shape
    K = int(A.max() + 1)

    XtX_a = np.zeros((K, d, d), dtype=np.float64)
    XtR_a = np.zeros((K, d), dtype=np.float64)
    cnt_a = np.zeros((K,), dtype=np.int64)
    sum_r = np.zeros((K,), dtype=np.float64)

    batch = int(args.batch)
    for start in range(0, n, batch):
        end = min(n, start + batch)

        Xb = np.asarray(X[start:end], dtype=np.float64)          # (m,d)
        ab = np.asarray(A[start:end], dtype=np.int32)            # (m,)
        rb = np.asarray(R[start:end], dtype=np.float64)          # (m,)

        # обновляем по каждому arm, который встретился в чанке
        for a in np.unique(ab):
            mask = (ab == a)
            if not mask.any():
                continue
            Xa = Xb[mask]                 # (m_a,d)
            ra = rb[mask]                 # (m_a,)
            XtX_a[a] += Xa.T @ Xa
            XtR_a[a] += Xa.T @ ra
            cnt_a[a] += int(mask.sum())
            sum_r[a] += float(ra.sum())

        if (start // batch) % 10 == 0:
            print(f"processed {end}/{n}")

    out_path = ddir / args.out

    # сразу считаем theta_a (ridge) чтобы симулятору было проще
    I = np.eye(d, dtype=np.float64)
    Theta = np.zeros((K, d), dtype=np.float64)
    for a in range(K):
        Theta[a] = np.linalg.solve(XtX_a[a] + float(args.lam) * I, XtR_a[a])

    conv_rate = sum_r / np.maximum(cnt_a, 1)

    np.savez(
        out_path,
        Theta=Theta.astype(np.float32),
        XtX_a=XtX_a,
        XtR_a=XtR_a,
        cnt_a=cnt_a,
        conv_rate=conv_rate.astype(np.float32),
        lam=float(args.lam),
    )
    print("Saved:", out_path)
    print("K =", K, "d =", d)
    print("conv_rate mean:", float(conv_rate.mean()), "min/max:", float(conv_rate.min()), float(conv_rate.max()))


if __name__ == "__main__":
    main()
