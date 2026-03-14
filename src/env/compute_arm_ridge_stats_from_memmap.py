import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap_dir", type=str, required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--out", type=str, default="arm_ridge_stats.npz")
    ap.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=["auto", "all", "train", "test"],
        help="Which row split to fit on. 'auto' uses train if split.npy exists, otherwise all.",
    )
    args = ap.parse_args()

    ddir = Path(args.memmap_dir)
    X = np.load(ddir / "X.npy", mmap_mode="r")  # (n,d)
    A = np.load(ddir / "A.npy", mmap_mode="r")  # (n,)
    R = np.load(ddir / "R.npy", mmap_mode="r")  # (n,)
    split_path = ddir / "split.npy"
    split_arr = np.load(split_path, mmap_mode="r") if split_path.exists() else None

    row_split = str(args.split)
    if row_split == "auto":
        row_split = "train" if split_arr is not None else "all"
    if row_split in {"train", "test"} and split_arr is None:
        raise ValueError(f"Requested --split {row_split}, but {split_path} does not exist.")

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
        if row_split == "all":
            keep = np.ones((end - start,), dtype=bool)
        else:
            split_b = np.asarray(split_arr[start:end], dtype=np.uint8)
            keep = (split_b == 0) if row_split == "train" else (split_b == 1)
        if not bool(np.any(keep)):
            continue

        Xb = Xb[keep]
        ab = ab[keep]
        rb = rb[keep]

        # Update each arm that appears in the current chunk.
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

    # Compute ridge estimates theta_a up front to simplify the simulator.
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
        row_split=row_split,
    )
    print("Saved:", out_path)
    print("K =", K, "d =", d, "row_split =", row_split)
    print("conv_rate mean:", float(conv_rate.mean()), "min/max:", float(conv_rate.min()), float(conv_rate.max()))


if __name__ == "__main__":
    main()
