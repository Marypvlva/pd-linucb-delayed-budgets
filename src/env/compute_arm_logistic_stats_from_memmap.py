from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap_dir", type=str, required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--max_iter", type=int, default=8)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--damping", type=float, default=1.0)
    ap.add_argument("--out", type=str, default="arm_logistic_stats.npz")
    ap.add_argument(
        "--split",
        type=str,
        default="auto",
        choices=["auto", "all", "train", "test"],
        help="Which row split to fit on. 'auto' uses train if split.npy exists, otherwise all.",
    )
    args = ap.parse_args()

    ddir = Path(args.memmap_dir)
    meta_stats = np.load(ddir / "meta_and_stats.npz", allow_pickle=True)
    meta = meta_stats["meta"].item()
    X = np.load(ddir / "X.npy", mmap_mode="r")
    A = np.load(ddir / "A.npy", mmap_mode="r")
    R = np.load(ddir / "R.npy", mmap_mode="r")
    split_path = ddir / "split.npy"
    split_arr = np.load(split_path, mmap_mode="r") if split_path.exists() else None

    row_split = str(args.split)
    if row_split == "auto":
        row_split = "train" if split_arr is not None else "all"
    if row_split in {"train", "test"} and split_arr is None:
        raise ValueError(f"Requested --split {row_split}, but {split_path} does not exist.")

    n, d = X.shape
    K = int(meta["k_cap"])
    p = d + 1
    beta = np.zeros((K, p), dtype=np.float64)
    cnt_a = np.zeros((K,), dtype=np.int64)
    sum_r = np.zeros((K,), dtype=np.float64)
    reg_diag = np.ones((p,), dtype=np.float64)
    reg_diag[-1] = 0.0
    eye_reg = np.diag(reg_diag)
    batch = int(args.batch)

    for it in range(int(args.max_iter)):
        grad = np.zeros((K, p), dtype=np.float64)
        hess = np.zeros((K, p, p), dtype=np.float64)
        neg_loglik = 0.0
        if it == 0:
            cnt_a.fill(0)
            sum_r.fill(0.0)

        for start in range(0, n, batch):
            end = min(n, start + batch)
            ab = np.asarray(A[start:end], dtype=np.int32)
            rb = np.asarray(R[start:end], dtype=np.float64)
            if row_split == "all":
                keep = np.ones((end - start,), dtype=bool)
            else:
                split_b = np.asarray(split_arr[start:end], dtype=np.uint8)
                keep = (split_b == 0) if row_split == "train" else (split_b == 1)
            if not bool(np.any(keep)):
                continue

            ab = ab[keep]
            rb = rb[keep]
            Xb = np.asarray(X[start:end], dtype=np.float64)[keep]

            logits = np.sum(beta[ab, :-1] * Xb, axis=1) + beta[ab, -1]
            probs = np.asarray(sigmoid(logits), dtype=np.float64)
            weights = np.clip(probs * (1.0 - probs), 1e-4, 0.25)
            Xaug = np.empty((Xb.shape[0], p), dtype=np.float64)
            Xaug[:, :-1] = Xb
            Xaug[:, -1] = 1.0

            probs_clip = np.clip(probs, 1e-9, 1.0 - 1e-9)
            neg_loglik += float(-np.sum(rb * np.log(probs_clip) + (1.0 - rb) * np.log(1.0 - probs_clip)))

            for a in np.unique(ab):
                mask = (ab == a)
                if not bool(np.any(mask)):
                    continue
                Xa = Xaug[mask]
                ya = rb[mask]
                pa = probs[mask]
                wa = weights[mask]
                grad[a] += Xa.T @ (ya - pa)
                hess[a] += (Xa.T * wa) @ Xa
                if it == 0:
                    cnt_a[a] += int(mask.sum())
                    sum_r[a] += float(ya.sum())

            if (start // batch) % 10 == 0:
                print(f"iter {it + 1}: processed {end}/{n}")

        max_delta = 0.0
        for a in range(K):
            g = grad[a].copy()
            H = hess[a].copy()
            g[:-1] -= float(args.lam) * beta[a, :-1]
            H += float(args.lam) * eye_reg
            try:
                delta = np.linalg.solve(H, g)
            except np.linalg.LinAlgError:
                delta = np.linalg.pinv(H) @ g
            beta[a] += float(args.damping) * delta
            max_delta = max(max_delta, float(np.max(np.abs(delta))))

        penalty = 0.5 * float(args.lam) * float(np.sum(beta[:, :-1] ** 2))
        print(
            f"iter {it + 1}: penalized_obj={neg_loglik + penalty:.6f} "
            f"max_abs_delta={max_delta:.6e}"
        )
        if max_delta < float(args.tol):
            print(f"Converged at iter {it + 1}")
            break

    out_path = ddir / args.out
    np.savez(
        out_path,
        Theta=beta[:, :-1].astype(np.float32),
        bias=beta[:, -1].astype(np.float32),
        cnt_a=cnt_a,
        conv_rate=(sum_r / np.maximum(cnt_a, 1)).astype(np.float32),
        lam=float(args.lam),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        damping=float(args.damping),
        row_split=row_split,
    )
    print("Saved:", out_path)
    print("K =", K, "d =", d, "row_split =", row_split)


if __name__ == "__main__":
    main()
