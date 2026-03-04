# src/env/make_criteo_attrib_feedback.py
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


CRITEO_COLS = [
    "timestamp", "uid", "campaign", "conversion", "conversion_timestamp", "conversion_id",
    "attribution", "click", "click_pos", "click_nb", "cost", "cpo", "time_since_last_click",
    "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9"
]


def make_reader(path: str, chunksize: int):
    """
    Dataset may be:
      - with header row (columns already named)
      - without header (then we assign CRITEO_COLS)
    """
    df0 = pd.read_csv(path, sep="\t", compression="infer", nrows=5)
    if "timestamp" in df0.columns and "campaign" in df0.columns:
        return pd.read_csv(path, sep="\t", compression="infer", chunksize=chunksize)
    return pd.read_csv(path, sep="\t", compression="infer", header=None, names=CRITEO_COLS, chunksize=chunksize)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=str, default="data/raw/criteo_attrib/criteo_attribution_dataset.tsv.gz")
    ap.add_argument("--out", type=str, default="data/processed/criteo_attrib_feedback_delayed.npz")
    ap.add_argument("--n_rows", type=int, default=300_000)
    ap.add_argument("--d_hash", type=int, default=256)      # hashed feature dimension
    ap.add_argument("--max_delay", type=int, default=5000)   # cap delay (cts - ts)
    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument(
        "--k_cap",
        type=int,
        default=50,
        help="Cap number of arms by mapping campaign_id -> (campaign_id mod k_cap). "
             "Use this to make LinUCB feasible (e.g., k_cap=50).",
    )
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    d = int(args.d_hash) + 1  # +1 numeric feature
    X = np.zeros((args.n_rows, d), dtype=np.float32)
    A = np.zeros((args.n_rows,), dtype=np.int32)
    R = np.zeros((args.n_rows,), dtype=np.float32)
    C = np.zeros((args.n_rows, 1), dtype=np.float32)
    D = np.zeros((args.n_rows,), dtype=np.int32)

    camp2id: dict[str, int] = {}
    write = 0

    cat_cols = ["uid"] + [f"cat{i}" for i in range(1, 10)]

    reader = make_reader(args.inp, args.chunksize)
    for chunk in reader:
        if write >= args.n_rows:
            break

        # required columns
        need = ["timestamp", "conversion", "conversion_timestamp", "campaign", "cost", "time_since_last_click"]
        for col in need:
            if col not in chunk.columns:
                raise RuntimeError(f"Column {col} not found. Columns: {list(chunk.columns)[:30]}")

        # clip chunk to remaining capacity
        n_take = min(len(chunk), args.n_rows - write)
        chunk = chunk.iloc[:n_take].copy()

        idx = np.arange(write, write + n_take)

        # --- action: campaign -> id (optionally capped by k_cap)
        camp_vals = chunk["campaign"].astype(str).to_numpy()
        a_chunk = np.empty(n_take, dtype=np.int32)
        k_cap = None if args.k_cap is None else int(args.k_cap)
        for i, v in enumerate(camp_vals):
            if v not in camp2id:
                camp2id[v] = len(camp2id)
            aid = camp2id[v]
            if k_cap is not None and k_cap > 0:
                aid = aid % k_cap
            a_chunk[i] = aid
        A[idx] = a_chunk

        # --- reward: conversion (0/1)
        conv = chunk["conversion"].fillna(0).to_numpy(dtype=np.int32)
        R[idx] = conv.astype(np.float32)

        # --- cost: cost (NA -> 0)
        C[idx, 0] = chunk["cost"].fillna(0).to_numpy(dtype=np.float32)

        # --- delay: conversion_timestamp - timestamp if conversion==1 and valid, else -1
        ts = chunk["timestamp"].to_numpy(dtype=np.int64)
        cts = chunk["conversion_timestamp"].to_numpy(dtype=np.int64)
        raw_delay = cts - ts
        ok = (conv == 1) & (cts >= 0) & (raw_delay >= 0) & (raw_delay <= int(args.max_delay))
        D[idx] = np.where(ok, raw_delay, -1).astype(np.int32)

        # --- context X: hashing of uid + cat1..cat9, plus numeric time_since_last_click
        for col in cat_cols:
            s = chunk[col].astype(str)
            h = pd.util.hash_pandas_object(s, index=False).to_numpy(dtype=np.uint64)
            j = (h % int(args.d_hash)).astype(np.int32)
            X[idx, j] += 1.0

        tslc = chunk["time_since_last_click"].fillna(0).to_numpy(dtype=np.float32)
        X[idx, -1] = np.log1p(np.maximum(tslc, 0))

        write += n_take

        # some stats
        # Note: campaigns in raw may be huge, but K after capping is <= k_cap
        K_now = int(A[:write].max() + 1) if write > 0 else 0
        print(f"processed: {write}/{args.n_rows}, raw_campaigns_seen={len(camp2id)}, K_after_cap={K_now}")

    # trim
    X = X[:write]
    A = A[:write]
    R = R[:write]
    C = C[:write]
    D = D[:write]

    print("=== DONE ===")
    print("X", X.shape, "A", A.shape, "R", R.shape, "C", C.shape, "D", D.shape)
    print("K (after cap) =", int(A.max() + 1))
    print("conversion rate =", float(R.mean()))
    dpos = D[D >= 0]
    if len(dpos):
        print("delay stats (>=0): mean", float(dpos.mean()), "p90", float(np.quantile(dpos, 0.9)), "max", int(dpos.max()))

    np.savez(out_path, X=X, A=A, R=R, C=C, D=D, pscore=None)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
