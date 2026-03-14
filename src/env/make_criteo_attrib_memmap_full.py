import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import subprocess
import math


CRITEO_COLS = [
    "timestamp", "uid", "campaign", "conversion", "conversion_timestamp", "conversion_id",
    "attribution", "click", "click_pos", "click_nb", "cost", "cpo", "time_since_last_click",
    "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "cat9"
]


def detect_header(path: str) -> bool:
    df0 = pd.read_csv(path, sep="\t", compression="infer", nrows=5)
    return ("timestamp" in df0.columns) and ("campaign" in df0.columns)


def count_lines_gz(path: str) -> int:
    # macOS-friendly: gzip -dc file | wc -l
    out = subprocess.check_output(["bash", "-lc", f"gzip -dc '{path}' | wc -l"], text=True).strip()
    return int(out)


def open_memmap(path: Path, shape, dtype):
    path.parent.mkdir(parents=True, exist_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def ceil_div(a: np.ndarray, b: int) -> np.ndarray:
    # ceil(a/b) for nonnegative integer a
    return (a + (b - 1)) // b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", type=str, default="data/raw/criteo_attrib/criteo_attribution_dataset.tsv.gz")
    ap.add_argument("--out_dir", type=str, default="data/processed/criteo_full_k50_d64_real")

    ap.add_argument("--d_hash", type=int, default=64)
    ap.add_argument("--k_cap", type=int, default=50)

    # IMPORTANT: delays are stored in "steps", computed from timestamps using delta_seconds
    ap.add_argument("--delta_seconds", type=int, default=3600, help="time discretization step (e.g. 3600=1h)")
    ap.add_argument(
        "--censor_seconds",
        type=int,
        default=5000 * 3600,
        help=(
            "censoring window W in seconds (default matches paper: W/Δ=5000 at Δ=3600). "
            "Set 30*24*3600 for a 30-day window."
        ),
    )
    # keep backward compatibility with your old flag name:
    ap.add_argument("--d_max", "--max_delay", dest="d_max", type=int, default=5000,
                    help="max delay in STEPS (used for clipping and as censor delay)")

    ap.add_argument("--chunksize", type=int, default=200_000)
    ap.add_argument("--limit_rows", type=int, default=0,
                    help="0 = весь датасет; иначе ограничить первыми N строками (для теста)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    has_header = detect_header(args.inp)
    total_lines = count_lines_gz(args.inp)
    n = total_lines - (1 if has_header else 0)
    if args.limit_rows and args.limit_rows > 0:
        n = min(n, int(args.limit_rows))

    d = int(args.d_hash) + 1
    k_cap = int(args.k_cap)

    delta = int(args.delta_seconds)
    W = int(args.censor_seconds)
    d_max = int(args.d_max)

    # censor delay in steps (clip to d_max to match your experimental protocol)
    censor_steps = int(math.ceil(W / delta))
    censor_steps = min(censor_steps, d_max)

    print("Header:", has_header)
    print("Total lines (gz):", total_lines, "=> rows:", n)
    print("d =", d, "k_cap =", k_cap)
    print("delta_seconds =", delta, "censor_seconds =", W,
          "=> censor_steps =", censor_steps, "d_max =", d_max)

    # --- allocate memmaps ---
    X = open_memmap(out_dir / "X.npy", shape=(n, d), dtype=np.float32)
    A = open_memmap(out_dir / "A.npy", shape=(n,), dtype=np.int16)   # k_cap<=32767
    R = open_memmap(out_dir / "R.npy", shape=(n,), dtype=np.uint8)

    # store per-row cost (real field from dataset)
    C = open_memmap(out_dir / "C.npy", shape=(n,), dtype=np.float32)

    # store delay in STEPS, always >=0
    # no-delay ablation should be created later by setting D[:] = 0
    D = open_memmap(out_dir / "D.npy", shape=(n,), dtype=np.int32)

    # --- per-arm sufficient stats for ridge (arm-specific reward model) ---
    XtX_arm = np.zeros((k_cap, d, d), dtype=np.float64)
    XtR_arm = np.zeros((k_cap, d), dtype=np.float64)
    cnt_arm = np.zeros((k_cap,), dtype=np.int64)

    # --- per-arm cost stats (real cost) ---
    sum_cost_arm = np.zeros((k_cap,), dtype=np.float64)

    # --- delays pool for positive conversions ---
    delays_pos_parts = []

    cat_cols = ["uid"] + [f"cat{i}" for i in range(1, 10)]

    read_kwargs = dict(sep="\t", compression="infer", chunksize=args.chunksize)
    if has_header:
        reader = pd.read_csv(args.inp, **read_kwargs)
    else:
        reader = pd.read_csv(args.inp, header=None, names=CRITEO_COLS, **read_kwargs)

    write = 0
    for chunk in reader:
        if write >= n:
            break

        n_take = min(len(chunk), n - write)
        chunk = chunk.iloc[:n_take].copy()
        idx = slice(write, write + n_take)

        # --- action: hash(campaign) % k_cap ---
        camp = chunk["campaign"].astype(str)
        hcamp = pd.util.hash_pandas_object(camp, index=False).to_numpy(dtype=np.uint64)
        a = (hcamp % k_cap).astype(np.int16)
        A[idx] = a

        # --- reward: conversion (0/1) ---
        conv = pd.to_numeric(chunk["conversion"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)
        r = (conv > 0).astype(np.uint8)
        R[idx] = r

        # --- cost: real field from dataset ---
        cost = pd.to_numeric(chunk["cost"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        C[idx] = cost

        # --- delay from timestamps with discretization + censoring ---
        ts = pd.to_numeric(chunk["timestamp"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)
        cts = pd.to_numeric(chunk["conversion_timestamp"], errors="coerce").fillna(-1).to_numpy(dtype=np.int64)

        raw = (cts - ts).astype(np.int64)  # seconds
        valid_pos = (r == 1) & (ts >= 0) & (cts >= 0) & (raw >= 0)

        # ceil(raw/delta) in steps, clip to d_max
        raw_pos = raw.copy()
        raw_pos[~valid_pos] = 0
        d_steps = ceil_div(raw_pos, delta).astype(np.int32)
        d_steps = np.minimum(d_steps, d_max).astype(np.int32)

        # IMPORTANT:
        # - for conversions (r=1): D = delay in steps
        # - for non-conversions (r=0): censored delay = censor_steps (>=0), NOT -1
        D_chunk = np.full((n_take,), censor_steps, dtype=np.int32)
        D_chunk[valid_pos] = d_steps[valid_pos]
        D[idx] = D_chunk

        # collect positive delays for later inspection / simulator
        if valid_pos.any():
            delays_pos_parts.append(D_chunk[valid_pos].astype(np.int32, copy=False))

        # --- context X: hashing uid+cat1..cat9 into d_hash, plus numeric feature ---
        X_chunk = np.zeros((n_take, d), dtype=np.float32)
        rows = np.arange(n_take, dtype=np.int32)
        for col in cat_cols:
            s = chunk[col].astype(str)
            h = pd.util.hash_pandas_object(s, index=False).to_numpy(dtype=np.uint64)
            j = (h % int(args.d_hash)).astype(np.int32)
            X_chunk[rows, j] += 1.0

        tslc = pd.to_numeric(chunk["time_since_last_click"], errors="coerce").fillna(0).to_numpy(dtype=np.float32)
        X_chunk[:, -1] = np.log1p(np.maximum(tslc, 0))

        X[idx] = X_chunk

        # --- update per-arm ridge stats and cost stats ---
        X64 = X_chunk.astype(np.float64, copy=False)
        r64 = r.astype(np.float64, copy=False)
        a32 = a.astype(np.int32, copy=False)

        # loop only over arms present in the chunk (faster than range(k_cap))
        for aa in np.unique(a32):
            m = (a32 == aa)
            if not m.any():
                continue
            Xa = X64[m]
            ra = r64[m]
            XtX_arm[aa] += Xa.T @ Xa
            XtR_arm[aa] += Xa.T @ ra
            cnt_arm[aa] += int(m.sum())
            sum_cost_arm[aa] += float(cost[m].sum())

        write += n_take
        if write % (args.chunksize * 5) == 0 or write == n:
            print(f"processed {write}/{n}")

    # flush memmaps
    X.flush(); A.flush(); R.flush(); C.flush(); D.flush()

    # finalize derived arrays
    # mean cost per arm (avoid division by 0)
    costs_by_arm = (sum_cost_arm / np.maximum(cnt_arm, 1)).astype(np.float32)
    np.save(out_dir / "costs_by_arm.npy", costs_by_arm)

    if len(delays_pos_parts) > 0:
        delays_pos = np.concatenate(delays_pos_parts, axis=0).astype(np.int32, copy=False)
    else:
        delays_pos = np.zeros((0,), dtype=np.int32)
    np.save(out_dir / "delays_pos.npy", delays_pos)

    meta = {
        "n": int(write), "d": int(d), "k_cap": int(k_cap),
        "d_hash": int(args.d_hash),
        "delta_seconds": int(delta),
        "censor_seconds": int(W),
        "censor_steps": int(censor_steps),
        "d_max": int(d_max),
        "inp": args.inp
    }
    np.savez(
        out_dir / "meta_and_stats.npz",
        meta=meta,
        XtX_arm=XtX_arm,
        XtR_arm=XtR_arm,
        cnt_arm=cnt_arm,
        sum_cost_arm=sum_cost_arm,
    )

    print("DONE. Saved memmaps to:", out_dir)
    print("Rows written:", write)
    print("Saved:", out_dir / "costs_by_arm.npy")
    print("Saved:", out_dir / "delays_pos.npy")
    print("Saved:", out_dir / "meta_and_stats.npz")


if __name__ == "__main__":
    main()
