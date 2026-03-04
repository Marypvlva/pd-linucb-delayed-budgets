# src/env/make_obd_from_repo.py
import numpy as np
import pandas as pd
from pathlib import Path

RAW = Path("data/raw/obd")
OUT = Path("data/processed/obd_feedback.npz")

# У тебя папки именно так: data/raw/obd/bts/all/all.csv и item_context.csv рядом
BEHAVIOR_POLICY = "bts"   # "bts" или "random"
CAMPAIGN = "all"          # "all" / "men" / "women"
MAX_ROWS = None           # например 200000 для быстрого прогона


def parse_vector_cell(s: str) -> np.ndarray:
    """Parse string like '[1 2 3]' or '1,2,3' into numeric vector."""
    s = str(s).strip().replace("[", "").replace("]", "").replace(",", " ")
    return np.fromstring(s, sep=" ")


def to_float_matrix(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    """
    Convert selected columns to float matrix:
      - if column is numeric -> float
      - else (strings/categories/hashes) -> factorize to integer codes -> float
    """
    mats = []
    for c in cols:
        s = df[c]

        # try numeric
        sn = pd.to_numeric(s, errors="coerce")
        if sn.notna().all():
            mats.append(sn.to_numpy(dtype=np.float32).reshape(-1, 1))
        else:
            codes, _ = pd.factorize(s.astype(str), sort=True)
            mats.append(codes.astype(np.float32).reshape(-1, 1))

    if not mats:
        return np.zeros((len(df), 0), dtype=np.float32)
    return np.concatenate(mats, axis=1).astype(np.float32)


def main():
    folder = RAW / BEHAVIOR_POLICY / CAMPAIGN
    data_csv = folder / f"{CAMPAIGN}.csv"          # all.csv / men.csv / women.csv
    item_csv = folder / "item_context.csv"

    if not folder.exists():
        raise FileNotFoundError(f"Нет папки: {folder}")
    if not data_csv.exists():
        raise FileNotFoundError(f"Нет файла: {data_csv}")
    if not item_csv.exists():
        raise FileNotFoundError(f"Нет файла: {item_csv}")

    df = pd.read_csv(data_csv, nrows=MAX_ROWS)
    item = pd.read_csv(item_csv)

    print("Using:", data_csv)
    print("Rows:", len(df))
    print("Columns (first 30):", list(df.columns)[:30])

    # ===== reward =====
    if "click" in df.columns:
        R = df["click"].to_numpy(dtype=np.float32)
    elif "reward" in df.columns:
        R = df["reward"].to_numpy(dtype=np.float32)
    else:
        raise KeyError(f"Не нашла click/reward. Колонки: {list(df.columns)}")

    # ===== action =====
    if "item_id" in df.columns:
        A = df["item_id"].to_numpy(dtype=np.int64)
    elif "action" in df.columns:
        A = df["action"].to_numpy(dtype=np.int64)
    else:
        raise KeyError(f"Не нашла item_id/action. Колонки: {list(df.columns)}")

    # ===== pscore =====
    pscore = None
    if "action_prob" in df.columns:
        pscore = df["action_prob"].to_numpy(dtype=np.float32)
    elif "pscore" in df.columns:
        pscore = df["pscore"].to_numpy(dtype=np.float32)
    elif "propensity_score" in df.columns:
        pscore = df["propensity_score"].to_numpy(dtype=np.float32)

    # ===== user features =====
    user_cols = [c for c in df.columns if c.startswith("user_feature")]
    if user_cols:
        U = to_float_matrix(df, user_cols)
    elif "user_features" in df.columns:
        vecs = [parse_vector_cell(s) for s in df["user_features"].astype(str).tolist()]
        d0 = max(v.size for v in vecs) if vecs else 1
        U = np.zeros((len(vecs), d0), dtype=np.float32)
        for i, v in enumerate(vecs):
            U[i, :v.size] = v.astype(np.float32)
    else:
        U = np.ones((len(df), 1), dtype=np.float32)

    # ===== position =====
    pos = df["position"].to_numpy(dtype=np.float32).reshape(-1, 1) if "position" in df.columns else None

    # ===== item features from item_context.csv =====
    I = None
    item_feat_cols = [c for c in item.columns if c.startswith("item_feature")]
    if "item_id" in item.columns and item_feat_cols and "item_id" in df.columns:
        item_small = item[["item_id"] + item_feat_cols].copy()
        df_item = df[["item_id"]].merge(item_small, on="item_id", how="left")
        # item_feature_* тоже могут быть строками => factorize/numeric
        I = to_float_matrix(df_item, item_feat_cols)
        print("Loaded item features:", I.shape[1])

    # ===== build X =====
    parts = [U]
    if pos is not None:
        parts.append(pos.astype(np.float32))
    if I is not None:
        parts.append(I.astype(np.float32))
    X = np.concatenate(parts, axis=1).astype(np.float32)

    n = len(df)
    C = np.ones((n, 1), dtype=np.float32)  # MVP cost=1 per round
    D = -np.ones(n, dtype=np.int64)        # no delays yet

    OUT.parent.mkdir(parents=True, exist_ok=True)
    if pscore is None:
        np.savez(OUT, X=X, A=A, R=R, C=C, D=D)
    else:
        np.savez(OUT, X=X, A=A, R=R, C=C, D=D, pscore=pscore)

    print("Saved:", OUT)
    print("Shapes:",
          "X", X.shape,
          "A", A.shape,
          "R", R.shape,
          "C", C.shape,
          "D", D.shape,
          "pscore", None if pscore is None else pscore.shape)


if __name__ == "__main__":
    main()
