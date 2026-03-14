from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

import src.eval.mpl_setup  # noqa: F401
import matplotlib.pyplot as plt
from src.env.sim_bandit_env import SimBanditEnv


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"Empty rows for {path}")
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write(
            "Model & Brier & LogLoss & Mean pred & Mean true & Delay KS & "
            "Arm-delay MAE & Arm-cost MAE \\\\\n"
        )
        f.write("\\midrule\n")
        for row in rows:
            model = str(row["model"]).replace("_", "\\_")
            f.write(
                f"{model} & "
                f"{row['brier']:.6f} & {row['logloss']:.6f} & "
                f"{row['mean_pred']:.6f} & {row['mean_true']:.6f} & "
                f"{row['delay_ks']:.6f} & {row['arm_delay_mae']:.3f} & {row['arm_cost_mae']:.6f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def resolve_indices(split_arr: np.ndarray | None, split_name: str, n: int) -> np.ndarray:
    if split_name == "all":
        idx = np.arange(n, dtype=np.int64)
    elif split_arr is None:
        raise ValueError(f"Requested split={split_name}, but split.npy is not available.")
    else:
        code = 0 if split_name == "train" else 1
        idx = np.flatnonzero(np.asarray(split_arr[:], dtype=np.uint8) == code)
    if n <= np.iinfo(np.uint32).max:
        idx = idx.astype(np.uint32, copy=False)
    return idx


def normalize_arm_means(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    mask = counts > 0
    out = np.zeros_like(values, dtype=np.float64)
    if not bool(np.any(mask)):
        return out
    mean_value = float(np.sum(values[mask] * counts[mask]) / np.sum(counts[mask]))
    out[mask] = values[mask] / max(mean_value, 1e-12)
    return out


def collect_reward_metrics(
    env: SimBanditEnv,
    X: np.ndarray,
    A: np.ndarray,
    R: np.ndarray,
    idx: np.ndarray,
    batch: int,
    n_bins: int,
) -> tuple[dict, np.ndarray, np.ndarray]:
    total = 0
    sum_brier = 0.0
    sum_logloss = 0.0
    sum_pred = 0.0
    sum_true = 0.0
    preds_parts: list[np.ndarray] = []
    truth_parts: list[np.ndarray] = []

    for start in range(0, idx.shape[0], batch):
        batch_idx = idx[start:start + batch]
        Xb = np.asarray(X[batch_idx], dtype=np.float32)
        Ab = np.asarray(A[batch_idx], dtype=np.int64)
        Rb = np.asarray(R[batch_idx], dtype=np.float64)
        Pb = np.clip(env.predict_prob_batch(Xb, Ab), 1e-9, 1.0 - 1e-9)

        total += int(batch_idx.shape[0])
        sum_brier += float(np.sum((Pb - Rb) ** 2))
        sum_logloss += float(-np.sum(Rb * np.log(Pb) + (1.0 - Rb) * np.log(1.0 - Pb)))
        sum_pred += float(np.sum(Pb))
        sum_true += float(np.sum(Rb))
        preds_parts.append(Pb.astype(np.float32, copy=False))
        truth_parts.append(Rb.astype(np.float32, copy=False))

    metrics = {
        "brier": sum_brier / max(total, 1),
        "logloss": sum_logloss / max(total, 1),
        "mean_pred": sum_pred / max(total, 1),
        "mean_true": sum_true / max(total, 1),
    }
    preds = np.concatenate(preds_parts, axis=0).astype(np.float64, copy=False)
    truths = np.concatenate(truth_parts, axis=0).astype(np.float64, copy=False)
    return metrics, preds, truths


def calibration_curve_quantile(
    preds: np.ndarray,
    truths: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    quantiles = np.linspace(0.0, 1.0, int(n_bins) + 1)
    edges = np.quantile(preds, quantiles)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.asarray([float(np.mean(preds))]), np.asarray([float(np.mean(truths))]), np.asarray([int(preds.size)])

    bins = np.searchsorted(edges[1:-1], preds, side="right")
    n_eff = edges.size - 1
    mean_pred = np.zeros((n_eff,), dtype=np.float64)
    mean_true = np.zeros((n_eff,), dtype=np.float64)
    counts = np.zeros((n_eff,), dtype=np.int64)

    for b in range(n_eff):
        mask = (bins == b)
        if not bool(np.any(mask)):
            continue
        counts[b] = int(np.sum(mask))
        mean_pred[b] = float(np.mean(preds[mask]))
        mean_true[b] = float(np.mean(truths[mask]))

    keep = counts > 0
    return mean_pred[keep], mean_true[keep], counts[keep]


def summarize_arm_stat_from_indices(
    A: np.ndarray,
    values: np.ndarray,
    idx: np.ndarray,
    K: int,
    batch: int,
    positive_mask_source: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros((K,), dtype=np.float64)
    counts = np.zeros((K,), dtype=np.int64)
    for start in range(0, idx.shape[0], batch):
        batch_idx = idx[start:start + batch]
        if positive_mask_source is not None:
            keep = np.asarray(positive_mask_source[batch_idx], dtype=np.uint8) == 1
            if not bool(np.any(keep)):
                continue
            batch_idx = batch_idx[keep]
        arms = np.asarray(A[batch_idx], dtype=np.int64)
        vals = np.asarray(values[batch_idx], dtype=np.float64)
        if arms.size == 0:
            continue
        np.add.at(sums, arms, vals)
        np.add.at(counts, arms, 1)
    means = sums / np.maximum(counts, 1)
    return means, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap_dir", type=str, required=True)
    ap.add_argument("--artifact_dir", type=str, default="paper_artifacts")
    ap.add_argument("--batch", type=int, default=200_000)
    ap.add_argument("--n_bins", type=int, default=10)
    ap.add_argument(
        "--eval_split",
        type=str,
        default="auto",
        choices=["auto", "all", "train", "test"],
        help="Which held-out rows to diagnose on. 'auto' uses test if split metadata exists.",
    )
    ap.add_argument(
        "--reward_models",
        type=str,
        default="auto",
        help="Comma-separated list: auto, linear_clip, logistic.",
    )
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    fig_dir = artifact_dir / "figures"
    tab_dir = artifact_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    memmap_dir = Path(args.memmap_dir)
    meta_stats = np.load(memmap_dir / "meta_and_stats.npz", allow_pickle=True)
    meta = meta_stats["meta"].item()

    X = np.load(memmap_dir / "X.npy", mmap_mode="r")
    A = np.load(memmap_dir / "A.npy", mmap_mode="r")
    R = np.load(memmap_dir / "R.npy", mmap_mode="r")
    C = np.load(memmap_dir / "C.npy", mmap_mode="r")
    D = np.load(memmap_dir / "D.npy", mmap_mode="r")
    split_arr = np.load(memmap_dir / "split.npy", mmap_mode="r") if (memmap_dir / "split.npy").exists() else None
    K = int(meta["k_cap"])
    censor_steps = int(meta["censor_steps"])

    eval_split = args.eval_split
    if eval_split == "auto":
        eval_split = "test" if split_arr is not None and int(meta.get("n_test", 0)) > 0 else "all"
    fit_split = str(meta.get("fit_split", "all"))

    idx_eval = resolve_indices(split_arr, eval_split, int(meta["n"]))
    idx_fit = resolve_indices(split_arr, fit_split, int(meta["n"]))

    reward_models = [x.strip() for x in args.reward_models.split(",") if x.strip()]
    if not reward_models:
        raise RuntimeError("Empty --reward_models")
    available_models = {"linear_clip"}
    logistic_path = memmap_dir / "arm_logistic_stats.npz"
    if logistic_path.exists():
        available_models.add("logistic")
    if reward_models == ["auto"]:
        reward_models = ["linear_clip"]
        if "logistic" in available_models:
            reward_models.append("logistic")
    missing_models = [model for model in reward_models if model not in available_models]
    if missing_models:
        if missing_models == ["logistic"]:
            raise SystemExit(
                "Requested reward_model=logistic for diagnostics, but "
                f"{logistic_path} does not exist.\n"
                "Fit it first with:\n"
                "  python -m src.env.compute_arm_logistic_stats_from_memmap "
                f"--memmap_dir {memmap_dir} --split train --lam 1.0 --max_iter 8 --out arm_logistic_stats.npz\n"
                "Or run diagnostics with:\n"
                "  --reward_models linear_clip"
            )
        raise SystemExit(
            f"Requested unsupported or unavailable reward_models={missing_models}. "
            f"Available models: {sorted(available_models)}"
        )

    rows = []
    calibration_curves: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []

    fit_pos_mask = np.asarray(R[idx_fit], dtype=np.uint8) == 1
    eval_pos_mask = np.asarray(R[idx_eval], dtype=np.uint8) == 1
    fit_pos_delays = np.asarray(D[idx_fit][fit_pos_mask], dtype=np.int64)
    eval_pos_delays = np.asarray(D[idx_eval][eval_pos_mask], dtype=np.int64)

    fit_hist = np.bincount(np.clip(fit_pos_delays, 0, censor_steps), minlength=censor_steps + 1).astype(np.float64)
    eval_hist = np.bincount(np.clip(eval_pos_delays, 0, censor_steps), minlength=censor_steps + 1).astype(np.float64)
    fit_cdf = np.cumsum(fit_hist) / max(np.sum(fit_hist), 1.0)
    eval_cdf = np.cumsum(eval_hist) / max(np.sum(eval_hist), 1.0)
    delay_ks = float(np.max(np.abs(fit_cdf - eval_cdf))) if fit_cdf.size else 0.0

    fit_delay_means, fit_delay_counts = summarize_arm_stat_from_indices(
        A,
        D,
        idx_fit,
        K,
        batch=int(args.batch),
        positive_mask_source=R,
    )
    eval_delay_means, eval_delay_counts = summarize_arm_stat_from_indices(
        A,
        D,
        idx_eval,
        K,
        batch=int(args.batch),
        positive_mask_source=R,
    )
    common_delay = (fit_delay_counts > 0) & (eval_delay_counts > 0)
    arm_delay_mae = float(np.mean(np.abs(fit_delay_means[common_delay] - eval_delay_means[common_delay]))) if bool(np.any(common_delay)) else 0.0

    fit_costs = np.load(memmap_dir / "costs_by_arm.npy").astype(np.float64)
    eval_cost_means, eval_cost_counts = summarize_arm_stat_from_indices(
        A, C, idx_eval, K, batch=int(args.batch)
    )
    eval_costs_norm = normalize_arm_means(eval_cost_means, eval_cost_counts)
    common_cost = eval_cost_counts > 0
    arm_cost_mae = float(np.mean(np.abs(fit_costs[common_cost] - eval_costs_norm[common_cost]))) if bool(np.any(common_cost)) else 0.0

    for reward_model in reward_models:
        env = SimBanditEnv.from_memmap_dir(
            str(memmap_dir),
            seed=0,
            context_split="all",
            reward_model=reward_model,
        )
        metrics, preds, truths = collect_reward_metrics(
            env=env,
            X=X,
            A=A,
            R=R,
            idx=idx_eval,
            batch=int(args.batch),
            n_bins=int(args.n_bins),
        )
        pred_curve, true_curve, counts_curve = calibration_curve_quantile(
            preds=preds,
            truths=truths,
            n_bins=int(args.n_bins),
        )
        model_name = str(env.meta.get("reward_model_loaded", env.reward_model))
        rows.append({
            "model": model_name,
            "brier": metrics["brier"],
            "logloss": metrics["logloss"],
            "mean_pred": metrics["mean_pred"],
            "mean_true": metrics["mean_true"],
            "delay_ks": delay_ks,
            "arm_delay_mae": arm_delay_mae,
            "arm_cost_mae": arm_cost_mae,
        })
        calibration_curves.append((model_name, pred_curve, true_curve, counts_curve))

    plt.figure()
    plt.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="black", linewidth=1.0, label="ideal")
    for label, pred_curve, true_curve, counts_curve in calibration_curves:
        yerr = 1.96 * np.sqrt(np.clip(true_curve * (1.0 - true_curve) / np.maximum(counts_curve, 1), 0.0, None))
        plt.errorbar(pred_curve, true_curve, yerr=yerr, marker="o", capsize=3, linewidth=1.5, label=label)
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Empirical positive rate")
    plt.title(f"Simulator calibration on {eval_split} rows (quantile bins)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "simulator_calibration.png", dpi=220)
    plt.close()

    plt.figure()
    plt.plot(np.arange(censor_steps + 1), fit_cdf, label=f"{fit_split} positives")
    plt.plot(np.arange(censor_steps + 1), eval_cdf, label=f"{eval_split} positives")
    plt.xlabel("Delay (steps)")
    plt.ylabel("Empirical CDF")
    plt.title("Positive-delay distribution: fit vs held-out")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "simulator_delay_cdf.png", dpi=220)
    plt.close()

    plt.figure()
    if bool(np.any(common_cost)):
        plt.scatter(fit_costs[common_cost], eval_costs_norm[common_cost], s=25 + 0.03 * eval_cost_counts[common_cost])
    lo = float(min(np.min(fit_costs[common_cost]) if bool(np.any(common_cost)) else 0.0, np.min(eval_costs_norm[common_cost]) if bool(np.any(common_cost)) else 0.0))
    hi = float(max(np.max(fit_costs[common_cost]) if bool(np.any(common_cost)) else 1.0, np.max(eval_costs_norm[common_cost]) if bool(np.any(common_cost)) else 1.0))
    plt.plot([lo, hi], [lo, hi], linestyle="--", color="black", linewidth=1.0)
    plt.xlabel("Train normalized arm cost")
    plt.ylabel(f"{eval_split} normalized arm cost")
    plt.title("Arm-level cost stability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "simulator_cost_train_vs_eval.png", dpi=220)
    plt.close()

    plt.figure()
    if bool(np.any(common_delay)):
        plt.scatter(
            fit_delay_means[common_delay],
            eval_delay_means[common_delay],
            s=25 + 0.03 * np.minimum(fit_delay_counts[common_delay], eval_delay_counts[common_delay]),
        )
    dlo = float(min(np.min(fit_delay_means[common_delay]) if bool(np.any(common_delay)) else 0.0, np.min(eval_delay_means[common_delay]) if bool(np.any(common_delay)) else 0.0))
    dhi = float(max(np.max(fit_delay_means[common_delay]) if bool(np.any(common_delay)) else 1.0, np.max(eval_delay_means[common_delay]) if bool(np.any(common_delay)) else 1.0))
    plt.plot([dlo, dhi], [dlo, dhi], linestyle="--", color="black", linewidth=1.0)
    plt.xlabel("Train mean positive delay by arm")
    plt.ylabel(f"{eval_split} mean positive delay by arm")
    plt.title("Arm-conditional delay stability")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / "simulator_delay_by_arm.png", dpi=220)
    plt.close()

    write_csv(tab_dir / "simulator_diagnostics.csv", rows)
    write_tex(tab_dir / "simulator_diagnostics.tex", rows)
    print("Wrote:", tab_dir / "simulator_diagnostics.csv")
    print("Wrote:", tab_dir / "simulator_diagnostics.tex")
    print("Saved:", fig_dir / "simulator_calibration.png")
    print("Saved:", fig_dir / "simulator_delay_cdf.png")
    print("Saved:", fig_dir / "simulator_cost_train_vs_eval.png")
    print("Saved:", fig_dir / "simulator_delay_by_arm.png")


if __name__ == "__main__":
    main()
