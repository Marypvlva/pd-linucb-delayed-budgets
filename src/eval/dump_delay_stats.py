# src/eval/dump_delay_stats.py
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap_dir", type=str, required=True)
    ap.add_argument("--out_tex", type=str, default="tables/delay_stats.tex")
    args = ap.parse_args()

    p = Path(args.memmap_dir)

    meta_stats = np.load(p / "meta_and_stats.npz", allow_pickle=True)
    meta = meta_stats["meta"].item()

    # --- main arrays ---
    R = np.load(p / "R.npy", mmap_mode="r")  # conversions (0/1)
    D = np.load(p / "D.npy", mmap_mode="r")  # delays in steps, can contain -1

    n = int(meta.get("n", R.shape[0]))
    R = R[:n]
    D = D[:n]

    # --- pull time/censor params if present in meta (optional) ---
    delta_seconds = meta.get("delta_seconds", None)
    censor_seconds = meta.get("censor_seconds", None)
    censor_steps = meta.get("censor_steps", None)
    D_max = meta.get("max_delay", meta.get("d_max", None))

    # --- basic rates ---
    conv_rate = float(np.mean(R))
    # among conversions, how many have a valid observed delay in-window
    conv_mask = (R == 1)
    conv_total = int(np.sum(conv_mask))
    conv_observed = int(np.sum((conv_mask) & (D >= 0)))
    frac_observed_among_conversions = float(conv_observed / max(conv_total, 1))

    # positive delays pool (prefer precomputed)
    delays_path = p / "delays_pos.npy"
    if delays_path.exists():
        delays = np.load(delays_path).astype(np.int64).reshape(-1)
        delays = delays[delays >= 0]
        source = "delays_pos.npy"
    else:
        delays = np.asarray(D[D >= 0], dtype=np.int64)
        source = "D.npy (D>=0)"

    if delays.size == 0:
        raise RuntimeError("No non-negative delays found (delays pool is empty).")

    # percentiles
    p50, p75, p90, p95, p99 = np.percentile(delays, [50, 75, 90, 95, 99])
    dmin = int(np.min(delays))
    dmax = int(np.max(delays))
    dmean = float(np.mean(delays))

    # --- write LaTeX table ---
    out_tex = Path(args.out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)

    def fmt_opt(x):
        return "—" if x is None else str(x)

    with out_tex.open("w", encoding="utf-8") as f:
        f.write(r"\begin{table}[t]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Статистика задержек (в шагах) и параметры дискретизации/цензуры.}" + "\n")
        f.write(r"\label{tab:delay_stats}" + "\n")
        f.write(r"\begin{tabular}{ll}" + "\n")
        f.write(r"\toprule" + "\n")

        f.write(r"Параметр & Значение \\" + "\n")
        f.write(r"\midrule" + "\n")

        f.write(rf"Источник задержек & \texttt{{{source}}} \\" + "\n")
        f.write(rf"Число событий $n$ & {n} \\" + "\n")
        f.write(rf"Conversion rate $\mathbb{{P}}(r=1)$ & {conv_rate:.6f} \\" + "\n")
        f.write(rf"Доля конверсий с наблюдаемой задержкой & {frac_observed_among_conversions:.4f} \\" + "\n")

        f.write(rf"$\Delta$ (сек) & {fmt_opt(delta_seconds)} \\" + "\n")
        f.write(rf"$W$ (сек) & {fmt_opt(censor_seconds)} \\" + "\n")
        f.write(rf"$W/\Delta$ (шагов) & {fmt_opt(censor_steps)} \\" + "\n")
        f.write(rf"$D_{{\max}}$ (шагов) & {fmt_opt(D_max)} \\" + "\n")

        f.write(rf"Delay min / mean / max & {dmin} / {dmean:.2f} / {dmax} \\" + "\n")
        f.write(rf"Delay p50 / p75 & {p50:.0f} / {p75:.0f} \\" + "\n")
        f.write(rf"Delay p90 / p95 / p99 & {p90:.0f} / {p95:.0f} / {p99:.0f} \\" + "\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")

    print("Wrote:", out_tex)


if __name__ == "__main__":
    main()