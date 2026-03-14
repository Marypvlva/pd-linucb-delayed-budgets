# src/eval/run_delay_ablation.py
from __future__ import annotations

import sys
from pathlib import Path
_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import argparse
import numpy as np

from src.env.sim_bandit_env import SimBanditEnv
from src.algos.linucb_pd_delayed import DisjointLinUCB, PrimalDualLinUCB
from src.algos.cost_normalized_ucb import CostNormalizedDisjointUCB
from src.algos.context_free_bwk import ContextFreePrimalDualBwK
from src.eval.runner_utils import mean_ci, run_context_free_pd_delayed, run_contextual_delayed


def write_delay_ablation_tabular(tex_path: Path, rows: list[dict]):
    tex_path.parent.mkdir(parents=True, exist_ok=True)
    with tex_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lrrrrrr|rrrrrr}\n")
        f.write("\\toprule\n")
        f.write(" & \\multicolumn{6}{c}{Delayed} & \\multicolumn{6}{c}{No-delay ($D_t\\equiv0$)} \\\\\n")
        f.write("\\cmidrule(lr){2-7} \\cmidrule(lr){8-13}\n")
        f.write("Method & Reward & CI & Spent$/B$ & CI & $\\\\tau$ & CI & Reward & CI & Spent$/B$ & CI & $\\\\tau$ & CI \\\\\n")
        f.write("\\midrule\n")
        for r in rows:
            f.write(
                f"{r['method']} & "
                f"{r['reward_delayed_mean']:.1f} & {r['reward_delayed_ci']:.1f} & "
                f"{r['spent_delayed_mean']:.6f} & {r['spent_delayed_ci']:.6f} & "
                f"{r['tau_delayed_mean']:.1f} & {r['tau_delayed_ci']:.1f} & "
                f"{r['reward_nodelay_mean']:.1f} & {r['reward_nodelay_ci']:.1f} & "
                f"{r['spent_nodelay_mean']:.6f} & {r['spent_nodelay_ci']:.6f} & "
                f"{r['tau_nodelay_mean']:.1f} & {r['tau_nodelay_ci']:.1f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memmap_dir", type=str, required=True)
    ap.add_argument("--T", type=int, default=5000)
    ap.add_argument("--budget_ratio", type=float, default=0.7)
    ap.add_argument("--stop_at_budget", action="store_true")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--seed0", type=int, default=123)
    ap.add_argument("--env_seed", type=int, default=0)

    ap.add_argument("--alpha_lin", type=float, default=1.0)
    ap.add_argument("--alpha_pd", type=float, default=1.5)
    ap.add_argument("--eta_pd", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--alpha_cnu", type=float, default=1.0)
    ap.add_argument("--mode_cnu", type=str, default="ratio", choices=["ratio", "sub"])
    ap.add_argument("--gamma_cnu", type=float, default=1.0)
    ap.add_argument("--eps_cnu", type=float, default=1e-3)
    ap.add_argument("--alpha_cf", type=float, default=1.0)
    ap.add_argument("--eta_cf", type=float, default=0.05)

    ap.add_argument("--artifact_dir", type=str, default="paper_artifacts")
    args = ap.parse_args()

    artifact_dir = Path(args.artifact_dir)
    tab_dir = artifact_dir / "tables"
    tab_dir.mkdir(parents=True, exist_ok=True)

    env_base = SimBanditEnv.from_memmap_dir(args.memmap_dir, seed=args.env_seed, ridge_lambda=1.0, cost_mode="lin")
    seeds = [int(args.seed0 + i) for i in range(int(args.n_seeds))]

    store = {
        "delayed": {"lin": [], "pd": [], "cnu": [], "cf": []},
        "nodelay": {"lin": [], "pd": [], "cnu": [], "cf": []},
    }

    for seed in seeds:
        X_seq = env_base.sample_contexts(args.T, rng=np.random.default_rng(seed))

        for mode in ["delayed", "nodelay"]:
            if mode == "delayed":
                env_lin = env_base.clone(seed=seed + 10_001)
                env_pd = env_base.clone(seed=seed + 10_002)
                env_cnu = env_base.clone(seed=seed + 10_003)
                env_cf = env_base.clone(seed=seed + 10_004)
            else:
                env_lin = env_base.make_no_delay(seed=seed + 20_001)
                env_pd = env_base.make_no_delay(seed=seed + 20_002)
                env_cnu = env_base.make_no_delay(seed=seed + 20_003)
                env_cf = env_base.make_no_delay(seed=seed + 20_004)

            lin = DisjointLinUCB(env_base.K, env_base.d, alpha=args.alpha_lin, lam=args.lam, seed=seed + 101)
            pd = PrimalDualLinUCB(
                env_base.K,
                env_base.d,
                costs=env_base.costs,
                alpha=args.alpha_pd,
                lam=args.lam,
                eta=args.eta_pd,
                seed=seed + 102,
            )
            cnu = CostNormalizedDisjointUCB(
                env_base.K,
                env_base.d,
                costs=env_base.costs,
                alpha=args.alpha_cnu,
                lam=args.lam,
                mode=args.mode_cnu,
                gamma=args.gamma_cnu,
                eps=args.eps_cnu,
                seed=seed + 103,
            )
            cf = ContextFreePrimalDualBwK(
                env_base.K,
                costs=env_base.costs,
                alpha=args.alpha_cf,
                eta=args.eta_cf,
                seed=seed + 104,
            )

            res_lin = run_contextual_delayed(
                env_lin, X_seq, lin, args.budget_ratio, bool(args.stop_at_budget), is_primal_dual=False
            )
            res_pd = run_contextual_delayed(
                env_pd, X_seq, pd, args.budget_ratio, bool(args.stop_at_budget), is_primal_dual=True
            )
            res_cnu = run_contextual_delayed(
                env_cnu, X_seq, cnu, args.budget_ratio, bool(args.stop_at_budget), is_primal_dual=False
            )
            res_cf = run_context_free_pd_delayed(env_cf, X_seq, cf, args.budget_ratio, bool(args.stop_at_budget))

            store[mode]["lin"].append(res_lin)
            store[mode]["pd"].append(res_pd)
            store[mode]["cnu"].append(res_cnu)
            store[mode]["cf"].append(res_cf)

        print(f"done seed={seed}")

    def summarize(mode: str, key: str):
        rr = store[mode][key]
        r_m, r_ci = mean_ci([x.reward for x in rr])
        s_m, s_ci = mean_ci([x.spent_ratio for x in rr])
        t_m, t_ci = mean_ci([float(x.t_stop) for x in rr])
        return r_m, r_ci, s_m, s_ci, t_m, t_ci

    rows = []
    for key, name in [
        ("lin", "LinUCB"),
        ("pd", "PD-LinUCB"),
        ("cnu", f"CostNormUCB[{args.mode_cnu}]"),
        ("cf", "CF-PD-BwK"),
    ]:
        rd_m, rd_ci, sd_m, sd_ci, td_m, td_ci = summarize("delayed", key)
        rn_m, rn_ci, sn_m, sn_ci, tn_m, tn_ci = summarize("nodelay", key)
        rows.append({
            "method": name,
            "reward_delayed_mean": rd_m,
            "reward_delayed_ci": rd_ci,
            "spent_delayed_mean": sd_m,
            "spent_delayed_ci": sd_ci,
            "tau_delayed_mean": td_m,
            "tau_delayed_ci": td_ci,
            "reward_nodelay_mean": rn_m,
            "reward_nodelay_ci": rn_ci,
            "spent_nodelay_mean": sn_m,
            "spent_nodelay_ci": sn_ci,
            "tau_nodelay_mean": tn_m,
            "tau_nodelay_ci": tn_ci,
        })

    out_path = tab_dir / "delay_ablation.tex"
    write_delay_ablation_tabular(out_path, rows)
    print("Wrote:", out_path)


if __name__ == "__main__":
    main()
