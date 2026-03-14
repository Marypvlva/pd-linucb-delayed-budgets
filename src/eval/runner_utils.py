from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Any

import numpy as np

from src.env.sim_bandit_env import SimBanditEnv


def feasible_mask(costs: np.ndarray, remaining_budget: float) -> np.ndarray:
    rem = float(remaining_budget)
    return costs.astype(np.float64) <= (rem + 1e-12)


def t_crit_975(df: int) -> float:
    # 95% CI => t_{0.975, df}
    try:
        import scipy.stats as st  # optional
        return float(st.t.ppf(0.975, df))
    except Exception:
        return 1.96


def mean_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    n = int(arr.size)
    if n <= 0:
        raise ValueError("mean_ci: empty input")
    m = float(arr.mean())
    if n == 1:
        return m, 0.0
    s = float(arr.std(ddof=1))
    ci = t_crit_975(n - 1) * s / sqrt(n)
    return m, ci


@dataclass(frozen=True)
class RunResult:
    cum_r: np.ndarray
    cum_c: np.ndarray
    reward: float
    cost: float
    spent_ratio: float
    t_stop: int


def run_contextual_delayed(
    env: SimBanditEnv,
    X_seq: np.ndarray,
    algo: Any,
    budget_ratio: float,
    stop_at_budget: bool,
    *,
    is_primal_dual: bool,
) -> RunResult:
    """
    Generic contextual runner for algorithms with:
      - select(x) / select_feasible(x, feasible_mask)
      - update_design(a, x)
      - update_reward(a, x, r)
    and optionally (if is_primal_dual): update_dual(cost, b_t)
    """
    T = int(len(X_seq))
    B = float(budget_ratio) * float(T)
    costs = env.costs

    pending: dict[int, list[tuple[int, np.ndarray, float]]] = {}
    cum_r = np.zeros(T, dtype=np.float64)
    cum_c = np.zeros(T, dtype=np.float64)

    total_r = 0.0
    total_c = 0.0
    t_stop = T

    for t in range(T):
        for (a_upd, x_upd, r_upd) in pending.pop(t, []):
            algo.update_reward(int(a_upd), x_upd, float(r_upd))

        x = X_seq[t]

        if stop_at_budget:
            feas = feasible_mask(costs, B - total_c)
            if not bool(np.any(feas)):
                t_stop = t
                cum_r[t:] = total_r
                cum_c[t:] = total_c
                break
            a = int(algo.select_feasible(x, feas))
        else:
            a = int(algo.select(x))

        c = float(costs[a])

        if is_primal_dual:
            H = T - t
            b_t = (B - total_c) / max(H, 1)
            algo.update_dual(c, b_t)

        algo.update_design(a, x)

        r, _, dly = env.step(x, a)

        total_c += c
        total_r += float(r)
        cum_r[t] = total_r
        cum_c[t] = total_c

        dly = int(dly)
        if dly <= 0:
            algo.update_reward(a, x, float(r))
        else:
            t_due = t + dly
            if t_due < T:
                pending.setdefault(t_due, []).append((a, x, float(r)))

    spent_ratio = total_c / max(B, 1e-12)
    return RunResult(
        cum_r=cum_r,
        cum_c=cum_c,
        reward=float(total_r),
        cost=float(total_c),
        spent_ratio=float(spent_ratio),
        t_stop=int(t_stop),
    )


def run_context_free_pd_delayed(
    env: SimBanditEnv,
    X_seq: np.ndarray,
    algo: Any,
    budget_ratio: float,
    stop_at_budget: bool,
) -> RunResult:
    """
    Runner for ContextFreePrimalDualBwK:
      - select(t) / select_feasible(t, feasible_mask)
      - update_dual(cost, b_t)
      - update(a, r)
    """
    T = int(len(X_seq))
    B = float(budget_ratio) * float(T)
    costs = env.costs

    pending: dict[int, list[tuple[int, float]]] = {}
    cum_r = np.zeros(T, dtype=np.float64)
    cum_c = np.zeros(T, dtype=np.float64)

    total_r = 0.0
    total_c = 0.0
    t_stop = T

    for t in range(T):
        for (a_upd, r_upd) in pending.pop(t, []):
            algo.update(int(a_upd), float(r_upd))

        if stop_at_budget:
            feas = feasible_mask(costs, B - total_c)
            if not bool(np.any(feas)):
                t_stop = t
                cum_r[t:] = total_r
                cum_c[t:] = total_c
                break
            a = int(algo.select_feasible(t, feas))
        else:
            a = int(algo.select(t))

        c = float(costs[a])

        H = T - t
        b_t = (B - total_c) / max(H, 1)
        algo.update_dual(c, b_t)

        r, _, dly = env.step(X_seq[t], a)

        total_c += c
        total_r += float(r)
        cum_r[t] = total_r
        cum_c[t] = total_c

        dly = int(dly)
        if dly <= 0:
            algo.update(a, float(r))
        else:
            t_due = t + dly
            if t_due < T:
                pending.setdefault(t_due, []).append((a, float(r)))

    spent_ratio = total_c / max(B, 1e-12)
    return RunResult(
        cum_r=cum_r,
        cum_c=cum_c,
        reward=float(total_r),
        cost=float(total_c),
        spent_ratio=float(spent_ratio),
        t_stop=int(t_stop),
    )
