# src/algos/context_free_bwk.py
from __future__ import annotations
import numpy as np


class ContextFreePrimalDualBwK:
    """
    Context-free BwK baseline (ignores x):
      score = (mu_hat + bonus) - lambda * cost
      lambda <- [lambda + eta*(cost - b_t)]_+

    IMPORTANT: b_t is dynamic and must be computed in runner from remaining budget.
    """

    def __init__(
        self,
        n_arms: int,
        costs: np.ndarray,
        alpha: float = 1.0,
        eta: float = 0.05,
        seed: int = 0,
    ):
        self.K = int(n_arms)
        self.costs = costs.astype(np.float64)
        self.alpha = float(alpha)
        self.eta = float(eta)
        self.dual = 0.0
        self.rng = np.random.default_rng(seed)

        self.n = np.zeros(self.K, dtype=np.int64)
        self.sum_r = np.zeros(self.K, dtype=np.float64)

    def select(self, t: int) -> int:
        tt = float(t + 1)
        mu = self.sum_r / np.maximum(self.n, 1)
        bonus = self.alpha * np.sqrt(np.log(tt + 1.0) / np.maximum(self.n, 1))
        score = (mu + bonus) - self.dual * self.costs
        return int(np.argmax(score))

    def update_dual(self, cost: float, b_t: float):
        self.dual = max(0.0, self.dual + self.eta * (float(cost) - float(b_t)))

    def update(self, a: int, r: float):
        a = int(a)
        self.n[a] += 1
        self.sum_r[a] += float(r)