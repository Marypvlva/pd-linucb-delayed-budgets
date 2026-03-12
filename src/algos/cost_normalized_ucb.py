from __future__ import annotations
import numpy as np


class CostNormalizedDisjointUCB:
    """
    Disjoint linear UCB with cost-aware scoring + OTF design update.

    mode="ratio": score = UCB / (cost + eps)
    mode="sub":   score = UCB - gamma * cost

    Delays: runner must call update_design immediately, update_reward on feedback arrival.
    """

    def __init__(
        self,
        n_arms: int,
        d: int,
        costs: np.ndarray,
        alpha: float = 1.0,
        lam: float = 1.0,
        mode: str = "ratio",
        gamma: float = 1.0,
        eps: float = 1e-3,
        seed: int = 0,
    ):
        self.K = int(n_arms)
        self.d = int(d)
        self.costs = costs.astype(np.float64)
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.mode = str(mode)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.rng = np.random.default_rng(seed)

        I = np.eye(self.d, dtype=np.float64)
        self.A_inv = np.stack([(1.0 / self.lam) * I for _ in range(self.K)], axis=0)
        self.b = np.zeros((self.K, self.d), dtype=np.float64)
        self.theta = np.zeros((self.K, self.d), dtype=np.float64)

    def select(self, x: np.ndarray) -> int:
        feasible = np.ones(self.K, dtype=bool)
        return self.select_feasible(x, feasible)

    def select_feasible(self, x: np.ndarray, feasible: np.ndarray) -> int:
        x64 = x.astype(np.float64)
        mu = self.theta @ x64
        v = self.A_inv @ x64
        sigma = np.sqrt(np.sum(v * x64[None, :], axis=1))
        ucb = mu + self.alpha * sigma

        if self.mode == "ratio":
            score = ucb / (self.costs + self.eps)
        elif self.mode == "sub":
            score = ucb - self.gamma * self.costs
        else:
            raise ValueError("mode must be 'ratio' or 'sub'")

        score = np.where(feasible, score, -np.inf)
        return int(np.argmax(score))

    def update_design(self, a: int, x: np.ndarray):
        a = int(a)
        x64 = x.astype(np.float64)
        Ainv = self.A_inv[a]
        v = Ainv @ x64
        denom = 1.0 + float(x64 @ v)
        self.A_inv[a] = Ainv - np.outer(v, v) / denom
        self.theta[a] = self.A_inv[a] @ self.b[a]

    def update_reward(self, a: int, x: np.ndarray, r: float):
        a = int(a)
        x64 = x.astype(np.float64)
        self.b[a] += float(r) * x64
        self.theta[a] = self.A_inv[a] @ self.b[a]