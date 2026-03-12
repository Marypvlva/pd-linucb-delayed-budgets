from __future__ import annotations
import numpy as np


class DisjointLinUCB:
    """
    Disjoint LinUCB with delayed feedback using OTF design update:
      - update_design(a, x): immediately updates A_inv[a] via Sherman–Morrison
      - update_reward(a, x, r): updates b[a] and theta[a] only (NO A update here)
    """

    def __init__(self, n_arms: int, d: int, alpha: float = 1.0, lam: float = 1.0, seed: int = 0):
        self.K = int(n_arms)
        self.d = int(d)
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.rng = np.random.default_rng(seed)

        I = np.eye(self.d, dtype=np.float64)
        self.A_inv = np.stack([(1.0 / self.lam) * I for _ in range(self.K)], axis=0)  # (K,d,d)
        self.b = np.zeros((self.K, self.d), dtype=np.float64)                        # (K,d)
        self.theta = np.zeros((self.K, self.d), dtype=np.float64)                    # (K,d)

    def select(self, x: np.ndarray) -> int:
        feasible = np.ones(self.K, dtype=bool)
        return self.select_feasible(x, feasible)

    def select_feasible(self, x: np.ndarray, feasible: np.ndarray) -> int:
        x64 = x.astype(np.float64)
        mu = self.theta @ x64
        v = self.A_inv @ x64
        sigma = np.sqrt(np.sum(v * x64[None, :], axis=1))
        ucb = mu + self.alpha * sigma
        ucb = np.where(feasible, ucb, -np.inf)
        return int(np.argmax(ucb))

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


class PrimalDualLinUCB:
    """
    Primal-Dual LinUCB for single-resource CBwK with dynamic target b_t provided by runner:
      score(a) = UCB(x,a) - lambda * cost[a]
      lambda <- [lambda + eta * (cost[a] - b_t)]_+
    Also uses OTF design update (A updated immediately; b only on reward arrival).
    """

    def __init__(
        self,
        n_arms: int,
        d: int,
        costs: np.ndarray,
        alpha: float = 1.0,
        lam: float = 1.0,
        eta: float = 0.05,
        seed: int = 0,
    ):
        self.K = int(n_arms)
        self.d = int(d)
        self.costs = costs.astype(np.float64)
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.eta = float(eta)
        self.dual = 0.0
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
        score = ucb - self.dual * self.costs
        score = np.where(feasible, score, -np.inf)
        return int(np.argmax(score))

    def update_dual(self, cost: float, b_t: float):
        self.dual = max(0.0, self.dual + self.eta * (float(cost) - float(b_t)))

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