from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


class _BaseDisjointLogisticUCB:
    """
    Disjoint logistic UCB with delayed feedback.

    Key difference from the linear OTF heuristic:
      - update_design(a, x) is a no-op
      - uncertainty only shrinks when the binary label is observed
      - y=0 updates the model through the logistic gradient/Hessian

    The optimistic score is computed on the logit scale:
      score_ucb(a, x) = sigmoid(logit_a(x) + alpha * sqrt(x^T H_a^{-1} x))
    """

    def __init__(
        self,
        n_arms: int,
        d: int,
        alpha: float = 1.0,
        lam: float = 1.0,
        step_scale: float = 1.0,
        min_curvature: float = 1e-3,
        seed: int = 0,
    ):
        self.K = int(n_arms)
        self.d = int(d)
        self.p = self.d + 1
        self.alpha = float(alpha)
        self.lam = float(lam)
        self.step_scale = float(step_scale)
        self.min_curvature = float(min_curvature)
        self.rng = np.random.default_rng(seed)

        I = np.eye(self.p, dtype=np.float64)
        self.H_inv = np.stack([(1.0 / self.lam) * I for _ in range(self.K)], axis=0)
        self.beta = np.zeros((self.K, self.p), dtype=np.float64)

    def _augment(self, x: np.ndarray) -> np.ndarray:
        x_aug = np.empty((self.p,), dtype=np.float64)
        x_aug[:-1] = x.astype(np.float64, copy=False)
        x_aug[-1] = 1.0
        return x_aug

    def _ucb_prob(self, x: np.ndarray) -> np.ndarray:
        x_aug = self._augment(x)
        logits = self.beta @ x_aug
        v = self.H_inv @ x_aug
        sigma = np.sqrt(np.maximum(np.sum(v * x_aug[None, :], axis=1), 0.0))
        return np.asarray(sigmoid(logits + self.alpha * sigma), dtype=np.float64)

    def select(self, x: np.ndarray) -> int:
        feasible = np.ones(self.K, dtype=bool)
        return self.select_feasible(x, feasible)

    def update_design(self, a: int, x: np.ndarray):
        del a, x
        return None

    def update_reward(self, a: int, x: np.ndarray, r: float):
        a = int(a)
        y = float(r)
        x_aug = self._augment(x)

        beta = self.beta[a]
        H_inv = self.H_inv[a]

        logit = float(beta @ x_aug)
        p = float(sigmoid(logit))
        curvature = float(np.clip(p * (1.0 - p), self.min_curvature, 0.25))

        # Hessian approximation update for the observed label only.
        v = H_inv @ x_aug
        denom = 1.0 + curvature * float(x_aug @ v)
        H_inv = H_inv - (curvature * np.outer(v, v)) / max(denom, 1e-12)
        self.H_inv[a] = H_inv

        # One online Newton step on the penalized logistic objective.
        grad = (y - p) * x_aug - self.lam * beta
        grad[-1] += self.lam * beta[-1]
        self.beta[a] = beta + self.step_scale * (H_inv @ grad)


class DisjointLogisticUCB(_BaseDisjointLogisticUCB):
    def select_feasible(self, x: np.ndarray, feasible: np.ndarray) -> int:
        score = self._ucb_prob(x)
        score = np.where(feasible, score, -np.inf)
        return int(np.argmax(score))


class PrimalDualLogisticUCB(_BaseDisjointLogisticUCB):
    def __init__(
        self,
        n_arms: int,
        d: int,
        costs: np.ndarray,
        alpha: float = 1.0,
        lam: float = 1.0,
        eta: float = 0.05,
        step_scale: float = 1.0,
        min_curvature: float = 1e-3,
        seed: int = 0,
    ):
        super().__init__(
            n_arms=n_arms,
            d=d,
            alpha=alpha,
            lam=lam,
            step_scale=step_scale,
            min_curvature=min_curvature,
            seed=seed,
        )
        self.costs = costs.astype(np.float64)
        self.eta = float(eta)
        self.dual = 0.0

    def select_feasible(self, x: np.ndarray, feasible: np.ndarray) -> int:
        score = self._ucb_prob(x) - self.dual * self.costs
        score = np.where(feasible, score, -np.inf)
        return int(np.argmax(score))

    def update_dual(self, cost: float, b_t: float):
        self.dual = max(0.0, self.dual + self.eta * (float(cost) - float(b_t)))


class CostNormalizedDisjointLogisticUCB(_BaseDisjointLogisticUCB):
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
        step_scale: float = 1.0,
        min_curvature: float = 1e-3,
        seed: int = 0,
    ):
        super().__init__(
            n_arms=n_arms,
            d=d,
            alpha=alpha,
            lam=lam,
            step_scale=step_scale,
            min_curvature=min_curvature,
            seed=seed,
        )
        self.costs = costs.astype(np.float64)
        self.mode = str(mode)
        self.gamma = float(gamma)
        self.eps = float(eps)

    def select_feasible(self, x: np.ndarray, feasible: np.ndarray) -> int:
        ucb = self._ucb_prob(x)
        if self.mode == "ratio":
            score = ucb / (self.costs + self.eps)
        elif self.mode == "sub":
            score = ucb - self.gamma * self.costs
        else:
            raise ValueError("mode must be 'ratio' or 'sub'")

        score = np.where(feasible, score, -np.inf)
        return int(np.argmax(score))
