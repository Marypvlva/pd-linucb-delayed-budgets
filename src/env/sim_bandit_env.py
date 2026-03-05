# src/env/sim_bandit_env.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


@dataclass
class SimBanditEnv:
    """
    Contextual bandit simulator with:
      - arm-dependent costs (prefer real costs_by_arm.npy)
      - delayed feedback with censoring window W=censor_steps

    Delay semantics returned by step():
      - always returns dly >= 0  (NEVER -1)
      - if r=1: dly sampled from delay_pool (may include 0)
      - if r=0: dly = censor_steps (censoring; "no conversion" confirmed after W steps)
        If censor_steps==0 -> immediate feedback for r=0 (no-delay mode).
    """
    X_data: np.ndarray          # memmap/ndarray (n,d)
    D_data: np.ndarray          # memmap/ndarray (n,) may contain -1 on disk
    costs: np.ndarray           # (K,) arm-dependent costs (float32)
    rng: np.random.Generator
    delay_pool: np.ndarray      # (m,) delays in steps, >=0 (int32)
    censor_steps: int           # W in steps, >=0

    # reward model (linear + clipping; consistent with LinUCB)
    Theta: np.ndarray | None = None     # (K,d) if available
    w: np.ndarray | None = None         # (d,) fallback global linear
    b: np.ndarray | None = None         # (K,) fallback per-arm bias

    @property
    def n(self) -> int:
        return int(self.X_data.shape[0])

    @property
    def d(self) -> int:
        return int(self.X_data.shape[1])

    @property
    def K(self) -> int:
        if self.Theta is not None:
            return int(self.Theta.shape[0])
        if self.b is not None:
            return int(self.b.shape[0])
        raise RuntimeError("No reward model parameters loaded (Theta or b).")

    # ------------------------ helpers ------------------------

    @staticmethod
    def _make_costs(K: int, cost_mode: str, seed: int) -> np.ndarray:
        """Fallback ONLY if real costs file is missing."""
        if cost_mode == "lin":
            if K == 1:
                return np.array([1.0], dtype=np.float32)
            return (0.5 + 1.5 * (np.arange(K) / (K - 1))).astype(np.float32)
        if cost_mode == "rand":
            rng0 = np.random.default_rng(seed)
            return rng0.uniform(0.5, 2.0, size=K).astype(np.float32)
        raise ValueError("cost_mode must be 'lin' or 'rand'")

    @staticmethod
    def _build_delay_pool(
        D: np.ndarray,
        rng: np.random.Generator,
        pool_size: int = 200_000,
        max_tries: int = 2_000_000,
    ) -> np.ndarray:
        """Fallback: sample positive delays from D (D>=0)."""
        n = int(D.shape[0])
        pool = np.empty((0,), dtype=np.int32)
        tries = 0
        while pool.size < pool_size and tries < max_tries:
            batch = min(200_000, max_tries - tries)
            idx = rng.integers(0, n, size=batch)
            d = D[idx]
            d = d[d >= 0]
            if d.size:
                pool = np.concatenate([pool, d.astype(np.int32)], axis=0)
            tries += batch
        if pool.size == 0:
            return np.empty((0,), dtype=np.int32)
        return pool[:pool_size] if pool.size > pool_size else pool

    @staticmethod
    def _compute_theta_from_arm_stats(
        meta_stats: np.lib.npyio.NpzFile, K: int, d: int, ridge_lambda: float
    ) -> np.ndarray | None:
        """
        Optional: if meta_and_stats.npz has XtX_arm / XtR_arm, build per-arm ridge Theta:
          Theta[a] = (XtX_arm[a] + lam I)^{-1} XtR_arm[a]
        """
        if ("XtX_arm" not in meta_stats.files) or ("XtR_arm" not in meta_stats.files):
            return None
        XtX_arm = meta_stats["XtX_arm"].astype(np.float64)  # (K,d,d)
        XtR_arm = meta_stats["XtR_arm"].astype(np.float64)  # (K,d)
        if XtX_arm.shape != (K, d, d):
            raise ValueError(f"XtX_arm shape mismatch: expected {(K, d, d)}, got {XtX_arm.shape}")
        if XtR_arm.shape != (K, d):
            raise ValueError(f"XtR_arm shape mismatch: expected {(K, d)}, got {XtR_arm.shape}")

        Theta = np.zeros((K, d), dtype=np.float64)
        I = np.eye(d, dtype=np.float64)
        for a in range(K):
            Theta[a] = np.linalg.solve(XtX_arm[a] + ridge_lambda * I, XtR_arm[a])
        return Theta.astype(np.float32)

    # ------------------------ loading ------------------------

    @classmethod
    def from_memmap_dir(
        cls,
        memmap_dir: str,
        seed: int = 123,
        ridge_lambda: float = 1.0,
        cost_mode: str = "lin",
        delay_pool_size: int = 200_000,
        normalize_costs: bool = True,
    ) -> "SimBanditEnv":
        p = Path(memmap_dir)

        meta_stats = np.load(p / "meta_and_stats.npz", allow_pickle=True)
        meta = meta_stats["meta"].item()
        n = int(meta["n"])
        d = int(meta["d"])
        K = int(meta["k_cap"])

        # ---- censor window W (in steps) ----
        censor_steps = int(meta.get("censor_steps", meta.get("max_delay", meta.get("d_max", 0))))
        censor_steps = max(0, censor_steps)

        X = np.load(p / "X.npy", mmap_mode="r")
        D = np.load(p / "D.npy", mmap_mode="r")
        if X.shape != (n, d):
            raise ValueError(f"X shape mismatch: expected {(n, d)}, got {X.shape}")
        if D.shape[0] != n:
            raise ValueError(f"D length mismatch: expected {n}, got {D.shape[0]}")

        rng = np.random.default_rng(seed)

        # ---- delays: prefer precomputed positive delays ----
        delays_path = p / "delays_pos.npy"
        if delays_path.exists():
            delay_pool = np.load(delays_path).astype(np.int32).reshape(-1)
            delay_pool = delay_pool[delay_pool >= 0]
        else:
            delay_pool = cls._build_delay_pool(D, rng, pool_size=delay_pool_size)

        # keep delays within censor window if censor_steps>0
        if censor_steps > 0 and delay_pool.size > 0:
            delay_pool = delay_pool[delay_pool <= censor_steps]

        if delay_pool.size > delay_pool_size:
            idx = rng.choice(delay_pool.size, size=delay_pool_size, replace=False)
            delay_pool = delay_pool[idx]
        delay_pool = delay_pool.astype(np.int32)

        # ---- costs: prefer real per-arm costs ----
        costs_path = p / "costs_by_arm.npy"
        if costs_path.exists():
            raw_costs = np.load(costs_path).astype(np.float64).reshape(-1)
            if raw_costs.shape[0] != K:
                raise ValueError(f"costs_by_arm.npy shape mismatch: expected {(K,)}, got {raw_costs.shape}")

            costs = raw_costs.copy()

            # normalize to mean~1 to keep budget_ratio=B/T interpretable
            if normalize_costs:
                # IMPORTANT: memmap_full pipeline saves cnt_arm (not cnt_a).
                cnt = None
                if "cnt_arm" in meta_stats.files:
                    cnt = meta_stats["cnt_arm"].astype(np.float64).reshape(-1)
                elif "cnt_a" in meta_stats.files:
                    cnt = meta_stats["cnt_a"].astype(np.float64).reshape(-1)

                if cnt is not None and cnt.shape[0] == K and float(np.sum(cnt)) > 0:
                    mean_cost = float(np.sum(costs * cnt) / np.sum(cnt))
                else:
                    mean_cost = float(np.mean(costs))

                mean_cost = max(mean_cost, 1e-12)
                costs = costs / mean_cost

            costs = costs.astype(np.float32)
        else:
            costs = cls._make_costs(K, cost_mode=cost_mode, seed=seed)

        # ---- reward model: prefer arm-specific Theta ----
        arm_path = p / "arm_ridge_stats.npz"
        if arm_path.exists():
            arm = np.load(arm_path, allow_pickle=True)
            if "Theta" not in arm.files:
                raise ValueError("arm_ridge_stats.npz exists but has no 'Theta'")
            Theta = arm["Theta"].astype(np.float32)
            if Theta.shape != (K, d):
                raise ValueError(f"Theta shape mismatch: expected {(K, d)}, got {Theta.shape}")
            return cls(
                X_data=X, D_data=D, costs=costs, rng=rng,
                delay_pool=delay_pool, censor_steps=censor_steps,
                Theta=Theta
            )

        Theta = cls._compute_theta_from_arm_stats(meta_stats, K=K, d=d, ridge_lambda=ridge_lambda)
        if Theta is not None:
            return cls(
                X_data=X, D_data=D, costs=costs, rng=rng,
                delay_pool=delay_pool, censor_steps=censor_steps,
                Theta=Theta
            )

        # ---- fallback: global linear w + per-arm bias b ----
        # (not used in the memmap_full Criteo pipeline; kept for compatibility with other artifacts)
        if ("XtX" not in meta_stats.files) or ("XtR" not in meta_stats.files):
            raise ValueError(
                "No arm-specific reward model found (Theta / XtX_arm+XtR_arm), "
                "and no global (XtX, XtR) stats present. "
                f"Available keys: {list(meta_stats.files)}"
            )

        XtX = meta_stats["XtX"].astype(np.float64)
        XtR = meta_stats["XtR"].astype(np.float64)
        w = np.linalg.solve(XtX + ridge_lambda * np.eye(d), XtR).astype(np.float32)

        # per-arm bias: try common key variants
        sum_r = None
        cnt = None
        sum_x = None

        if ("sum_r_arm" in meta_stats.files) and ("cnt_arm" in meta_stats.files) and ("sum_x_arm" in meta_stats.files):
            sum_r = meta_stats["sum_r_arm"].astype(np.float64)
            cnt = meta_stats["cnt_arm"].astype(np.float64)
            sum_x = meta_stats["sum_x_arm"].astype(np.float64)
        elif ("sum_r" in meta_stats.files) and ("cnt_a" in meta_stats.files) and ("sum_x" in meta_stats.files):
            sum_r = meta_stats["sum_r"].astype(np.float64)
            cnt = meta_stats["cnt_a"].astype(np.float64)
            sum_x = meta_stats["sum_x"].astype(np.float64)

        if sum_r is None or cnt is None or sum_x is None:
            # if bias stats not available, set to zeros (still provides a valid model)
            b = np.zeros((K,), dtype=np.float32)
        else:
            mean_r = sum_r / np.maximum(cnt, 1.0)
            mean_x = sum_x / np.maximum(cnt[:, None], 1.0)
            b = (mean_r - (mean_x @ w.astype(np.float64))).astype(np.float32)

        return cls(
            X_data=X, D_data=D, costs=costs, rng=rng,
            delay_pool=delay_pool, censor_steps=censor_steps,
            w=w, b=b
        )

    # ------------------------ interaction ------------------------

    def sample_contexts(self, T: int) -> np.ndarray:
        idx = self.rng.integers(0, self.n, size=T)
        return np.asarray(self.X_data[idx], dtype=np.float32)

    def step(self, x: np.ndarray, a: int) -> tuple[float, float, int]:
        """
        Returns: (reward r in {0,1}, cost c, delay dly in {0,1,2,...})
        Delay semantics:
          - r=1: dly ~ delay_pool (or 0 if pool empty)
          - r=0: dly = censor_steps (or 0 if censor_steps==0)
        """
        a = int(a)
        x64 = x.astype(np.float64)

        if self.Theta is not None:
            mu = float(np.clip(self.Theta[a].astype(np.float64) @ x64, 0.0, 1.0))
        else:
            mu = float(np.clip(self.w.astype(np.float64) @ x64 + float(self.b[a]), 0.0, 1.0))

        r = float(self.rng.random() < mu)
        c = float(self.costs[a])

        if r > 0.0:
            if self.delay_pool.size > 0:
                dly = int(self.rng.choice(self.delay_pool))
                if dly < 0:
                    dly = 0
                if self.censor_steps > 0:
                    dly = min(dly, self.censor_steps)
            else:
                dly = 0
        else:
            dly = int(self.censor_steps) if self.censor_steps > 0 else 0

        return r, c, dly