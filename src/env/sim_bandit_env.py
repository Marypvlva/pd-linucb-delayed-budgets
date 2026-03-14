# src/env/sim_bandit_env.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    z = np.clip(z, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class SimBanditEnv:
    """
    Semi-synthetic contextual bandit environment built from logged data artifacts.

    Main behavior:
      - Contexts are sampled from X.npy, optionally restricted to a split (train/test).
      - Rewards are generated from either:
          * logistic arm-specific model: sigmoid(theta_a^T x + bias_a)
          * legacy clipped-linear model: clip(theta_a^T x, 0, 1)
          * fallback global linear model: clip(w^T x + b_a, 0, 1)
      - Costs used online are arm-level averages c(a) computed on the fit split.
      - Positive delays can be global or arm-conditional, depending on available artifacts.
    """

    X_data: np.ndarray
    D_data: np.ndarray
    costs: np.ndarray
    rng: np.random.Generator
    delay_pool: np.ndarray
    censor_steps: int

    Theta: np.ndarray | None = None
    intercept: np.ndarray | None = None
    w: np.ndarray | None = None
    b: np.ndarray | None = None

    meta: dict | None = None
    context_index: np.ndarray | None = None
    context_split: str = "all"
    reward_model: str = "linear_clip"
    delay_values_by_arm: np.ndarray | None = None
    delay_offsets_by_arm: np.ndarray | None = None

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

    def clone(self, seed: int | None = None) -> "SimBanditEnv":
        if seed is None:
            seed = int(self.rng.integers(0, 2**32 - 1))
        rng = np.random.default_rng(int(seed))
        return SimBanditEnv(
            X_data=self.X_data,
            D_data=self.D_data,
            costs=self.costs,
            rng=rng,
            delay_pool=self.delay_pool,
            censor_steps=int(self.censor_steps),
            Theta=self.Theta,
            intercept=self.intercept,
            w=self.w,
            b=self.b,
            meta=self.meta,
            context_index=self.context_index,
            context_split=self.context_split,
            reward_model=self.reward_model,
            delay_values_by_arm=self.delay_values_by_arm,
            delay_offsets_by_arm=self.delay_offsets_by_arm,
        )

    def with_delays(
        self,
        *,
        delay_pool: np.ndarray | None = None,
        censor_steps: int | None = None,
        seed: int | None = None,
    ) -> "SimBanditEnv":
        env = self.clone(seed=seed)
        if delay_pool is not None:
            env.delay_pool = np.asarray(delay_pool, dtype=np.int32).reshape(-1)
            env.delay_values_by_arm = None
            env.delay_offsets_by_arm = None
        if censor_steps is not None:
            env.censor_steps = max(0, int(censor_steps))
        return env

    def make_no_delay(self, seed: int | None = None) -> "SimBanditEnv":
        return self.with_delays(delay_pool=np.array([0], dtype=np.int32), censor_steps=0, seed=seed)

    @staticmethod
    def _make_costs(K: int, cost_mode: str, seed: int) -> np.ndarray:
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
        if ("XtX_arm" not in meta_stats.files) or ("XtR_arm" not in meta_stats.files):
            return None
        XtX_arm = meta_stats["XtX_arm"].astype(np.float64)
        XtR_arm = meta_stats["XtR_arm"].astype(np.float64)
        if XtX_arm.shape != (K, d, d):
            raise ValueError(f"XtX_arm shape mismatch: expected {(K, d, d)}, got {XtX_arm.shape}")
        if XtR_arm.shape != (K, d):
            raise ValueError(f"XtR_arm shape mismatch: expected {(K, d)}, got {XtR_arm.shape}")

        Theta = np.zeros((K, d), dtype=np.float64)
        I = np.eye(d, dtype=np.float64)
        for a in range(K):
            Theta[a] = np.linalg.solve(XtX_arm[a] + ridge_lambda * I, XtR_arm[a])
        return Theta.astype(np.float32)

    @staticmethod
    def _load_arm_delay_pools(path: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not path.exists():
            return None, None
        data = np.load(path, allow_pickle=False)
        if ("values" not in data.files) or ("offsets" not in data.files):
            raise ValueError(f"{path} missing required keys 'values' and 'offsets'")
        return data["values"].astype(np.int32), data["offsets"].astype(np.int64)

    @classmethod
    def from_memmap_dir(
        cls,
        memmap_dir: str,
        seed: int = 123,
        ridge_lambda: float = 1.0,
        cost_mode: str = "lin",
        delay_pool_size: int = 200_000,
        normalize_costs: bool = True,
        context_split: str = "auto",
        reward_model: str = "auto",
    ) -> "SimBanditEnv":
        if context_split not in {"auto", "all", "train", "test"}:
            raise ValueError("context_split must be one of: auto, all, train, test")
        if reward_model not in {"auto", "linear_clip", "logistic"}:
            raise ValueError("reward_model must be one of: auto, linear_clip, logistic")

        p = Path(memmap_dir)
        meta_stats = np.load(p / "meta_and_stats.npz", allow_pickle=True)
        meta = meta_stats["meta"].item()
        n = int(meta["n"])
        d = int(meta["d"])
        K = int(meta["k_cap"])
        has_split = bool(meta.get("has_split", False)) and (p / "split.npy").exists()

        if context_split == "auto":
            effective_context_split = "test" if has_split and int(meta.get("n_test", 0)) > 0 else "all"
        else:
            effective_context_split = context_split

        context_index = None
        if effective_context_split != "all":
            split_path = p / "split.npy"
            if not split_path.exists():
                raise ValueError(f"context_split={effective_context_split} requested, but {split_path} does not exist")
            split = np.load(split_path, mmap_mode="r")
            split_code = 0 if effective_context_split == "train" else 1
            context_index = np.flatnonzero(split == split_code)
            if context_index.size == 0:
                raise ValueError(f"context_split={effective_context_split} selected zero rows in {split_path}")
            if n <= np.iinfo(np.uint32).max:
                context_index = context_index.astype(np.uint32, copy=False)

        censor_steps = int(meta.get("censor_steps", meta.get("max_delay", meta.get("d_max", 0))))
        censor_steps = max(0, censor_steps)

        X = np.load(p / "X.npy", mmap_mode="r")
        D = np.load(p / "D.npy", mmap_mode="r")
        if X.shape != (n, d):
            raise ValueError(f"X shape mismatch: expected {(n, d)}, got {X.shape}")
        if D.shape[0] != n:
            raise ValueError(f"D length mismatch: expected {n}, got {D.shape[0]}")

        rng = np.random.default_rng(seed)

        delays_path = p / "delays_pos.npy"
        if delays_path.exists():
            delay_pool = np.load(delays_path).astype(np.int32).reshape(-1)
            delay_pool = delay_pool[delay_pool >= 0]
        else:
            delay_pool = cls._build_delay_pool(D, rng, pool_size=delay_pool_size)

        if censor_steps > 0 and delay_pool.size > 0:
            delay_pool = delay_pool[delay_pool <= censor_steps]
        if delay_pool.size > delay_pool_size:
            idx = rng.choice(delay_pool.size, size=delay_pool_size, replace=False)
            delay_pool = delay_pool[idx]
        delay_pool = delay_pool.astype(np.int32)

        delay_values_by_arm, delay_offsets_by_arm = cls._load_arm_delay_pools(p / "delays_pos_by_arm.npz")

        costs_path = p / "costs_by_arm.npy"
        if costs_path.exists():
            raw_costs = np.load(costs_path).astype(np.float64).reshape(-1)
            if raw_costs.shape[0] != K:
                raise ValueError(f"costs_by_arm.npy shape mismatch: expected {(K,)}, got {raw_costs.shape}")
            costs = raw_costs.copy()

            if normalize_costs:
                cnt = None
                if "cnt_arm" in meta_stats.files:
                    cnt = meta_stats["cnt_arm"].astype(np.float64).reshape(-1)
                elif "cnt_a" in meta_stats.files:
                    cnt = meta_stats["cnt_a"].astype(np.float64).reshape(-1)

                if cnt is not None and cnt.shape[0] == K and float(np.sum(cnt)) > 0:
                    mean_cost = float(np.sum(costs * cnt) / np.sum(cnt))
                else:
                    mean_cost = float(np.mean(costs))
                costs = costs / max(mean_cost, 1e-12)

            costs = costs.astype(np.float32)
        else:
            costs = cls._make_costs(K, cost_mode=cost_mode, seed=seed)

        meta_loaded = dict(meta)
        meta_loaded["context_split"] = effective_context_split

        logistic_path = p / "arm_logistic_stats.npz"
        if reward_model in {"auto", "logistic"} and logistic_path.exists():
            logi = np.load(logistic_path, allow_pickle=True)
            if "Theta" not in logi.files or "bias" not in logi.files:
                raise ValueError("arm_logistic_stats.npz exists but is missing Theta/bias")
            logi_row_split = "all"
            if "row_split" in logi.files:
                raw_split = logi["row_split"]
                logi_row_split = str(raw_split.item() if getattr(raw_split, "shape", ()) == () else raw_split)
            use_logi = not (
                has_split and str(meta.get("fit_split", "all")) == "train" and logi_row_split not in {"train", "fit"}
            )
            if use_logi:
                Theta = logi["Theta"].astype(np.float32)
                intercept = logi["bias"].astype(np.float32).reshape(-1)
                if Theta.shape != (K, d):
                    raise ValueError(f"Logistic Theta shape mismatch: expected {(K, d)}, got {Theta.shape}")
                if intercept.shape[0] != K:
                    raise ValueError(f"Logistic bias shape mismatch: expected {(K,)}, got {intercept.shape}")
                meta_loaded["reward_model_loaded"] = "logistic"
                return cls(
                    X_data=X,
                    D_data=D,
                    costs=costs,
                    rng=rng,
                    delay_pool=delay_pool,
                    censor_steps=censor_steps,
                    Theta=Theta,
                    intercept=intercept,
                    meta=meta_loaded,
                    context_index=context_index,
                    context_split=effective_context_split,
                    reward_model="logistic",
                    delay_values_by_arm=delay_values_by_arm,
                    delay_offsets_by_arm=delay_offsets_by_arm,
                )
        elif reward_model == "logistic":
            raise ValueError(f"Requested reward_model=logistic, but {logistic_path} does not exist")

        arm_path = p / "arm_ridge_stats.npz"
        if arm_path.exists():
            arm = np.load(arm_path, allow_pickle=True)
            arm_row_split = "all"
            if "row_split" in arm.files:
                raw_split = arm["row_split"]
                arm_row_split = str(raw_split.item() if getattr(raw_split, "shape", ()) == () else raw_split)
            use_arm_file = not (
                has_split and str(meta.get("fit_split", "all")) == "train" and arm_row_split not in {"train", "fit"}
            )
            if use_arm_file:
                if "Theta" not in arm.files:
                    raise ValueError("arm_ridge_stats.npz exists but has no 'Theta'")
                Theta = arm["Theta"].astype(np.float32)
                if Theta.shape != (K, d):
                    raise ValueError(f"Theta shape mismatch: expected {(K, d)}, got {Theta.shape}")
                meta_loaded["reward_model_loaded"] = "linear_clip"
                return cls(
                    X_data=X,
                    D_data=D,
                    costs=costs,
                    rng=rng,
                    delay_pool=delay_pool,
                    censor_steps=censor_steps,
                    Theta=Theta,
                    meta=meta_loaded,
                    context_index=context_index,
                    context_split=effective_context_split,
                    reward_model="linear_clip",
                    delay_values_by_arm=delay_values_by_arm,
                    delay_offsets_by_arm=delay_offsets_by_arm,
                )

        Theta = cls._compute_theta_from_arm_stats(meta_stats, K=K, d=d, ridge_lambda=ridge_lambda)
        if Theta is not None:
            meta_loaded["reward_model_loaded"] = "linear_clip"
            return cls(
                X_data=X,
                D_data=D,
                costs=costs,
                rng=rng,
                delay_pool=delay_pool,
                censor_steps=censor_steps,
                Theta=Theta,
                meta=meta_loaded,
                context_index=context_index,
                context_split=effective_context_split,
                reward_model="linear_clip",
                delay_values_by_arm=delay_values_by_arm,
                delay_offsets_by_arm=delay_offsets_by_arm,
            )

        if ("XtX" not in meta_stats.files) or ("XtR" not in meta_stats.files):
            raise ValueError(
                "No arm-specific reward model found (logistic / Theta / XtX_arm+XtR_arm), "
                "and no global (XtX, XtR) stats present. "
                f"Available keys: {list(meta_stats.files)}"
            )

        XtX = meta_stats["XtX"].astype(np.float64)
        XtR = meta_stats["XtR"].astype(np.float64)
        w = np.linalg.solve(XtX + ridge_lambda * np.eye(d), XtR).astype(np.float32)

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
            b = np.zeros((K,), dtype=np.float32)
        else:
            mean_r = sum_r / np.maximum(cnt, 1.0)
            mean_x = sum_x / np.maximum(cnt[:, None], 1.0)
            b = (mean_r - (mean_x @ w.astype(np.float64))).astype(np.float32)

        meta_loaded["reward_model_loaded"] = "global_linear_clip"
        return cls(
            X_data=X,
            D_data=D,
            costs=costs,
            rng=rng,
            delay_pool=delay_pool,
            censor_steps=censor_steps,
            w=w,
            b=b,
            meta=meta_loaded,
            context_index=context_index,
            context_split=effective_context_split,
            reward_model="global_linear_clip",
            delay_values_by_arm=delay_values_by_arm,
            delay_offsets_by_arm=delay_offsets_by_arm,
        )

    def sample_contexts(self, T: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = self.rng if rng is None else rng
        if self.context_index is None:
            idx = rng.integers(0, self.n, size=int(T))
        else:
            pool_idx = rng.integers(0, int(self.context_index.shape[0]), size=int(T))
            idx = self.context_index[pool_idx]
        return np.asarray(self.X_data[idx], dtype=np.float32)

    def predict_prob(self, x: np.ndarray, a: int) -> float:
        a = int(a)
        x64 = x.astype(np.float64, copy=False)

        if self.Theta is not None:
            if self.reward_model == "logistic":
                bias = 0.0 if self.intercept is None else float(self.intercept[a])
                return float(sigmoid(float(self.Theta[a].astype(np.float64) @ x64) + bias))
            return float(np.clip(self.Theta[a].astype(np.float64) @ x64, 0.0, 1.0))

        return float(np.clip(self.w.astype(np.float64) @ x64 + float(self.b[a]), 0.0, 1.0))

    def predict_prob_batch(self, X: np.ndarray, arms: np.ndarray) -> np.ndarray:
        X64 = np.asarray(X, dtype=np.float64)
        arms = np.asarray(arms, dtype=np.int64)

        if self.Theta is not None:
            theta = self.Theta[arms].astype(np.float64)
            scores = np.sum(theta * X64, axis=1)
            if self.reward_model == "logistic":
                if self.intercept is not None:
                    scores = scores + self.intercept[arms].astype(np.float64)
                return np.asarray(sigmoid(scores), dtype=np.float64)
            return np.clip(scores, 0.0, 1.0)

        scores = X64 @ self.w.astype(np.float64) + self.b[arms].astype(np.float64)
        return np.clip(scores, 0.0, 1.0)

    def _sample_positive_delay(self, a: int) -> int:
        if self.delay_values_by_arm is not None and self.delay_offsets_by_arm is not None:
            lo = int(self.delay_offsets_by_arm[a])
            hi = int(self.delay_offsets_by_arm[a + 1])
            if hi > lo:
                j = int(self.rng.integers(lo, hi))
                dly = int(self.delay_values_by_arm[j])
                if self.censor_steps > 0:
                    dly = min(dly, self.censor_steps)
                return max(dly, 0)

        if self.delay_pool.size > 0:
            dly = int(self.rng.choice(self.delay_pool))
            if self.censor_steps > 0:
                dly = min(dly, self.censor_steps)
            return max(dly, 0)
        return 0

    def step(self, x: np.ndarray, a: int) -> tuple[float, float, int]:
        a = int(a)
        mu = self.predict_prob(x, a)
        r = float(self.rng.random() < mu)
        c = float(self.costs[a])

        if r > 0.0:
            dly = self._sample_positive_delay(a)
        else:
            dly = int(self.censor_steps) if self.censor_steps > 0 else 0

        return r, c, dly
