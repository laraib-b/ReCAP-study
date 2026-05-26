from __future__ import annotations

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .data import MARKET_FEATURE_COLUMNS, resolve_market_feature_columns


class PortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        data: pd.DataFrame,
        tickers: list[str],
        feature_columns: list[str],
        initial_cash: float,
        transaction_cost: float,
    ) -> None:
        super().__init__()
        self.data = data.sort_values(["date", "tic"]).reset_index(drop=True)
        self.tickers = tickers
        self.feature_columns = feature_columns
        self.initial_cash = float(initial_cash)
        self.transaction_cost = float(transaction_cost)

        self.dates = sorted(self.data["date"].unique())
        if len(self.dates) < 2:
            raise ValueError("PortfolioEnv requires at least two trading dates.")

        self.close_matrix = self._build_matrix("close")
        self.feature_matrix = self._build_feature_matrix()
        self.asset_count = len(self.tickers)
        self.action_dim = self.asset_count + 1
        self.market_feature_columns = resolve_market_feature_columns(self.feature_columns, self.data)
        self.market_feature_indices = [
            self.feature_columns.index(column)
            for column in self.market_feature_columns
            if column in self.feature_columns
        ]
        observation_dim = 1 + self.asset_count + self.asset_count * len(self.feature_columns)

        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_dim,), dtype=np.float32)

        self.current_step = 0
        self.previous_weights = np.full(self.action_dim, 1.0 / self.action_dim, dtype=np.float32)
        self.portfolio_value = self.initial_cash
        self.portfolio_values = [self.initial_cash]
        self.portfolio_returns: list[float] = []
        self.actions_memory: list[np.ndarray] = []

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.current_step = 0
        self.previous_weights = np.full(self.action_dim, 1.0 / self.action_dim, dtype=np.float32)
        self.portfolio_value = self.initial_cash
        self.portfolio_values = [self.initial_cash]
        self.portfolio_returns = []
        self.actions_memory = []
        return self._get_observation(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        weights = self._softmax(action)
        self.actions_memory.append(weights.copy())

        current_close = self.close_matrix[self.current_step]
        next_close = self.close_matrix[self.current_step + 1]
        asset_growth = next_close / np.clip(current_close, 1e-12, None)
        full_growth = np.concatenate(([1.0], asset_growth.astype(np.float32)))

        turnover = np.abs(weights - self.previous_weights).sum()
        transaction_penalty = self.transaction_cost * turnover
        portfolio_growth = float(np.dot(weights, full_growth) - transaction_penalty)
        portfolio_growth = max(portfolio_growth, 1e-6)

        self.portfolio_value *= portfolio_growth
        self.portfolio_values.append(self.portfolio_value)
        self.portfolio_returns.append(portfolio_growth - 1.0)
        self.previous_weights = weights
        self.current_step += 1

        terminated = self.current_step >= len(self.dates) - 1
        reward = float(np.log(portfolio_growth))
        observation = self._get_observation() if not terminated else self._terminal_observation()
        info = {
            "date": self.dates[min(self.current_step, len(self.dates) - 1)],
            "portfolio_value": self.portfolio_value,
            "weights": weights,
        }
        return observation, reward, terminated, False, info

    def _get_observation(self) -> np.ndarray:
        features = self.feature_matrix[self.current_step].reshape(-1)
        cash_weight = np.asarray([self.previous_weights[0]], dtype=np.float32)
        asset_weights = self.previous_weights[1:].astype(np.float32)
        observation = np.concatenate([cash_weight, asset_weights, features.astype(np.float32)])
        return observation.astype(np.float32)

    def _terminal_observation(self) -> np.ndarray:
        features = self.feature_matrix[-1].reshape(-1)
        cash_weight = np.asarray([self.previous_weights[0]], dtype=np.float32)
        asset_weights = self.previous_weights[1:].astype(np.float32)
        observation = np.concatenate([cash_weight, asset_weights, features.astype(np.float32)])
        return observation.astype(np.float32)

    def get_market_features(self, step: int | None = None) -> np.ndarray:
        if step is None:
            step = self.current_step
        features = self.feature_matrix[step]
        if not self.market_feature_indices:
            return np.zeros(len(MARKET_FEATURE_COLUMNS), dtype=np.float32)
        market_features = features[:, self.market_feature_indices].mean(axis=0)
        if market_features.shape[0] < len(MARKET_FEATURE_COLUMNS):
            padding = np.zeros(len(MARKET_FEATURE_COLUMNS) - market_features.shape[0], dtype=np.float32)
            market_features = np.concatenate([market_features, padding])
        return market_features.astype(np.float32)

    def get_market_feature_matrix(self) -> np.ndarray:
        if not self.market_feature_indices:
            return np.zeros((len(self.dates), len(MARKET_FEATURE_COLUMNS)), dtype=np.float32)
        market_matrix = self.feature_matrix[:, :, self.market_feature_indices].mean(axis=1)
        if market_matrix.shape[1] < len(MARKET_FEATURE_COLUMNS):
            padding = np.zeros(
                (market_matrix.shape[0], len(MARKET_FEATURE_COLUMNS) - market_matrix.shape[1]),
                dtype=np.float32,
            )
            market_matrix = np.concatenate([market_matrix, padding], axis=1)
        return market_matrix.astype(np.float32)

    def _build_matrix(self, column: str) -> np.ndarray:
        pivot = self.data.pivot(index="date", columns="tic", values=column)
        pivot = pivot.reindex(columns=self.tickers)
        return pivot.to_numpy(dtype=np.float32)

    def _build_feature_matrix(self) -> np.ndarray:
        stacked = []
        for feature in self.feature_columns:
            pivot = self.data.pivot(index="date", columns="tic", values=feature)
            pivot = pivot.reindex(columns=self.tickers).fillna(0.0)
            stacked.append(pivot.to_numpy(dtype=np.float32))
        return np.stack(stacked, axis=-1)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        shifted = values - np.max(values)
        exp_values = np.exp(shifted)
        return exp_values / np.clip(exp_values.sum(), 1e-12, None)
