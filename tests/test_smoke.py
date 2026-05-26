from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recap.ard import AdaptiveRegimeDetector
from recap.data import FEATURE_COLUMNS, MARKET_FEATURE_COLUMNS, add_market_features
from recap.env import PortfolioEnv
from recap.metrics import calculate_portfolio_metrics
from recap.recap import ReCAP, ReCAPConfig


def build_synthetic_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=80, freq="B")
    tickers = ["AAA", "BBB", "CCC"]
    rows = []
    for ticker_index, ticker in enumerate(tickers):
        base_price = 100.0 + 5.0 * ticker_index
        for day_index, date in enumerate(dates):
            drift = 0.002 if day_index < 40 else -0.001 + 0.0008 * ticker_index
            close = base_price * (1.0 + drift) ** day_index
            rows.append(
                {
                    "date": date,
                    "tic": ticker,
                    "open": close * 0.99,
                    "high": close * 1.01,
                    "low": close * 0.98,
                    "close": close,
                    "volume": 1_000_000 + 100 * day_index,
                }
            )
    frame = pd.DataFrame(rows)
    return add_market_features(frame, use_vix=False)


def test_metrics_keys() -> None:
    metrics = calculate_portfolio_metrics([1.0, 1.02, 0.99, 1.03])
    assert set(metrics) == {"CR", "SR", "MDD"}


def test_ard_produces_segments() -> None:
    data = build_synthetic_data()
    detector = AdaptiveRegimeDetector(min_segment_days=10, threshold_scale=1.0)
    segments = detector.slice_segments(data)
    assert len(segments) >= 1


def test_portfolio_env_runs() -> None:
    data = build_synthetic_data()
    env = PortfolioEnv(
        data=data,
        tickers=["AAA", "BBB", "CCC"],
        feature_columns=FEATURE_COLUMNS,
        initial_cash=1000.0,
        transaction_cost=1e-3,
    )
    observation, _ = env.reset()
    done = False
    while not done:
        observation, _, done, _, _ = env.step(np.zeros((env.action_dim,), dtype=np.float32))
    assert observation.shape == env.observation_space.shape
    assert len(env.portfolio_values) >= 2


def test_portfolio_env_market_features() -> None:
    data = build_synthetic_data()
    env = PortfolioEnv(
        data=data,
        tickers=["AAA", "BBB", "CCC"],
        feature_columns=FEATURE_COLUMNS,
        initial_cash=1000.0,
        transaction_cost=1e-3,
    )
    market_features = env.get_market_features()
    assert market_features.shape == (len(MARKET_FEATURE_COLUMNS),)


def test_feature_pipeline_exposes_paper_feature_set() -> None:
    data = build_synthetic_data()
    assert all(column in data.columns for column in FEATURE_COLUMNS)


def test_policy_vector_norm_pruning() -> None:
    data = build_synthetic_data()
    train_data = data[data["date"] < pd.Timestamp("2020-03-15")].copy()
    runner = ReCAP(
        config=ReCAPConfig(
            initial_cash=1000.0,
            pretrain_timesteps=8,
            finetune_timesteps=4,
            n_steps=8,
            batch_size=8,
            min_segment_days=8,
            cusum_threshold_scale=1.0,
            min_policy_vector_norm=1e-2,
        ),
        tickers=["AAA", "BBB", "CCC"],
        use_vix=False,
    )
    runner.fit(train_data)
    zero_vector = {
        name: torch.zeros_like(tensor)
        for name, tensor in runner.base_actor_parameters.items()
    }
    assert runner._insert_policy_vector(zero_vector, mean_new_weight=1.0) == -1


def test_recap_smoke_run() -> None:
    data = build_synthetic_data()
    train_data = data[data["date"] < pd.Timestamp("2020-03-15")].copy()
    test_data = data[data["date"] >= pd.Timestamp("2020-03-15")].copy()
    runner = ReCAP(
        config=ReCAPConfig(
            initial_cash=1000.0,
            pretrain_timesteps=16,
            finetune_timesteps=8,
            n_steps=8,
            batch_size=8,
            min_segment_days=8,
            cusum_threshold_scale=1.0,
            gate_epochs=10,
        ),
        tickers=["AAA", "BBB", "CCC"],
        use_vix=False,
    )
    results = runner.run(train_data=train_data, test_data=test_data, online_update=True)
    assert "metrics" in results
    metrics = results["metrics"]
    assert isinstance(metrics, dict)
    assert set(metrics) == {"CR", "SR", "MDD"}
