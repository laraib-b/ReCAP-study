from __future__ import annotations

import math

import numpy as np


def calculate_portfolio_metrics(portfolio_values: list[float] | np.ndarray, trading_days_per_year: int = 252) -> dict[str, float]:
    values = np.asarray(portfolio_values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("portfolio_values must contain at least two points")

    returns = values[1:] / values[:-1] - 1.0
    cumulative_return = float(values[-1] / values[0] - 1.0)

    mean_return = float(np.mean(returns))
    std_return = float(np.std(returns))
    sharpe_ratio = 0.0
    if std_return > 1e-12:
        sharpe_ratio = math.sqrt(trading_days_per_year) * mean_return / std_return

    running_max = np.maximum.accumulate(values)
    drawdown = (running_max - values) / np.clip(running_max, 1e-12, None)
    max_drawdown = float(np.max(drawdown))

    return {
        "CR": cumulative_return,
        "SR": float(sharpe_ratio),
        "MDD": max_drawdown,
    }
