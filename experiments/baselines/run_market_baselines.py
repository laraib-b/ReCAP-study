"""Classical market baselines matched to the reduced DOW30 study window.

Baselines:
  - buy_and_hold: equal initial weights, never rebalanced
  - equal_weight_daily: daily equal-weight rebalancing (high-turnover reference)
  - equal_weight_monthly: equal-weight rebalanced at month ends (fixed calendar)

Uses the same 28-ticker universe and chronological test window as the reduced
pilot (2020-05-01 → 2025-04-29). Transaction cost applied as a simple turnover
penalty consistent with ReCAPConfig.transaction_cost = 1e-3.

These are non-RL reference strategies. They are not substitutes for a matched
fixed-window PPO retrain (see checklist if that experiment remains incomplete).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_IMPL = Path(__file__).resolve().parents[2]
if str(_IMPL) not in sys.path:
    sys.path.insert(0, str(_IMPL))

from paths import PROJECT_ROOT
from recap.metrics import calculate_portfolio_metrics
from study.market_tickers import DOW30_TICKERS

TRANSACTION_COST = 1e-3
INITIAL_CASH = 1_000_000.0
START = "2008-05-01"
TRAIN_END = "2020-05-01"
TEST_END = "2025-04-29"
ROOT = PROJECT_ROOT
OUT_DIR = ROOT / "results" / "tables_phase3"
FIG_SUPPORT = _IMPL / "experiments" / "baselines"


def download_adjusted_close(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    # yfinance end is exclusive for daily bars in recent versions; pad one day.
    end_inclusive = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    data = yf.download(
        tickers=tickers,
        start=start,
        end=end_inclusive,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )
    closes = {}
    if isinstance(data.columns, pd.MultiIndex):
        for tic in tickers:
            if tic not in data.columns.get_level_values(0):
                continue
            series = data[tic]["Close"] if "Close" in data[tic].columns else data[tic].iloc[:, 0]
            closes[tic] = series
    else:
        # Single-ticker fallback
        closes[tickers[0]] = data["Close"]
    prices = pd.DataFrame(closes).sort_index()
    prices = prices.loc[:, prices.notna().any(axis=0)]
    # Require continuous coverage over the full panel used by the study.
    prices = prices.dropna(axis=0, how="any")
    return prices


def portfolio_path_from_weights(
    prices: pd.DataFrame,
    weight_schedule: pd.DataFrame,
    transaction_cost: float = TRANSACTION_COST,
    initial_cash: float = INITIAL_CASH,
) -> tuple[np.ndarray, float]:
    """Simulate long-only fully invested portfolios with turnover costs."""
    common = prices.index.intersection(weight_schedule.index)
    prices = prices.loc[common]
    weights = weight_schedule.loc[common].to_numpy(dtype=np.float64)
    px = prices.to_numpy(dtype=np.float64)
    n_days, n_assets = px.shape
    values = np.empty(n_days, dtype=np.float64)
    holdings = weights[0] * initial_cash / px[0]
    values[0] = float(np.sum(holdings * px[0]))
    turnover_total = 0.0
    for t in range(1, n_days):
        pre_rebalance = float(np.sum(holdings * px[t]))
        target = weights[t] * pre_rebalance / px[t]
        turnover = float(np.abs(target - holdings).sum())
        cost = transaction_cost * turnover
        turnover_total += turnover
        # Pay cost by scaling holdings proportionally after rebalance.
        if pre_rebalance > 0:
            scale = max((pre_rebalance - cost) / pre_rebalance, 0.0)
        else:
            scale = 0.0
        holdings = target * scale
        values[t] = float(np.sum(holdings * px[t]))
    return values, turnover_total


def make_weight_schedules(prices: pd.DataFrame) -> dict[str, pd.DataFrame]:
    n = prices.shape[1]
    equal = np.full(n, 1.0 / n)
    idx = prices.index

    bh = pd.DataFrame(np.tile(equal, (len(idx), 1)), index=idx, columns=prices.columns)
    # Buy-and-hold: rebalance only on day 0 (first row). Keep target as initial equal;
    # simulation holds shares constant after day 0 by using previous holdings when
    # weights are marked "hold". We encode BH by setting weights equal only on day 0
    # and then propagating market weights (no forced rebalance) via a special path.
    daily = pd.DataFrame(np.tile(equal, (len(idx), 1)), index=idx, columns=prices.columns)

    month_ends = idx.to_series().groupby([idx.year, idx.month]).tail(1).index
    monthly = bh.copy()
    monthly.loc[:] = np.nan
    monthly.iloc[0] = equal
    monthly.loc[month_ends] = equal
    monthly = monthly.ffill()

    return {
        "equal_weight_daily": daily,
        "equal_weight_monthly": monthly,
    }


def buy_and_hold_path(prices: pd.DataFrame, initial_cash: float = INITIAL_CASH) -> tuple[np.ndarray, float]:
    n = prices.shape[1]
    equal = np.full(n, 1.0 / n)
    px0 = prices.iloc[0].to_numpy(dtype=np.float64)
    holdings = equal * initial_cash / px0
    # One-time formation cost approximated as full initial allocation turnover.
    turnover = float(np.abs(holdings).sum())  # in share units; better: weight L1 = 1
    # Use weight-space turnover of 1.0 at formation.
    cost = TRANSACTION_COST * 1.0
    holdings = holdings * max((initial_cash - cost * initial_cash) / initial_cash, 0.0)
    values = (prices.to_numpy(dtype=np.float64) * holdings).sum(axis=1)
    return values.astype(np.float64), 1.0


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_SUPPORT.mkdir(parents=True, exist_ok=True)
    tickers = list(DOW30_TICKERS)
    print(json.dumps({"phase": "download", "n_tickers": len(tickers), "start": START, "end": TEST_END}))
    prices_all = download_adjusted_close(tickers, START, TEST_END)
    used = list(prices_all.columns)
    missing = sorted(set(tickers) - set(used))
    test_prices = prices_all.loc[TRAIN_END:TEST_END]
    # Align to exclusive train_end: first test day on/after train_end.
    test_prices = test_prices[test_prices.index >= pd.Timestamp(TRAIN_END)]
    if test_prices.empty:
        raise RuntimeError("Empty test price panel after date filter.")

    schedules = make_weight_schedules(test_prices)
    results = {}

    bh_values, bh_turn = buy_and_hold_path(test_prices)
    results["buy_and_hold"] = {
        **calculate_portfolio_metrics(bh_values),
        "turnover_proxy": bh_turn,
        "n_days": int(len(bh_values)),
        "rebalance": "none_after_formation",
    }

    for name, weights in schedules.items():
        values, turn = portfolio_path_from_weights(test_prices, weights)
        results[name] = {
            **calculate_portfolio_metrics(values),
            "turnover_proxy": turn,
            "n_days": int(len(values)),
            "rebalance": name,
        }

    payload = {
        "created_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "dataset": "DOW30_study_28",
        "tickers_requested": tickers,
        "tickers_used": used,
        "tickers_missing": missing,
        "dates": {"start": START, "train_end": TRAIN_END, "test_end": TEST_END},
        "transaction_cost": TRANSACTION_COST,
        "initial_cash": INITIAL_CASH,
        "note": (
            "Matched to reduced study chronological test window and 28-ticker universe. "
            "Not an RL baseline; provides classical return/risk references."
        ),
        "results": results,
        "equity_curves": {
            "buy_and_hold": {
                "dates": [d.strftime("%Y-%m-%d") for d in test_prices.index],
                "values": bh_values.tolist(),
            },
            "equal_weight_monthly": {
                "dates": [d.strftime("%Y-%m-%d") for d in test_prices.index],
                "values": portfolio_path_from_weights(test_prices, schedules["equal_weight_monthly"])[0].tolist(),
            },
        },
    }
    out_json = OUT_DIR / "market_baselines.json"
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = [{"variant": k, **{m: v[m] for m in ("CR", "SR", "MDD", "turnover_proxy", "n_days")}} for k, v in results.items()]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "market_baselines.csv", index=False)
    print(df.to_string(index=False))
    print(f"saved {out_json}")


if __name__ == "__main__":
    main()
