"""Tiny end-to-end smoke run using synthetic data (no market download).

Usage (from repo root, with study venv active):
    python study/smoke_run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.data import FEATURE_COLUMNS, add_market_features
from recap.recap import ReCAP, ReCAPConfig


def build_synthetic_data(n_days: int = 80, n_tickers: int = 3) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_days, freq="B")
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for ticker_index, ticker in enumerate(tickers):
        base_price = 100.0 + 5.0 * ticker_index
        for day_index, date in enumerate(dates):
            drift = 0.002 if day_index < n_days // 2 else -0.001 + 0.0008 * ticker_index
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


def main() -> None:
    data = build_synthetic_data()
    missing = [c for c in FEATURE_COLUMNS if c not in data.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing}")

    split = pd.Timestamp("2020-03-15")
    train_data = data[data["date"] < split].copy()
    test_data = data[data["date"] >= split].copy()
    tickers = sorted(data["tic"].unique().tolist())

    runner = ReCAP(
        config=ReCAPConfig(
            initial_cash=1_000.0,
            pretrain_timesteps=32,
            finetune_timesteps=16,
            n_steps=16,
            batch_size=16,
            min_segment_days=8,
            cusum_threshold_scale=1.0,
            gate_epochs=5,
            seed=0,
            device="cpu",
        ),
        tickers=tickers,
        use_vix=False,
    )
    results = runner.run(train_data=train_data, test_data=test_data, online_update=False)

    out_dir = ROOT / "outputs" / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": "synthetic_smoke",
        "tickers": tickers,
        "library_size": results["library_size"],
        "train_segments": results["train_segments"],
        "metrics": results["metrics"],
        "n_test_segments": len(results.get("segment_summaries", [])),
    }
    out_path = out_dir / "smoke_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(json.dumps(payload, indent=2, default=float))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
