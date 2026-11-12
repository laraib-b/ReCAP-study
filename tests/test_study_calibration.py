from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.data import MARKET_FEATURE_COLUMNS
from study.diagnostics import run_synthetic_diagnostic
from study.streaming_ard import calibrate_threshold_scale, count_alarms


def _fake_daily(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {"date": pd.date_range("2015-01-01", periods=n, freq="B")}
    for column in MARKET_FEATURE_COLUMNS:
        data[column] = rng.normal(size=n).cumsum() * 0.01
    return pd.DataFrame(data)


def test_count_alarms_and_calibrate() -> None:
    daily = _fake_daily()
    loose = count_alarms(daily, threshold_scale=0.8, min_segment_days=10)
    tight = count_alarms(daily, threshold_scale=5.0, min_segment_days=10)
    assert tight.n_alarms <= loose.n_alarms
    scale, log, meta = calibrate_threshold_scale(daily, target_alarms_per_year=1.0, min_segment_days=10)
    assert scale > 0
    assert log.n_days == len(daily)
    assert "feasible" in meta


def test_synthetic_diagnostic_runs() -> None:
    payload = run_synthetic_diagnostic({"default_2.5": 2.5, "budget_1": 1.5})
    assert payload["mode"] == "synthetic_shift_diagnostic"
    assert len(payload["results"]) >= 2
