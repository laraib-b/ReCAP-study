from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recap.ard import AdaptiveRegimeDetector
from recap.data import MARKET_FEATURE_COLUMNS
from study.diagnostics import run_synthetic_diagnostic
from study.streaming_ard import (
    calibrate_threshold_scale,
    fit_null_model,
    null_alarm_rate,
    simulate_null,
    with_threshold,
)


def _fake_daily(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {"date": pd.bdate_range("2015-01-01", periods=n)}
    for column in MARKET_FEATURE_COLUMNS:
        data[column] = rng.normal(size=n)
    return pd.DataFrame(data)


def _paper(**kwargs) -> AdaptiveRegimeDetector:
    return AdaptiveRegimeDetector(mode="paper", min_segment_days=20, **kwargs)


def test_repo_mode_unchanged_two_sided() -> None:
    series = np.r_[np.zeros(100) + np.random.default_rng(1).normal(0, 1, 100), np.full(100, -10.0)]
    repo = AdaptiveRegimeDetector(mode="repo", min_segment_days=20)
    assert repo.detect_change_points(series), "repo mode must still alarm on downward shifts"


def test_paper_mode_is_one_sided() -> None:
    rng = np.random.default_rng(2)
    down = np.r_[rng.normal(0, 1, 252), rng.normal(-10, 1, 200)]
    up = np.r_[rng.normal(0, 1, 252), rng.normal(10, 1, 200)]
    det = _paper(initial_reference_days=252)
    assert det.detect_change_points(down) == []
    assert det.detect_change_points(up)[0] >= 252


def test_paper_mode_reestimates_mu0_after_alarm() -> None:
    rng = np.random.default_rng(3)
    series = np.r_[rng.normal(0, 1, 252), rng.normal(8, 1, 400)]
    paper = _paper(initial_reference_days=252, threshold_scale=2.5).detect_change_points(series)
    repo = AdaptiveRegimeDetector(mode="repo", min_segment_days=20, threshold_scale=2.5).detect_change_points(series)
    # One level shift: paper re-anchors mu0 (later alarms are noise crossings);
    # repo keeps firing every min_segment_days against the stale mu0.
    assert paper[0] == 252
    assert len(paper) <= 4
    assert len(repo) > 15


def test_paper_mode_is_causal() -> None:
    series = np.random.default_rng(4).normal(size=600).cumsum()
    det = _paper(initial_reference_days=252)
    full = det.detect_change_points(series)
    for cut in (300, 400, 500):
        assert [p for p in full if p < cut - 20] == [p for p in det.detect_change_points(series[:cut]) if p < cut - 20]


def test_history_streaming_matches_full_series() -> None:
    rng = np.random.default_rng(5)
    dates = pd.bdate_range("2019-01-01", periods=500)
    rows = []
    for tic in ("A", "B"):
        for i, date in enumerate(dates):
            row = {"date": date, "tic": tic}
            for column in MARKET_FEATURE_COLUMNS:
                row[column] = rng.normal() + (3.0 if i > 380 else 0.0)
            rows.append(row)
    data = pd.DataFrame(rows)
    train = data[data["date"] < dates[300]]
    test = data[data["date"] >= dates[300]]
    det = _paper(initial_reference_days=150)
    full_alarms = det.alarm_indices(det.build_daily_features(data), 300)
    segments = det.segment_dates(test, history=train)
    starts = [dates.get_loc(s.start_date) for s in segments[1:]]
    assert starts == full_alarms


def test_null_calibration_hits_target_on_holdout() -> None:
    daily = _fake_daily(400)
    det = _paper(initial_reference_days=200)
    model = fit_null_model(daily, 200)
    cal = simulate_null(model, 200 + 504, 40, seed=1)
    hold = simulate_null(model, 200 + 504, 60, seed=2)
    loose = null_alarm_rate(with_threshold(det, 1.0), hold, 200)[0]
    tight = null_alarm_rate(with_threshold(det, 8.0), hold, 200)[0]
    assert tight <= loose
    scale, meta = calibrate_threshold_scale(det, cal, hold, reference_days=200, target_alarms_per_year=2.0)
    assert scale > 0
    assert meta["holdout_sim_alarms_per_year"] > 0
    assert abs(meta["holdout_sim_alarms_per_year"] - 2.0) < 1.5


def test_synthetic_diagnostic_runs() -> None:
    daily = _fake_daily(400)
    det = _paper(initial_reference_days=200)
    model = fit_null_model(daily, 200)
    payload = run_synthetic_diagnostic(
        det, model, {"default_2.5": 2.5}, sim_days=200 + 504, reference_days=200, n_reps=5
    )
    rows = payload["results"]
    assert {r["shift_sigma"] for r in rows} == {1.0, 2.0, 3.0, -2.0}
    big_up = next(r for r in rows if r["shift_sigma"] == 3.0)
    assert big_up["detection_rate"] > 0.5
