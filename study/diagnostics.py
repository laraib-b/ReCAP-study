"""Synthetic-shift detector diagnostic (separate from market portfolio results)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from recap.data import MARKET_FEATURE_COLUMNS
from study.streaming_ard import count_alarms


@dataclass
class ShiftCaseResult:
    shift_day: int
    magnitude: float
    threshold_scale: float
    detected: bool
    detection_delay: int | None
    missed: bool
    n_alarms: int
    false_alarms_before_shift: int


def make_feature_series(
    n_days: int = 400,
    shift_day: int = 250,
    magnitude: float = 1.5,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="B")
    data = {"date": dates}
    for column in MARKET_FEATURE_COLUMNS:
        noise = rng.normal(0.0, 1.0, size=n_days)
        series = noise.copy()
        series[shift_day:] += magnitude
        data[column] = series
    return pd.DataFrame(data)


def score_detection(
    daily_features: pd.DataFrame,
    *,
    shift_day: int,
    threshold_scale: float,
    detection_window: int = 30,
    min_segment_days: int = 20,
) -> ShiftCaseResult:
    log = count_alarms(
        daily_features,
        threshold_scale=threshold_scale,
        min_segment_days=min_segment_days,
    )
    post = [e.index for e in log.events if shift_day <= e.index <= shift_day + detection_window]
    pre = [e.index for e in log.events if e.index < shift_day]
    detected = len(post) > 0
    delay = int(post[0] - shift_day) if detected else None
    return ShiftCaseResult(
        shift_day=shift_day,
        magnitude=float(daily_features.iloc[shift_day][MARKET_FEATURE_COLUMNS[0]] - daily_features.iloc[shift_day - 1][MARKET_FEATURE_COLUMNS[0]]),
        threshold_scale=threshold_scale,
        detected=detected,
        detection_delay=delay,
        missed=not detected,
        n_alarms=log.n_alarms,
        false_alarms_before_shift=len(pre),
    )


def run_synthetic_diagnostic(
    threshold_scales: dict[str, float],
    *,
    magnitudes: tuple[float, ...] = (0.75, 1.5, 2.5),
    detection_window: int = 30,
    seed: int = 0,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    for mag_i, magnitude in enumerate(magnitudes):
        features = make_feature_series(magnitude=magnitude, seed=seed + mag_i)
        for label, scale in threshold_scales.items():
            result = score_detection(
                features,
                shift_day=250,
                threshold_scale=scale,
                detection_window=detection_window,
            )
            payload = asdict(result)
            payload["label"] = label
            payload["injected_magnitude"] = magnitude
            rows.append(payload)
    return {
        "mode": "synthetic_shift_diagnostic",
        "warning": "Injected-shift metrics are detector diagnostics only, not market performance.",
        "detection_window": detection_window,
        "results": rows,
    }
