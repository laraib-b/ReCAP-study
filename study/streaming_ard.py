"""Instrumented CUSUM ARD helpers for alarm logging and rate calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from recap.ard import AdaptiveRegimeDetector
from recap.data import MARKET_FEATURE_COLUMNS


@dataclass
class AlarmEvent:
    date: str
    index: int
    votes: int
    threshold_scale: float


@dataclass
class DetectorRunLog:
    threshold_scale: float
    n_days: int
    n_alarms: int
    alarms_per_year: float
    alarm_dates: list[str]
    events: list[AlarmEvent]


def count_alarms(
    daily_features: pd.DataFrame,
    *,
    threshold_scale: float,
    min_segment_days: int = 20,
    lookback_days: int = 252,
    vote_threshold: int = 1,
) -> DetectorRunLog:
    detector = AdaptiveRegimeDetector(
        min_segment_days=min_segment_days,
        lookback_days=lookback_days,
        threshold_scale=threshold_scale,
        vote_threshold=vote_threshold,
    )
    point_votes: dict[int, int] = {}
    for column in MARKET_FEATURE_COLUMNS:
        if column not in daily_features.columns:
            continue
        for point in detector.detect_change_points(daily_features[column].to_numpy(dtype=np.float64)):
            point_votes[point] = point_votes.get(point, 0) + 1

    candidates = sorted(p for p, votes in point_votes.items() if votes >= vote_threshold)
    accepted: list[int] = []
    last = -10**9
    for point in candidates:
        if point - last >= min_segment_days:
            accepted.append(point)
            last = point

    events = [
        AlarmEvent(
            date=str(pd.Timestamp(daily_features.iloc[idx]["date"]).date()),
            index=int(idx),
            votes=int(point_votes[idx]),
            threshold_scale=float(threshold_scale),
        )
        for idx in accepted
    ]
    n_days = int(len(daily_features))
    alarms_per_year = float(len(events) / max(n_days / 252.0, 1e-12))
    return DetectorRunLog(
        threshold_scale=float(threshold_scale),
        n_days=n_days,
        n_alarms=len(events),
        alarms_per_year=alarms_per_year,
        alarm_dates=[e.date for e in events],
        events=events,
    )


def calibrate_threshold_scale(
    daily_features: pd.DataFrame,
    *,
    target_alarms_per_year: float,
    min_segment_days: int = 20,
    lookback_days: int = 252,
    vote_threshold: int = 1,
    low: float = 0.5,
    high: float = 8.0,
    max_iters: int = 24,
) -> tuple[float, DetectorRunLog]:
    """Train-only search for threshold_scale matching a target alarm rate."""
    best_scale = high
    best_log = count_alarms(
        daily_features,
        threshold_scale=high,
        min_segment_days=min_segment_days,
        lookback_days=lookback_days,
        vote_threshold=vote_threshold,
    )

    lo, hi = low, high
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        mid_log = count_alarms(
            daily_features,
            threshold_scale=mid,
            min_segment_days=min_segment_days,
            lookback_days=lookback_days,
            vote_threshold=vote_threshold,
        )
        best_scale, best_log = mid, mid_log
        if mid_log.alarms_per_year > target_alarms_per_year:
            lo = mid
        else:
            hi = mid

    final_log = count_alarms(
        daily_features,
        threshold_scale=best_scale,
        min_segment_days=min_segment_days,
        lookback_days=lookback_days,
        vote_threshold=vote_threshold,
    )
    return float(best_scale), final_log
