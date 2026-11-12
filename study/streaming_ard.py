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
    max_high: float = 128.0,
    max_iters: int = 28,
    relative_tol: float = 0.25,
) -> tuple[float, DetectorRunLog, dict[str, object]]:
    """Train-only search for threshold_scale matching a target alarm rate.

    Expands the upper bound until the achieved rate is at/below target or the
    alarm count stops decreasing. Returns metadata describing feasibility.
    """

    def _eval(scale: float) -> DetectorRunLog:
        return count_alarms(
            daily_features,
            threshold_scale=scale,
            min_segment_days=min_segment_days,
            lookback_days=lookback_days,
            vote_threshold=vote_threshold,
        )

    search_high = float(high)
    hi_log = _eval(search_high)
    while hi_log.alarms_per_year > target_alarms_per_year and search_high < max_high:
        prev_alarms = hi_log.n_alarms
        next_high = min(search_high * 2.0, max_high)
        next_log = _eval(next_high)
        search_high, hi_log = next_high, next_log
        # Alarm floor from min_segment_days / feature-union; further h won't help.
        if next_log.n_alarms >= prev_alarms:
            break

    lo = float(low)
    hi = float(search_high)
    best_scale = hi
    best_log = hi_log
    for _ in range(max_iters):
        mid = 0.5 * (lo + hi)
        mid_log = _eval(mid)
        best_scale, best_log = mid, mid_log
        if mid_log.alarms_per_year > target_alarms_per_year:
            lo = mid
        else:
            hi = mid

    final_log = _eval(best_scale)
    rel_err = abs(final_log.alarms_per_year - target_alarms_per_year) / max(target_alarms_per_year, 1e-12)
    feasible = final_log.alarms_per_year <= target_alarms_per_year * (1.0 + relative_tol)
    meta = {
        "target_alarms_per_year": target_alarms_per_year,
        "threshold_scale": float(best_scale),
        "achieved_alarms_per_year": final_log.alarms_per_year,
        "relative_error": float(rel_err),
        "feasible": bool(feasible),
        "search_high_used": float(search_high),
        "notes": (
            None
            if feasible
            else (
                "Could not reach target alarm rate; min_segment_days and multi-feature "
                "union likely impose an alarm floor on this window."
            )
        ),
    }
    return float(best_scale), final_log, meta
