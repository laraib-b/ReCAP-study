"""Alarm logging and no-change-simulation calibration of the ARD threshold h.

Only ``threshold_scale`` (h in units of the reference-window sigma) is varied.
Features, kappa = 0.5 sigma, the reference-mean update rule and reset behaviour
are whatever ``AdaptiveRegimeDetector`` implements for the configured mode.
"""

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
    threshold_scale: float


@dataclass
class DetectorRunLog:
    threshold_scale: float
    n_days: int
    n_alarms: int
    alarms_per_year: float
    alarm_dates: list[str]
    events: list[AlarmEvent]


def with_threshold(detector: AdaptiveRegimeDetector, threshold_scale: float) -> AdaptiveRegimeDetector:
    return AdaptiveRegimeDetector(
        min_segment_days=detector.min_segment_days,
        lookback_days=detector.lookback_days,
        threshold_scale=float(threshold_scale),
        vote_threshold=detector.vote_threshold,
        mode=detector.mode,
        initial_reference_days=detector.initial_reference_days,
    )


def count_alarms(
    detector: AdaptiveRegimeDetector,
    data: pd.DataFrame,
    *,
    history: pd.DataFrame | None = None,
) -> DetectorRunLog:
    """Alarms (accepted segment boundaries) the detector raises inside ``data``.

    These are historical alarms, not labelled true/false: market data has no
    known change dates.
    """
    if history is not None and detector.mode == "paper":
        history = history[history["date"] < data["date"].min()]
        combined = detector.build_daily_features(pd.concat([history, data], ignore_index=True))
        start = int((combined["date"] < data["date"].min()).sum())
    else:
        combined = detector.build_daily_features(data)
        start = 0
    indices = detector.alarm_indices(combined, start)
    n_days = int(len(combined) - start)
    events = [
        AlarmEvent(
            date=str(pd.Timestamp(combined.iloc[idx]["date"]).date()),
            index=int(idx - start),
            threshold_scale=float(detector.threshold_scale),
        )
        for idx in indices
    ]
    return DetectorRunLog(
        threshold_scale=float(detector.threshold_scale),
        n_days=n_days,
        n_alarms=len(events),
        alarms_per_year=float(len(events) / max(n_days / 252.0, 1e-12)),
        alarm_dates=[e.date for e in events],
        events=events,
    )


@dataclass
class NullModel:
    """VAR(1) no-change model fitted on the detector's initial reference window."""

    intercept: np.ndarray
    coef: np.ndarray
    chol: np.ndarray
    start: np.ndarray
    columns: list[str]
    reference_days: int
    spectral_radius_raw: float
    spectral_radius_used: float

    def describe(self) -> dict[str, object]:
        return {
            "model": "VAR(1) with correlated Gaussian innovations",
            "fit_window": f"first {self.reference_days} train days (detector initial reference window)",
            "columns": self.columns,
            "spectral_radius_raw": self.spectral_radius_raw,
            "spectral_radius_used": self.spectral_radius_used,
        }


def fit_null_model(daily_train: pd.DataFrame, reference_days: int, max_radius: float = 0.995) -> NullModel:
    columns = [c for c in MARKET_FEATURE_COLUMNS if c in daily_train.columns and daily_train[c].std() > 1e-12]
    x = daily_train[columns].to_numpy(dtype=np.float64)[:reference_days]
    lagged = np.hstack([np.ones((len(x) - 1, 1)), x[:-1]])
    beta, *_ = np.linalg.lstsq(lagged, x[1:], rcond=None)
    intercept, coef = beta[0], beta[1:].T
    resid = x[1:] - lagged @ beta
    cov = np.cov(resid, rowvar=False) + 1e-12 * np.eye(len(columns))

    radius_raw = float(np.max(np.abs(np.linalg.eigvals(coef))))
    radius = radius_raw
    if radius_raw > max_radius:
        # Shrink towards the reference mean so the no-change process is stationary.
        scale = max_radius / radius_raw
        mean = x.mean(axis=0)
        coef = coef * scale
        intercept = (np.eye(len(columns)) - coef) @ mean
        radius = max_radius
    return NullModel(
        intercept=intercept,
        coef=coef,
        chol=np.linalg.cholesky(cov),
        start=x[-1].copy(),
        columns=columns,
        reference_days=int(reference_days),
        spectral_radius_raw=radius_raw,
        spectral_radius_used=float(radius),
    )


def simulate_null(model: NullModel, n_days: int, n_reps: int, seed: int, burn_in: int = 500) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dim = len(model.columns)
    total = burn_in + n_days
    dates = pd.bdate_range("2000-01-03", periods=n_days)
    frames: list[pd.DataFrame] = []
    for _ in range(n_reps):
        shocks = rng.standard_normal((total, dim)) @ model.chol.T
        path = np.empty((total, dim))
        state = model.start
        for t in range(total):
            state = model.intercept + model.coef @ state + shocks[t]
            path[t] = state
        frame = pd.DataFrame(path[burn_in:], columns=model.columns)
        for column in MARKET_FEATURE_COLUMNS:
            if column not in frame.columns:
                frame[column] = 0.0
        frame.insert(0, "date", dates)
        frames.append(frame)
    return frames


def null_alarm_rate(
    detector: AdaptiveRegimeDetector, sims: list[pd.DataFrame], reference_days: int
) -> tuple[float, int, float]:
    """Alarms/year after the reference window, pooled over no-change simulations."""
    sim_detector = AdaptiveRegimeDetector(
        min_segment_days=detector.min_segment_days,
        lookback_days=detector.lookback_days,
        threshold_scale=detector.threshold_scale,
        vote_threshold=detector.vote_threshold,
        mode=detector.mode,
        initial_reference_days=reference_days,
    )
    alarms = 0
    years = 0.0
    for frame in sims:
        alarms += len(sim_detector.alarm_indices(frame, reference_days))
        years += (len(frame) - reference_days) / 252.0
    return alarms / max(years, 1e-12), alarms, years


def poisson_ci(count: int, years: float) -> tuple[float, float]:
    """Approximate 95% CI for a Poisson rate (Wilson-Hilferty)."""
    if years <= 0:
        return (float("nan"), float("nan"))
    lower = 0.0 if count == 0 else count * (1 - 1 / (9 * count) - 1.96 / (3 * np.sqrt(count))) ** 3
    upper_n = count + 1
    upper = upper_n * (1 - 1 / (9 * upper_n) + 1.96 / (3 * np.sqrt(upper_n))) ** 3
    return (float(lower / years), float(upper / years))


def calibrate_threshold_scale(
    detector: AdaptiveRegimeDetector,
    calibration_sims: list[pd.DataFrame],
    holdout_sims: list[pd.DataFrame],
    *,
    reference_days: int,
    target_alarms_per_year: float,
    low: float = 0.25,
    high: float = 64.0,
    max_iters: int = 18,
    relative_tol: float = 0.25,
) -> tuple[float, dict[str, object]]:
    """Choose h on calibration sims; report the achieved rate on held-out sims."""

    def rate(scale: float, sims: list[pd.DataFrame]) -> tuple[float, int, float]:
        return null_alarm_rate(with_threshold(detector, scale), sims, reference_days)

    high_rate = rate(high, calibration_sims)[0]
    if high_rate > target_alarms_per_year:
        scale = high
    else:
        lo, hi = np.log(low), np.log(high)
        for _ in range(max_iters):
            mid = 0.5 * (lo + hi)
            if rate(float(np.exp(mid)), calibration_sims)[0] > target_alarms_per_year:
                lo = mid
            else:
                hi = mid
        scale = float(np.exp(hi))

    cal_rate, _, _ = rate(scale, calibration_sims)
    hold_rate, hold_alarms, hold_years = rate(scale, holdout_sims)
    feasible = abs(hold_rate - target_alarms_per_year) <= relative_tol * target_alarms_per_year
    return scale, {
        "target_alarms_per_year": float(target_alarms_per_year),
        "threshold_scale": float(scale),
        "calibration_sim_alarms_per_year": float(cal_rate),
        "holdout_sim_alarms_per_year": float(hold_rate),
        "holdout_sim_alarms": int(hold_alarms),
        "holdout_sim_years": float(hold_years),
        "holdout_sim_rate_ci95": poisson_ci(hold_alarms, hold_years),
        "within_tolerance": bool(feasible),
        "tolerance": relative_tol,
        "hit_search_ceiling": bool(scale >= high),
    }


def default_null_rate(
    detector: AdaptiveRegimeDetector, holdout_sims: list[pd.DataFrame], reference_days: int
) -> dict[str, object]:
    hold_rate, hold_alarms, hold_years = null_alarm_rate(detector, holdout_sims, reference_days)
    return {
        "threshold_scale": float(detector.threshold_scale),
        "holdout_sim_alarms_per_year": float(hold_rate),
        "holdout_sim_alarms": int(hold_alarms),
        "holdout_sim_years": float(hold_years),
        "holdout_sim_rate_ci95": poisson_ci(hold_alarms, hold_years),
    }


__all__ = [
    "AlarmEvent",
    "DetectorRunLog",
    "NullModel",
    "calibrate_threshold_scale",
    "count_alarms",
    "default_null_rate",
    "fit_null_model",
    "null_alarm_rate",
    "simulate_null",
    "with_threshold",
]
