"""Injected-shift detector diagnostic on no-change simulations (not market performance)."""

from __future__ import annotations

import numpy as np

from recap.ard import AdaptiveRegimeDetector
from study.streaming_ard import NullModel, simulate_null, with_threshold


def run_synthetic_diagnostic(
    detector: AdaptiveRegimeDetector,
    null_model: NullModel,
    threshold_scales: dict[str, float],
    *,
    sim_days: int,
    reference_days: int,
    magnitudes: tuple[float, ...] = (1.0, 2.0, 3.0, -2.0),
    shift_offset_days: int = 252,
    detection_window: int = 60,
    n_reps: int = 50,
    seed: int = 30_000,
) -> dict[str, object]:
    """Add a step of ``magnitude`` reference-window sigmas to every ARD feature.

    Change dates are known here, so alarms before the shift are genuine false
    alarms of the simulation. Negative magnitudes probe the one-sided detector.
    """
    shift_day = reference_days + shift_offset_days
    if shift_day + detection_window >= sim_days:
        raise ValueError("Simulation too short for the requested shift and detection window.")

    base_sims = simulate_null(null_model, sim_days, n_reps, seed=seed)
    rows: list[dict[str, object]] = []
    for magnitude in magnitudes:
        shifted = []
        for frame in base_sims:
            frame = frame.copy()
            for column in null_model.columns:
                sigma = float(frame[column].iloc[:reference_days].std(ddof=0))
                frame.loc[shift_day:, column] = frame.loc[shift_day:, column] + magnitude * sigma
            shifted.append(frame)

        for label, scale in threshold_scales.items():
            sim_detector = with_threshold(detector, scale)
            sim_detector.initial_reference_days = reference_days
            delays: list[int] = []
            pre_alarms = 0
            for frame in shifted:
                alarms = sim_detector.alarm_indices(frame, reference_days)
                pre_alarms += sum(1 for a in alarms if a < shift_day)
                hits = [a for a in alarms if shift_day <= a <= shift_day + detection_window]
                if hits:
                    delays.append(int(hits[0] - shift_day))
            pre_years = n_reps * (shift_day - reference_days) / 252.0
            rows.append(
                {
                    "label": label,
                    "threshold_scale": float(scale),
                    "shift_sigma": float(magnitude),
                    "detection_rate": len(delays) / n_reps,
                    "median_delay_days": float(np.median(delays)) if delays else None,
                    "pre_shift_false_alarms_per_year": pre_alarms / pre_years,
                }
            )
    return {
        "mode": "synthetic_shift_diagnostic",
        "warning": "Injected-shift metrics are detector diagnostics only, not market performance.",
        "base_process": "held-out no-change VAR(1) simulations",
        "shift_day": shift_day,
        "detection_window": detection_window,
        "n_reps": n_reps,
        "seed": seed,
        "results": rows,
    }
