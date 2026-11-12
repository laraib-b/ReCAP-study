"""Causal test-time evaluation: no max(baseline, adapted) look-ahead selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from recap.metrics import calculate_portfolio_metrics
from recap.recap import ReCAP
from study.streaming_ard import count_alarms


def evaluate_causal(
    runner: ReCAP,
    test_data: pd.DataFrame,
    *,
    online_update: bool = True,
    adapt_on_alarm: bool = True,
) -> dict[str, Any]:
    """Evaluate chronologically with fixed per-segment decision rules.

    Protocol:
    - Segment test data with the fitted detector (CUSUM decisions at t use only
      past observations under the repo's fixed baseline window).
    - For each completed segment, trade with either the adaptive path or the
      frozen library path — never choose after seeing both terminal values.
    - Optionally promote the adapted vector into the library after the segment.
    """
    if runner.base_model is None:
        raise RuntimeError("ReCAP must be fitted before causal evaluation.")

    test_data = test_data[test_data["tic"].isin(runner.tickers)].copy()
    daily = runner.detector.build_daily_features(test_data)
    detector_log = count_alarms(
        daily,
        threshold_scale=runner.config.cusum_threshold_scale,
        min_segment_days=runner.config.min_segment_days,
        vote_threshold=runner.config.ard_vote_threshold,
    )

    segments = runner.detector.slice_segments(test_data)
    if not segments:
        raise ValueError("Test data does not contain any valid segments.")

    portfolio_path: list[float] = [runner.config.initial_cash]
    daily_rows: list[dict[str, object]] = []
    adaptation_events: list[dict[str, object]] = []
    segment_summaries: list[dict[str, object]] = []
    current_cash = runner.config.initial_cash
    turnover_total = 0.0
    cost_total = 0.0

    for segment_index, (segment_data, _) in enumerate(segments):
        close_matrix, feature_matrix, market_matrix = runner._build_segment_tensors(segment_data)
        dates = sorted(pd.to_datetime(segment_data["date"].unique()))

        if adapt_on_alarm and segment_data["date"].nunique() >= 2:
            values, trace, current_vector = runner._simulate_online_adaptive_segment(
                close_matrix=close_matrix,
                feature_matrix=feature_matrix,
                market_matrix=market_matrix,
                initial_cash=current_cash,
            )
            used_current_vector = True
            mean_new_weight = float(trace[:, -1].mean()) if trace.size > 0 else 0.0
            promoted = False
            if online_update:
                promoted = runner._promote_current_vector(
                    current_vector=current_vector,
                    mean_new_weight=mean_new_weight,
                    segment_data=segment_data,
                )
            adaptation_events.append(
                {
                    "segment_index": segment_index,
                    "start_date": str(segment_data["date"].min().date()),
                    "end_date": str(segment_data["date"].max().date()),
                    "adapted": True,
                    "promoted_current_vector": promoted,
                    "mean_new_weight": mean_new_weight,
                    "library_size_after": len(runner.policy_library),
                }
            )
        else:
            values, trace = runner._simulate_dynamic_segment(
                close_matrix=close_matrix,
                feature_matrix=feature_matrix,
                market_matrix=market_matrix,
                initial_cash=current_cash,
                gate_source=runner.gate,
                current_vector=None,
            )
            used_current_vector = False
            mean_new_weight = 0.0
            promoted = False

        # Approximate turnover/cost from consecutive weight traces when available.
        if trace.size > 1:
            diffs = np.abs(np.diff(trace[:, : trace.shape[1]], axis=0)).sum(axis=1)
            segment_turnover = float(diffs.sum())
            turnover_total += segment_turnover
            cost_total += runner.config.transaction_cost * segment_turnover

        for step_idx, value in enumerate(values):
            date = dates[min(step_idx, len(dates) - 1)]
            daily_rows.append(
                {
                    "date": str(pd.Timestamp(date).date()),
                    "portfolio_value": float(value),
                    "segment_index": segment_index,
                    "used_current_vector": used_current_vector,
                }
            )

        portfolio_path.extend(values[1:])
        current_cash = float(values[-1])
        mean_weights = trace.mean(axis=0).tolist() if trace.size > 0 else []
        segment_summaries.append(
            {
                "segment_index": segment_index,
                "start_date": str(segment_data["date"].min().date()),
                "end_date": str(segment_data["date"].max().date()),
                "mean_weights": [float(v) for v in mean_weights],
                "used_current_vector": used_current_vector,
                "mean_new_weight": mean_new_weight,
                "promoted_current_vector": promoted,
            }
        )

    metrics = calculate_portfolio_metrics(
        portfolio_path,
        trading_days_per_year=runner.config.trading_days_per_year,
    )
    n_test_days = max(len(daily_rows) - 1, 1)
    return {
        "protocol": "causal_no_lookahead_selection",
        "adapt_on_alarm": adapt_on_alarm,
        "online_update": online_update,
        "metrics": metrics,
        "library_size": len(runner.policy_library),
        "n_test_segments": len(segments),
        "n_adaptation_events": len(adaptation_events),
        "adaptations_per_year": float(len(adaptation_events) / max(n_test_days / 252.0, 1e-12)),
        "turnover_proxy": turnover_total,
        "transaction_cost_proxy": cost_total,
        "detector": {
            "threshold_scale": detector_log.threshold_scale,
            "n_alarms": detector_log.n_alarms,
            "alarms_per_year": detector_log.alarms_per_year,
            "alarm_dates": detector_log.alarm_dates,
        },
        "adaptation_events": adaptation_events,
        "segment_summaries": segment_summaries,
        "daily": daily_rows,
    }
