from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MARKET_FEATURE_COLUMNS, resolve_market_feature_columns

ARD_MODES = ("repo", "paper")


@dataclass(frozen=True)
class RegimeSegment:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    features: np.ndarray


class AdaptiveRegimeDetector:
    """ARD: 使用简化版多特征 CUSUM 进行市场分段。

    mode="repo"  — original release: two-sided CUSUM, mu0/sigma from the first
                   min(lookback, n // 2) days of the given series, mu0 never
                   re-estimated after an alarm.
    mode="paper" — paper Eq.: one-sided S_t = max(0, S_{t-1} + (u_t - mu0) - kappa),
                   mu0/sigma from an initial historical reference window, and after
                   each change point S resets and mu0 is re-estimated on the
                   subsequent reference window (length = min_segment_days) before
                   detection resumes. kappa = 0.5 sigma and h = threshold_scale * sigma
                   in both modes.
    """

    def __init__(
        self,
        min_segment_days: int = 60,
        lookback_days: int = 252,
        threshold_scale: float = 2.5,
        vote_threshold: int = 1,
        mode: str = "repo",
        initial_reference_days: int | None = None,
    ) -> None:
        if mode not in ARD_MODES:
            raise ValueError(f"Unknown ARD mode {mode!r}; expected one of {ARD_MODES}.")
        self.min_segment_days = min_segment_days
        self.lookback_days = lookback_days
        self.threshold_scale = threshold_scale
        self.vote_threshold = vote_threshold
        self.mode = mode
        self.initial_reference_days = initial_reference_days

    def build_daily_features(self, data: pd.DataFrame) -> pd.DataFrame:
        feature_columns = resolve_market_feature_columns(data.columns.tolist(), data)
        grouped = data.groupby("date")[feature_columns].mean()
        for column in MARKET_FEATURE_COLUMNS:
            if column not in grouped.columns:
                grouped[column] = 0.0
        grouped = grouped.ffill().bfill().fillna(0.0)
        return grouped.reset_index()[["date", *MARKET_FEATURE_COLUMNS]]

    def reference_days_for(self, n_days: int) -> int:
        if self.mode == "paper" and self.initial_reference_days is not None:
            return int(self.initial_reference_days)
        return min(self.lookback_days, max(n_days // 2, 1))

    def detect_change_points(self, series: np.ndarray) -> list[int]:
        values = np.asarray(series, dtype=np.float64)
        if values.size < 3:
            return []
        if self.mode == "paper":
            return self._detect_paper(values)
        return self._detect_repo(values)

    def _detect_repo(self, values: np.ndarray) -> list[int]:
        lookback = self.reference_days_for(values.size)
        baseline = values[:lookback]
        mu = float(np.mean(baseline))
        sigma = float(np.std(baseline))
        if sigma < 1e-12:
            return []

        kappa = 0.5 * sigma
        threshold = self.threshold_scale * sigma
        s_plus = np.zeros_like(values)
        s_minus = np.zeros_like(values)
        change_points: list[int] = []

        for index in range(1, values.size):
            s_plus[index] = max(0.0, s_plus[index - 1] + values[index] - mu - kappa)
            s_minus[index] = max(0.0, s_minus[index - 1] - values[index] + mu - kappa)
            if s_plus[index] > threshold or s_minus[index] > threshold:
                if not change_points or index - change_points[-1] >= self.min_segment_days:
                    change_points.append(index)
                s_plus[index] = 0.0
                s_minus[index] = 0.0

        return change_points

    def _detect_paper(self, values: np.ndarray) -> list[int]:
        reference = self.reference_days_for(values.size)
        if reference >= values.size:
            return []
        baseline = values[:reference]
        mu = float(np.mean(baseline))
        sigma = float(np.std(baseline))
        if sigma < 1e-12:
            return []

        kappa = 0.5 * sigma
        threshold = self.threshold_scale * sigma
        window = max(int(self.min_segment_days), 1)
        stream = values.tolist()
        change_points: list[int] = []
        statistic = 0.0
        index = reference
        n = len(stream)
        while index < n:
            statistic = max(0.0, statistic + stream[index] - mu - kappa)
            if statistic > threshold:
                change_points.append(index)
                ref_end = min(index + 1 + window, n)
                if ref_end - (index + 1) < 1:
                    break
                mu = float(np.mean(values[index + 1 : ref_end]))
                statistic = 0.0
                index = ref_end
                continue
            index += 1
        return change_points

    def alarm_indices(self, daily: pd.DataFrame, start_index: int = 0) -> list[int]:
        """Union/vote/min-segment filtered change points at or after start_index."""
        point_votes: dict[int, int] = {}
        for column in MARKET_FEATURE_COLUMNS:
            if column not in daily.columns:
                continue
            for point in self.detect_change_points(daily[column].to_numpy(dtype=np.float64)):
                point_votes[point] = point_votes.get(point, 0) + 1

        candidate_points = sorted(point for point, votes in point_votes.items() if votes >= self.vote_threshold)
        accepted: list[int] = []
        last = start_index
        for point in candidate_points:
            if point < start_index:
                continue
            if point - last >= self.min_segment_days:
                accepted.append(point)
                last = point
        return accepted

    def segment_dates(self, data: pd.DataFrame, history: pd.DataFrame | None = None) -> list[RegimeSegment]:
        """Segment ``data``. In paper mode, ``history`` (strictly earlier dates) is
        streamed through the detector first so the reference window is historical."""
        feature_columns = MARKET_FEATURE_COLUMNS
        if history is not None and self.mode == "paper":
            history = history[history["date"] < data["date"].min()]
            combined = self.build_daily_features(pd.concat([history, data], ignore_index=True))
            start_index = int((combined["date"] < data["date"].min()).sum())
        else:
            combined = self.build_daily_features(data)
            start_index = 0

        daily = combined.iloc[start_index:].reset_index(drop=True)
        boundaries = [0, *[p - start_index for p in self.alarm_indices(combined, start_index)]]
        if len(daily) - boundaries[-1] >= 2:
            boundaries.append(len(daily))

        if len(boundaries) == 2 and boundaries[1] - boundaries[0] < self.min_segment_days:
            features = daily[feature_columns].mean().to_numpy(dtype=np.float32)
            return [RegimeSegment(start_date=daily.iloc[0]["date"], end_date=daily.iloc[-1]["date"], features=features)]

        segments: list[RegimeSegment] = []
        for start, end in zip(boundaries[:-1], boundaries[1:], strict=True):
            segment_daily = daily.iloc[start:end]
            if len(segment_daily) < 2:
                continue
            features = segment_daily[feature_columns].mean().to_numpy(dtype=np.float32)
            segments.append(
                RegimeSegment(
                    start_date=segment_daily.iloc[0]["date"],
                    end_date=segment_daily.iloc[-1]["date"],
                    features=features,
                )
            )

        if not segments:
            features = daily[feature_columns].mean().to_numpy(dtype=np.float32)
            segments.append(RegimeSegment(start_date=daily.iloc[0]["date"], end_date=daily.iloc[-1]["date"], features=features))
        return segments

    def slice_segments(
        self, data: pd.DataFrame, history: pd.DataFrame | None = None
    ) -> list[tuple[pd.DataFrame, np.ndarray]]:
        segments = []
        for regime in self.segment_dates(data, history=history):
            mask = (data["date"] >= regime.start_date) & (data["date"] <= regime.end_date)
            segment = data.loc[mask].copy()
            if segment["date"].nunique() >= 2:
                segments.append((segment, regime.features))
        return segments
