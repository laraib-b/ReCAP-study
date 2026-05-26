from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MARKET_FEATURE_COLUMNS, resolve_market_feature_columns


@dataclass(frozen=True)
class RegimeSegment:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    features: np.ndarray


class AdaptiveRegimeDetector:
    """ARD: 使用简化版多特征 CUSUM 进行市场分段。"""

    def __init__(
        self,
        min_segment_days: int = 60,
        lookback_days: int = 252,
        threshold_scale: float = 2.5,
        vote_threshold: int = 1,
    ) -> None:
        self.min_segment_days = min_segment_days
        self.lookback_days = lookback_days
        self.threshold_scale = threshold_scale
        self.vote_threshold = vote_threshold

    def build_daily_features(self, data: pd.DataFrame) -> pd.DataFrame:
        feature_columns = resolve_market_feature_columns(data.columns.tolist(), data)
        grouped = data.groupby("date")[feature_columns].mean()
        for column in MARKET_FEATURE_COLUMNS:
            if column not in grouped.columns:
                grouped[column] = 0.0
        grouped = grouped.ffill().bfill().fillna(0.0)
        return grouped.reset_index()[["date", *MARKET_FEATURE_COLUMNS]]

    def detect_change_points(self, series: np.ndarray) -> list[int]:
        values = np.asarray(series, dtype=np.float64)
        if values.size < 3:
            return []

        lookback = min(self.lookback_days, max(values.size // 2, 1))
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

    def segment_dates(self, data: pd.DataFrame) -> list[RegimeSegment]:
        daily = self.build_daily_features(data)
        feature_columns = MARKET_FEATURE_COLUMNS
        point_votes: dict[int, int] = {}

        for column in feature_columns:
            points = self.detect_change_points(daily[column].to_numpy())
            for point in points:
                point_votes[point] = point_votes.get(point, 0) + 1

        candidate_points = sorted(point for point, votes in point_votes.items() if votes >= self.vote_threshold)
        boundaries = [0]
        for point in candidate_points:
            if point - boundaries[-1] >= self.min_segment_days:
                boundaries.append(point)
        if len(daily) - boundaries[-1] >= 2:
            boundaries.append(len(daily))

        if len(boundaries) == 2 and boundaries[1] - boundaries[0] < self.min_segment_days:
            features = daily[feature_columns].mean().to_numpy(dtype=np.float32)
            return [RegimeSegment(start_date=daily.iloc[0]["date"], end_date=daily.iloc[-1]["date"], features=features)]

        segments: list[RegimeSegment] = []
        for start_index, end_index in zip(boundaries[:-1], boundaries[1:], strict=True):
            segment_daily = daily.iloc[start_index:end_index]
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

    def slice_segments(self, data: pd.DataFrame) -> list[tuple[pd.DataFrame, np.ndarray]]:
        segments = []
        for regime in self.segment_dates(data):
            mask = (data["date"] >= regime.start_date) & (data["date"] <= regime.end_date)
            segment = data.loc[mask].copy()
            if segment["date"].nunique() >= 2:
                segments.append((segment, regime.features))
        return segments
