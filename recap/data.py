from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf


BASE_COLUMNS = ["open", "high", "low", "close", "volume"]
TECHNICAL_FEATURE_COLUMNS = [
    "macd",
    "boll_ub",
    "boll_lb",
    "rsi_30",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    "zd_5",
    "zd_10",
    "zd_15",
    "zd_20",
    "zd_25",
    "zd_30",
    "zopen",
    "zhigh",
    "zlow",
    "zadjcp",
    "zclose",
    "vix",
    "turbulence",
]
FEATURE_COLUMNS = [*TECHNICAL_FEATURE_COLUMNS, *BASE_COLUMNS]
MARKET_FEATURE_COLUMNS = ["vix", "turbulence", "boll_ub", "boll_lb", "zd_5", "rsi_30"]
MARKET_FEATURE_FALLBACK_COLUMNS = [
    "macd",
    "cci_30",
    "dx_30",
    "close_30_sma",
    "close_60_sma",
    "zd_10",
    "zclose",
]

LOCAL_CACHE_FILES = {
    "DOW30": "Yahoo_DOW_30_TICKER_1D_cache.csv",
    "NAS100": "Yahoo_NAS_100_TICKER_1D_cache.csv",
    "SP500": "Yahoo_SP_500_TICKER_1D_cache.csv",
    "NIKKEI30": "Yahoo_NIKKEI_30_TICKER_1D_cache.csv",
    "COMMODITY_ETF": "Yahoo_COMMODITY_ETF_TICKER_1D_cache.csv",
}

LOCAL_PICKLE_FILES = {
    "DOW30": "Yahoo_DOW30_1D.pkl",
    "NAS100": "Yahoo_NAS100_1D.pkl",
    "SP500": "Yahoo_SP500_1D.pkl",
    "NIKKEI30": "Yahoo_NIKKEI30_1D.pkl",
    "COMMODITY_ETF": "Yahoo_COMMODITY_ETF_1D.pkl",
}


def load_market_data(
    dataset_name: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
    use_vix: bool,
    data_dir: str | Path | None = None,
    prefer_local_cache: bool = True,
) -> pd.DataFrame:
    local_data: pd.DataFrame | None = None
    if prefer_local_cache:
        local_data = load_local_dataset(
            dataset_name=dataset_name,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
        )

    if local_data is None:
        local_data = download_price_data(tickers=tickers, start_date=start_date, end_date=end_date)

    return add_market_features(local_data, use_vix=use_vix)


def load_local_dataset(
    dataset_name: str,
    tickers: list[str],
    start_date: str,
    end_date: str,
    data_dir: str | Path | None = None,
) -> pd.DataFrame | None:
    directory = Path(data_dir) if data_dir is not None else Path(__file__).resolve().parents[1] / "datasets"
    csv_name = LOCAL_CACHE_FILES.get(dataset_name)
    pickle_name = LOCAL_PICKLE_FILES.get(dataset_name)

    if csv_name is not None:
        csv_path = directory / csv_name
        if csv_path.exists():
            return _load_local_csv(csv_path=csv_path, tickers=tickers, start_date=start_date, end_date=end_date)

    if pickle_name is not None:
        pickle_path = directory / pickle_name
        if pickle_path.exists():
            return _load_local_pickle(pickle_path=pickle_path, tickers=tickers, start_date=start_date, end_date=end_date)

    return None


def download_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
    batch_size: int = 50,
) -> pd.DataFrame:
    frames = []
    for start in range(0, len(tickers), batch_size):
        batch = tickers[start:start + batch_size]
        raw = yf.download(batch, start=start_date, end=end_date, auto_adjust=True, progress=False, group_by="ticker", threads=True)
        frames.extend(_normalize_download_frame(raw=raw, requested_tickers=batch))

    if not frames:
        raise ValueError("No price data was downloaded.")

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    data = data.sort_values(["date", "tic"]).reset_index(drop=True)
    return _keep_complete_trading_dates(data, tickers)


def add_market_features(data: pd.DataFrame, use_vix: bool) -> pd.DataFrame:
    df = data.copy()
    df = df.sort_values(["tic", "date"]).reset_index(drop=True)

    grouped = df.groupby("tic", group_keys=False)
    df["return_1"] = grouped["close"].pct_change().fillna(0.0)
    df["return_5"] = grouped["close"].pct_change(5).fillna(0.0)
    df["adjcp"] = pd.to_numeric(df.get("adjcp", df["close"]), errors="coerce")
    df["adjcp"] = df["adjcp"].fillna(df["close"])

    _ensure_feature_column(
        df,
        "close_30_sma",
        grouped["close"].transform(lambda values: values.rolling(window=30, min_periods=1).mean()),
    )
    _ensure_feature_column(
        df,
        "close_60_sma",
        grouped["close"].transform(lambda values: values.rolling(window=60, min_periods=1).mean()),
    )
    _ensure_feature_column(df, "macd", grouped["adjcp"].transform(_calculate_macd))
    boll_ub, boll_lb = _calculate_bollinger_bands(grouped["adjcp"])
    _ensure_feature_column(df, "boll_ub", boll_ub)
    _ensure_feature_column(df, "boll_lb", boll_lb)
    _ensure_feature_column(df, "rsi_30", grouped["adjcp"].transform(lambda values: _calculate_rsi(values, window=30)))
    grouped_prices = df.groupby("tic")[["high", "low", "close"]]
    _ensure_feature_column(
        df,
        "cci_30",
        grouped_prices.apply(_calculate_cci_30).reset_index(level=0, drop=True),
    )
    _ensure_feature_column(
        df,
        "dx_30",
        grouped_prices.apply(_calculate_dx_30).reset_index(level=0, drop=True),
    )

    for period in [5, 10, 15, 20, 25, 30]:
        _ensure_feature_column(
            df,
            f"zd_{period}",
            grouped["adjcp"].pct_change(period).fillna(0.0),
        )

    close_scale = np.clip(df["close"], 1e-12, None)
    _ensure_feature_column(df, "zopen", df["open"] / close_scale - 1.0)
    _ensure_feature_column(df, "zhigh", df["high"] / close_scale - 1.0)
    _ensure_feature_column(df, "zlow", df["low"] / close_scale - 1.0)
    _ensure_feature_column(df, "zclose", df["return_1"])
    _ensure_feature_column(df, "zadjcp", grouped["adjcp"].pct_change().fillna(0.0))

    df["ema_10"] = grouped["close"].transform(lambda values: values.ewm(span=10, adjust=False).mean())
    df["ema_10_ratio"] = df["close"] / np.clip(df["ema_10"], 1e-12, None) - 1.0
    df["regime_volatility"] = grouped["return_1"].transform(
        lambda values: values.rolling(window=20, min_periods=5).std().fillna(0.0)
    )
    df["rsi_14"] = grouped["close"].transform(_calculate_rsi)
    df["dispersion"] = df.groupby("date")["return_1"].transform(lambda values: float(values.std()) if len(values) > 1 else 0.0)

    has_vix = "vix" in df.columns and not df["vix"].isna().all()
    if use_vix and has_vix:
        df["vix"] = pd.to_numeric(df["vix"], errors="coerce").ffill().bfill()
    elif use_vix:
        vix_frame = _download_vix_frame(start_date=pd.Timestamp(df["date"].min()), end_date=pd.Timestamp(df["date"].max()))
        df = df.drop(columns=["vix"], errors="ignore")
        df = df.merge(vix_frame, on="date", how="left")
        df["vix"] = pd.to_numeric(df["vix"], errors="coerce").ffill().bfill()
    else:
        df["vix"] = np.nan

    if "turbulence" not in df.columns or df["turbulence"].isna().all():
        turbulence = _calculate_turbulence(df)
        df = df.drop(columns=["turbulence"], errors="ignore")
        df = df.merge(turbulence, on="date", how="left")

    df = df.drop(columns=["ema_10"])
    normalization_columns = [
        "adjcp",
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "cci_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
        "zd_5",
        "zd_10",
        "zd_15",
        "zd_20",
        "zd_25",
        "zd_30",
        "zopen",
        "zhigh",
        "zlow",
        "zadjcp",
        "zclose",
        "vix",
        "turbulence",
        "regime_volatility",
        "dispersion",
        "return_1",
        "return_5",
        "rsi_14",
        "ema_10_ratio",
    ]
    df[normalization_columns] = df[normalization_columns].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return _keep_complete_trading_dates(df, sorted(df["tic"].unique().tolist()))


def resolve_market_feature_columns(
    available_columns: list[str],
    data: pd.DataFrame | None = None,
    target_size: int = len(MARKET_FEATURE_COLUMNS),
) -> list[str]:
    selected: list[str] = []

    def is_usable(column: str) -> bool:
        if column not in available_columns:
            return False
        if data is None or column not in data.columns:
            return True
        series = pd.to_numeric(data[column], errors="coerce")
        return not series.isna().all()

    for column in MARKET_FEATURE_COLUMNS:
        if is_usable(column) and column not in selected:
            selected.append(column)

    for column in MARKET_FEATURE_FALLBACK_COLUMNS:
        if len(selected) >= target_size:
            break
        if is_usable(column) and column not in selected:
            selected.append(column)

    for column in available_columns:
        if len(selected) >= target_size:
            break
        if is_usable(column) and column not in selected:
            selected.append(column)

    return selected[:target_size]


def _normalize_download_frame(raw: pd.DataFrame, requested_tickers: list[str]) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []

    if raw.empty:
        return frames

    if isinstance(raw.columns, pd.MultiIndex):
        for ticker in requested_tickers:
            if ticker not in raw.columns.get_level_values(0):
                continue
            ticker_frame = raw[ticker].copy().reset_index()
            ticker_frame.columns = [column.lower() for column in ticker_frame.columns]
            ticker_frame["tic"] = ticker
            frames.append(ticker_frame[["date", "tic", *BASE_COLUMNS]])
        return frames

    ticker = requested_tickers[0]
    single_frame = raw.copy().reset_index()
    single_frame.columns = [column.lower() for column in single_frame.columns]
    single_frame["tic"] = ticker
    frames.append(single_frame[["date", "tic", *BASE_COLUMNS]])
    return frames


def _keep_complete_trading_dates(data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    expected_count = len(tickers)
    counts = Counter(data["date"])
    valid_dates = {date for date, count in counts.items() if count == expected_count}
    filtered = data[data["date"].isin(valid_dates)].copy()
    filtered = filtered.sort_values(["date", "tic"]).reset_index(drop=True)
    if filtered.empty:
        raise ValueError("No complete trading dates remain after alignment.")
    return filtered


def _calculate_rsi(close_prices: pd.Series, window: int = 14) -> pd.Series:
    delta = close_prices.diff().fillna(0.0)
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    average_gain = gain.rolling(window=window, min_periods=window).mean()
    average_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = average_gain / np.clip(average_loss, 1e-12, None)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def _calculate_macd(prices: pd.Series) -> pd.Series:
    ema_short = prices.ewm(span=12, adjust=False).mean()
    ema_long = prices.ewm(span=26, adjust=False).mean()
    return (ema_short - ema_long).fillna(0.0)


def _calculate_bollinger_bands(grouped: pd.core.groupby.generic.SeriesGroupBy) -> tuple[pd.Series, pd.Series]:
    rolling_mean = grouped.transform(lambda values: values.rolling(window=20, min_periods=1).mean())
    rolling_std = grouped.transform(lambda values: values.rolling(window=20, min_periods=1).std().fillna(0.0))
    boll_ub = rolling_mean + 2.0 * rolling_std
    boll_lb = rolling_mean - 2.0 * rolling_std
    return boll_ub.fillna(0.0), boll_lb.fillna(0.0)


def _calculate_cci_30(frame: pd.DataFrame) -> pd.Series:
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3.0
    rolling_mean = typical_price.rolling(window=30, min_periods=1).mean()
    mean_deviation = typical_price.rolling(window=30, min_periods=1).apply(
        lambda values: np.mean(np.abs(values - np.mean(values))),
        raw=True,
    )
    cci = (typical_price - rolling_mean) / np.clip(0.015 * mean_deviation, 1e-12, None)
    return cci.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _calculate_dx_30(frame: pd.DataFrame) -> pd.Series:
    high = frame["high"]
    low = frame["low"]
    close = frame["close"]
    up_move = high.diff().fillna(0.0)
    down_move = -low.diff().fillna(0.0)
    plus_dm = np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0)

    tr_components = pd.concat(
        [
            (high - low).abs(),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    )
    true_range = tr_components.max(axis=1).fillna(0.0)
    atr = true_range.rolling(window=30, min_periods=1).mean()
    plus_di = 100.0 * pd.Series(plus_dm, index=frame.index).rolling(window=30, min_periods=1).mean() / np.clip(atr, 1e-12, None)
    minus_di = 100.0 * pd.Series(minus_dm, index=frame.index).rolling(window=30, min_periods=1).mean() / np.clip(atr, 1e-12, None)
    dx = 100.0 * (plus_di - minus_di).abs() / np.clip(plus_di + minus_di, 1e-12, None)
    return dx.replace([np.inf, -np.inf], 0.0).fillna(0.0)


def _calculate_turbulence(data: pd.DataFrame, lookback: int = 30) -> pd.DataFrame:
    returns = data.pivot(index="date", columns="tic", values="close").pct_change().fillna(0.0)
    turbulence_values = np.zeros(len(returns), dtype=np.float32)
    for index in range(len(returns)):
        start = max(0, index - lookback)
        history = returns.iloc[start:index]
        if history.shape[0] < 5:
            continue
        mean = history.mean(axis=0).to_numpy(dtype=np.float64)
        cov = history.cov().to_numpy(dtype=np.float64)
        cov += np.eye(cov.shape[0], dtype=np.float64) * 1e-6
        current = returns.iloc[index].to_numpy(dtype=np.float64)
        diff = current - mean
        turbulence_values[index] = float(diff @ np.linalg.pinv(cov) @ diff)
    return pd.DataFrame({"date": returns.index, "turbulence": turbulence_values})


def _ensure_feature_column(df: pd.DataFrame, column: str, values: pd.Series | np.ndarray) -> None:
    computed = pd.Series(values, index=df.index, dtype=np.float64)
    if column not in df.columns:
        df[column] = computed
        return
    existing = pd.to_numeric(df[column], errors="coerce")
    df[column] = existing.where(~existing.isna(), computed)


def _download_vix_frame(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    raw = yf.download("^VIX", start=start_date, end=end_date, auto_adjust=True, progress=False, threads=False)
    if raw.empty:
        return pd.DataFrame({"date": pd.date_range(start_date, end_date, freq="B"), "vix": np.nan})
    frame = raw.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "vix"})
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


def _load_local_csv(csv_path: Path, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    data = pd.read_csv(csv_path)
    data.columns = [str(column).lower() for column in data.columns]
    data = data.drop(columns=["unnamed: 0"], errors="ignore")
    if "date" not in data.columns or "tic" not in data.columns:
        raise ValueError(f"Local cache file is missing required columns: {csv_path}")

    data["date"] = pd.to_datetime(data["date"])
    data = data[data["tic"].isin(tickers)].copy()
    data = data[(data["date"] >= pd.Timestamp(start_date)) & (data["date"] <= pd.Timestamp(end_date))].copy()
    data = data.sort_values(by=["date", "tic"]).reset_index(drop=True)
    available_tickers = sorted(data["tic"].unique().tolist())
    if not available_tickers:
        raise ValueError(f"No requested tickers were found in local cache: {csv_path}")
    base_columns = ["date", "tic", *BASE_COLUMNS]
    optional_columns = [column for column in ["adjcp", "vix", "turbulence"] if column in data.columns]
    return _keep_complete_trading_dates(data[base_columns + optional_columns], available_tickers)


def _load_local_pickle(pickle_path: Path, tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    data = pd.read_pickle(pickle_path)
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"Expected DataFrame in pickle file: {pickle_path}")

    data.columns = [str(column).lower() for column in data.columns]
    data["date"] = pd.to_datetime(data["date"])
    data = data[data["tic"].isin(tickers)].copy()
    data = data[(data["date"] >= pd.Timestamp(start_date)) & (data["date"] <= pd.Timestamp(end_date))].copy()
    data = data.sort_values(by=["date", "tic"]).reset_index(drop=True)
    available_tickers = sorted(data["tic"].unique().tolist())
    if not available_tickers:
        raise ValueError(f"No requested tickers were found in local pickle: {pickle_path}")
    base_columns = ["date", "tic", *BASE_COLUMNS]
    optional_columns = [column for column in ["adjcp", "vix", "turbulence"] if column in data.columns]
    return _keep_complete_trading_dates(data[base_columns + optional_columns], available_tickers)
