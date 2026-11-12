"""Plain-requests Yahoo chart downloader (no curl_cffi / browser impersonation)."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import requests

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


def _to_epoch(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def download_ticker_history(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
    params = {
        "period1": _to_epoch(start_date),
        "period2": _to_epoch(end_date),
        "interval": "1d",
        "events": "history",
        "includeAdjustedClose": "true",
    }
    response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=30)
    response.raise_for_status()
    payload = response.json()
    result = payload.get("chart", {}).get("result")
    if not result:
        raise ValueError(f"No chart result for {ticker}")

    block = result[0]
    timestamps = block.get("timestamp") or []
    quote = (block.get("indicators", {}).get("quote") or [{}])[0]
    adj = (block.get("indicators", {}).get("adjclose") or [{}])[0]

    if not timestamps:
        raise ValueError(f"Empty history for {ticker}")

    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s").tz_localize("UTC").tz_convert(None).normalize(),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
            "adjcp": adj.get("adjclose", quote.get("close")),
            "tic": ticker,
        }
    )
    frame = frame.dropna(subset=["close"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"All rows empty for {ticker}")
    return frame


def download_price_data_requests(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    errors: list[str] = []
    for ticker in tickers:
        try:
            frames.append(download_ticker_history(ticker, start_date, end_date))
        except Exception as exc:  # noqa: BLE001 - collect per-ticker failures
            errors.append(f"{ticker}: {exc}")

    if not frames:
        raise ValueError("No price data was downloaded via requests. " + "; ".join(errors[:5]))

    data = pd.concat(frames, ignore_index=True)
    data["date"] = pd.to_datetime(data["date"])
    return data.sort_values(["date", "tic"]).reset_index(drop=True)
