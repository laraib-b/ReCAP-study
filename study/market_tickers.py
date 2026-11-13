"""Ticker lists for study pilots without requiring the full FinRL install."""

from __future__ import annotations

# Standard FinRL DOW_30_TICKER list (stable snapshot for reproducible pilots).
DOW30_TICKERS = [
    "AXP",
    "AMGN",
    "AAPL",
    "BA",
    "CAT",
    "CSCO",
    "CVX",
    "GS",
    "HD",
    "HON",
    "IBM",
    "INTC",
    "JNJ",
    "KO",
    "JPM",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "TRV",
    "UNH",
    "CRM",
    "VZ",
    "V",
    # WBA removed: delisted / Yahoo returns no quotes (fills as all-zero prices).
    "WMT",
    "DIS",
    # DOW removed: IPO 2019-03-20; including it truncates a 2008+ panel to 2019
    # (or injects pre-IPO zeros). drop_unusable_tickers also guards against this.
]

DATASET_TICKERS = {
    "DOW30": DOW30_TICKERS,
}


def get_study_tickers(dataset: str) -> list[str]:
    try:
        return list(DATASET_TICKERS[dataset])
    except KeyError as exc:
        raise ValueError(f"Unsupported study dataset: {dataset}. Choose from {sorted(DATASET_TICKERS)}") from exc
