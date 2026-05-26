from __future__ import annotations

from dataclasses import dataclass

from finrl.config_tickers import DOW_30_TICKER, NAS_100_TICKER, SP_500_TICKER


NIKKEI_30_TICKER = [
    "7203.T",
    "6758.T",
    "9984.T",
    "8306.T",
    "6861.T",
    "9433.T",
    "6501.T",
    "8035.T",
    "6367.T",
    "6902.T",
    "6098.T",
    "4063.T",
    "6971.T",
    "8058.T",
    "4502.T",
    "8031.T",
    "4519.T",
    "6954.T",
    "7741.T",
    "4568.T",
    "6273.T",
    "9983.T",
    "8766.T",
    "8001.T",
    "6503.T",
    "7267.T",
    "4503.T",
    "7974.T",
    "2914.T",
    "4661.T",
]

COMMODITY_ETF_TICKER = [
    "GLD",
    "SLV",
    "DBC",
    "USO",
    "UNG",
    "DBA",
    "PDBC",
    "GSG",
]


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    tickers: list[str]
    use_vix: bool


DATASET_REGISTRY = {
    "DOW30": DatasetSpec(name="DOW30", tickers=list(DOW_30_TICKER), use_vix=True),
    "NAS100": DatasetSpec(name="NAS100", tickers=list(NAS_100_TICKER), use_vix=True),
    "SP500": DatasetSpec(name="SP500", tickers=list(SP_500_TICKER), use_vix=True),
    "NIKKEI30": DatasetSpec(name="NIKKEI30", tickers=NIKKEI_30_TICKER, use_vix=False),
    "COMMODITY_ETF": DatasetSpec(name="COMMODITY_ETF", tickers=COMMODITY_ETF_TICKER, use_vix=False),
}


def get_dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASET_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(DATASET_REGISTRY))
        raise ValueError(f"Unknown dataset: {name}. Available choices: {choices}") from exc
