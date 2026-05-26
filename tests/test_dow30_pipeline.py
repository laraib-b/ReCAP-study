from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recap.data import load_market_data
from recap.recap import ReCAP, ReCAPConfig
from recap.tickers import get_dataset_spec
from recap.utils import split_by_date


@pytest.mark.integration
def test_dow30_full_pipeline_with_local_cache() -> None:
    data_dir = PROJECT_ROOT / "datasets"
    cache_path = data_dir / "Yahoo_DOW_30_TICKER_1D_cache.csv"
    if not cache_path.exists():
        pytest.skip("DOW30 本地缓存不存在，跳过真实数据集成测试")

    dataset = get_dataset_spec("DOW30")
    data = load_market_data(
        dataset_name=dataset.name,
        tickers=dataset.tickers,
        start_date="2008-05-01",
        end_date="2012-12-31",
        use_vix=dataset.use_vix,
        data_dir=data_dir,
        prefer_local_cache=True,
    )
    train_data, test_data = split_by_date(data, train_end_date="2011-01-01")

    runner = ReCAP(
        config=ReCAPConfig(
            initial_cash=100_000.0,
            pretrain_timesteps=128,
            finetune_timesteps=64,
            n_steps=32,
            batch_size=32,
            min_segment_days=20,
            cusum_threshold_scale=1.0,
            gate_epochs=20,
            max_library_size=6,
            similarity_threshold=0.98,
            seed=0,
            device="cpu",
        ),
        tickers=dataset.tickers,
        use_vix=dataset.use_vix,
    )

    results = runner.run(train_data=train_data, test_data=test_data, online_update=True)
    metrics = results["metrics"]
    assert isinstance(metrics, dict)
    assert set(metrics) == {"CR", "SR", "MDD"}
    assert results["library_size"] >= 1
    assert len(results["segment_summaries"]) >= 1
    assert pd.notna(metrics["CR"])
    assert pd.notna(metrics["SR"])
    assert pd.notna(metrics["MDD"])
