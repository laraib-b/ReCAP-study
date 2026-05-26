from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from recap.data import load_market_data
from recap.recap import ReCAP, ReCAPConfig
from recap.tickers import get_dataset_spec
from recap.utils import split_by_date


@pytest.mark.effectiveness
def test_dow30_effectiveness_with_paper_range() -> None:
    data_dir = PROJECT_ROOT / "datasets"
    cache_path = data_dir / "Yahoo_DOW_30_TICKER_1D_cache.csv"
    if not cache_path.exists():
        pytest.skip("DOW30 本地缓存不存在，跳过有效性测试")

    dataset = get_dataset_spec("DOW30")
    data = load_market_data(
        dataset_name=dataset.name,
        tickers=dataset.tickers,
        start_date="2008-05-01",
        end_date="2025-04-29",
        use_vix=dataset.use_vix,
        data_dir=data_dir,
        prefer_local_cache=True,
    )
    train_data, test_data = split_by_date(data, train_end_date="2020-05-01")

    runner = ReCAP(
        config=ReCAPConfig(
            pretrain_timesteps=2_048,
            finetune_timesteps=256,
            max_library_size=20,
            similarity_threshold=0.5,
            min_segment_days=20,
            cusum_threshold_scale=2.5,
            gate_epochs=50,
            batch_size=256,
            n_steps=256,
            learning_rate=1e-4,
            gamma=0.99,
            ent_coef=0.01,
            clip_range=0.2,
            n_epochs=10,
            gae_lambda=0.95,
            max_grad_norm=0.5,
            vf_coef=0.3695,
            seed=0,
            device="cpu",
        ),
        tickers=dataset.tickers,
        use_vix=dataset.use_vix,
    )

    results = runner.run(train_data=train_data, test_data=test_data, online_update=False)
    metrics = results["metrics"]
    assert float(metrics["CR"]) > 0.0
    assert float(metrics["SR"]) > 0.0
    assert float(metrics["MDD"]) < 0.8
