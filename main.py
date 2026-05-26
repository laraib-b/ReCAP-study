from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from dataclasses import asdict

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from recap.data import load_market_data
from recap.recap import ReCAP, ReCAPConfig
from recap.tickers import get_dataset_spec
from recap.utils import split_by_date


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Minimal ReCAP review release")
    parser.add_argument("--dataset", choices=["DOW30", "NAS100", "SP500", "NIKKEI30", "COMMODITY_ETF"], default="DOW30")
    parser.add_argument("--start-date", default="2008-05-01")
    parser.add_argument("--train-end-date", default="2020-05-01")
    parser.add_argument("--test-end-date", default="2025-04-29")
    parser.add_argument("--initial-cash", type=float, default=1_000_000.0)
    parser.add_argument("--transaction-cost", type=float, default=1e-3)
    parser.add_argument("--total-training-steps", type=int, default=2_000_000)
    parser.add_argument("--pretrain-timesteps", type=int)
    parser.add_argument("--finetune-timesteps", type=int, default=10_000)
    parser.add_argument("--online-update", action="store_true")
    parser.add_argument("--max-library-size", type=int, default=20)
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--min-segment-days", type=int, default=60)
    parser.add_argument("--cusum-threshold-scale", type=float, default=2.5)
    parser.add_argument("--gate-epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--vf-coef", type=float, default=0.3695)
    parser.add_argument("--gate-learning-rate", type=float, default=1e-3)
    parser.add_argument("--data-dir", default=str(Path(__file__).resolve().parent / "datasets"))
    parser.add_argument("--disable-local-cache", action="store_true")
    return parser


def build_payload(
    *,
    dataset_name: str,
    config: ReCAPConfig,
    results: dict[str, object]
) -> dict[str, object]:
    segment_summaries = results.get("segment_summaries", [])
    payload = {
        "dataset": dataset_name,
        "config": asdict(config),
        "results": {
            "library_size": results["library_size"],
            "train_segments": results["train_segments"],
            "test_segments": len(segment_summaries) if isinstance(segment_summaries, list) else 0,
            "metrics": results["metrics"],
        },
    }
    return payload


def main() -> None:
    args = build_parser().parse_args()
    dataset = get_dataset_spec(args.dataset)
    data = load_market_data(
        dataset_name=dataset.name,
        tickers=dataset.tickers,
        start_date=args.start_date,
        end_date=args.test_end_date,
        use_vix=dataset.use_vix,
        data_dir=args.data_dir,
        prefer_local_cache=not args.disable_local_cache,
    )
    train_data, test_data = split_by_date(data, train_end_date=args.train_end_date)

    config = ReCAPConfig(
        initial_cash=args.initial_cash,
        transaction_cost=args.transaction_cost,
        total_training_steps=args.total_training_steps,
        pretrain_timesteps=args.pretrain_timesteps,
        finetune_timesteps=args.finetune_timesteps,
        max_library_size=args.max_library_size,
        similarity_threshold=args.similarity_threshold,
        min_segment_days=args.min_segment_days,
        cusum_threshold_scale=args.cusum_threshold_scale,
        gate_epochs=args.gate_epochs,
        seed=args.seed,
        device=args.device,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        ent_coef=args.ent_coef,
        clip_range=args.clip_range,
        n_epochs=args.n_epochs,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        gate_learning_rate=args.gate_learning_rate,
    )

    runner = ReCAP(config=config, tickers=dataset.tickers, use_vix=dataset.use_vix)
    results = runner.run(train_data=train_data, test_data=test_data, online_update=args.online_update)

    payload = build_payload(
        dataset_name=args.dataset,
        config=config,
        results=results
    )
    print(json.dumps(payload, indent=2, default=float))


if __name__ == "__main__":
    main()
