"""Reduced real-data pilot for Colab / modest hardware.

Default: DOW30, paper date split, tiny PPO budgets (not paper-scale).

Usage (repo root):
    python study/reduced_pilot.py
    python study/reduced_pilot.py --dataset DOW30 --pretrain-timesteps 1024
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os

# Colab/yfinance often break on curl_cffi browser impersonation mismatches.
os.environ.setdefault("YF_DISABLE_CURL_CFFI", "1")

import torch

from recap.data import load_market_data
from recap.recap import ReCAP, ReCAPConfig
from recap.utils import split_by_date
from study.market_tickers import get_study_tickers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Reduced ReCAP real-data pilot")
    parser.add_argument("--dataset", default="DOW30", choices=["DOW30"])
    parser.add_argument(
        "--profile",
        default="micro",
        choices=["micro", "reduced"],
        help="micro = Colab-safe tiny subset; reduced = fuller but still below paper scale",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--train-end-date", default=None)
    parser.add_argument("--test-end-date", default=None)
    parser.add_argument("--max-tickers", type=int, default=None, help="Use only the first N tickers")
    parser.add_argument("--pretrain-timesteps", type=int, default=None)
    parser.add_argument("--finetune-timesteps", type=int, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gate-epochs", type=int, default=None)
    parser.add_argument("--min-segment-days", type=int, default=None)
    parser.add_argument("--cusum-threshold-scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda")
    parser.add_argument("--data-dir", default=str(ROOT / "datasets"))
    parser.add_argument("--disable-local-cache", action="store_true")
    parser.add_argument("--online-update", action="store_true")
    return parser


def apply_profile(args: argparse.Namespace) -> argparse.Namespace:
    if args.profile == "micro":
        defaults = {
            "start_date": "2019-01-01",
            "train_end_date": "2020-06-01",
            "test_end_date": "2021-06-01",
            "max_tickers": 8,
            "pretrain_timesteps": 256,
            "finetune_timesteps": 64,
            "n_steps": 64,
            "batch_size": 32,
            "gate_epochs": 5,
            "min_segment_days": 20,
        }
    else:
        defaults = {
            "start_date": "2008-05-01",
            "train_end_date": "2020-05-01",
            "test_end_date": "2025-04-29",
            "max_tickers": None,
            "pretrain_timesteps": 1024,
            "finetune_timesteps": 256,
            "n_steps": 256,
            "batch_size": 64,
            "gate_epochs": 20,
            "min_segment_days": 20,
        }

    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    return args


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def main() -> None:
    args = apply_profile(build_parser().parse_args())
    device = resolve_device(args.device)
    tickers = get_study_tickers(args.dataset)
    if args.max_tickers is not None:
        tickers = tickers[: max(1, int(args.max_tickers))]

    print(
        json.dumps(
            {
                "phase": "load_data",
                "dataset": args.dataset,
                "profile": args.profile,
                "n_tickers": len(tickers),
                "tickers": tickers,
                "dates": {
                    "start": args.start_date,
                    "train_end": args.train_end_date,
                    "test_end": args.test_end_date,
                },
                "device": device,
            },
            indent=2,
        )
    )

    data = load_market_data(
        dataset_name=args.dataset,
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.test_end_date,
        use_vix=True,
        data_dir=args.data_dir,
        prefer_local_cache=not args.disable_local_cache,
    )
    train_data, test_data = split_by_date(data, train_end_date=args.train_end_date)

    config = ReCAPConfig(
        initial_cash=1_000_000.0,
        transaction_cost=1e-3,
        pretrain_timesteps=args.pretrain_timesteps,
        finetune_timesteps=args.finetune_timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gate_epochs=args.gate_epochs,
        min_segment_days=args.min_segment_days,
        cusum_threshold_scale=args.cusum_threshold_scale,
        seed=args.seed,
        device=device,
        total_training_steps=None,
    )

    runner = ReCAP(config=config, tickers=tickers, use_vix=True)
    results = runner.run(
        train_data=train_data,
        test_data=test_data,
        online_update=args.online_update,
    )

    out_dir = ROOT / "outputs" / "pilots"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"{args.dataset.lower()}_reduced_seed{args.seed}_{stamp}.json"

    payload = {
        "mode": "reduced_real_data_pilot",
        "profile": args.profile,
        "warning": "Budgets are intentionally below paper scale. Do not compare these metrics to Table 2 as replication.",
        "dataset": args.dataset,
        "tickers": tickers,
        "dates": {
            "start": args.start_date,
            "train_end": args.train_end_date,
            "test_end": args.test_end_date,
        },
        "device": device,
        "config": asdict(config),
        "n_tickers_requested": len(tickers),
        "n_tickers_used": len(runner.tickers),
        "library_size": results["library_size"],
        "train_segments": results["train_segments"],
        "metrics": results["metrics"],
        "segment_summaries": results.get("segment_summaries", []),
    }
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    print(json.dumps({"metrics": results["metrics"], "saved": str(out_path)}, indent=2, default=float))


if __name__ == "__main__":
    main()
