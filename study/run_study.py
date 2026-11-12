"""Primary study runner: default vs FA-calibrated thresholds under causal eval.

Example (Colab micro):
  python study/run_study.py --profile micro --seed 0 --disable-local-cache
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("YF_DISABLE_CURL_CFFI", "1")

import pandas as pd
import torch

from recap.ard import AdaptiveRegimeDetector
from recap.data import load_market_data
from recap.recap import ReCAP, ReCAPConfig
from recap.utils import split_by_date
from study.causal_eval import evaluate_causal
from study.diagnostics import run_synthetic_diagnostic
from study.market_tickers import get_study_tickers
from study.reduced_pilot import apply_profile, build_parser as build_pilot_parser
from study.streaming_ard import calibrate_threshold_scale


def resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def build_parser() -> argparse.ArgumentParser:
    parser = build_pilot_parser()
    parser.description = "False-alarm-calibrated ReCAP study runner (causal eval)"
    parser.add_argument(
        "--budgets",
        default="0.5,1,2",
        help="Comma-separated target alarms/year for calibration",
    )
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument(
        "--no-adapt-control",
        action="store_true",
        help="Also run no-adaptation causal control",
    )
    return parser


def clone_fitted_runner(base: ReCAP, threshold_scale: float) -> ReCAP:
    """Copy a fitted ReCAP and swap only the detector threshold (test-time effect)."""
    cfg = deepcopy(base.config)
    cfg.cusum_threshold_scale = float(threshold_scale)
    cloned = ReCAP(config=cfg, tickers=list(base.tickers), use_vix=base.use_vix)
    cloned.base_model = base.base_model
    cloned.base_actor_parameters = {
        name: tensor.detach().clone() for name, tensor in base.base_actor_parameters.items()
    }
    cloned.policy_library = [
        {name: tensor.detach().clone() for name, tensor in vector.items()}
        for vector in base.policy_library
    ]
    cloned.gate = base.gate
    cloned.gate_reference_data = (
        None if base.gate_reference_data is None else base.gate_reference_data.copy()
    )
    cloned.detector = AdaptiveRegimeDetector(
        min_segment_days=cfg.min_segment_days,
        threshold_scale=cfg.cusum_threshold_scale,
        vote_threshold=cfg.ard_vote_threshold,
    )
    return cloned


def main() -> None:
    args = apply_profile(build_parser().parse_args())
    device = resolve_device(args.device)
    tickers = get_study_tickers(args.dataset)
    if args.max_tickers is not None:
        tickers = tickers[: max(1, int(args.max_tickers))]
    budgets = [float(x.strip()) for x in str(args.budgets).split(",") if x.strip()]

    print(
        json.dumps(
            {
                "phase": "load_data",
                "profile": args.profile,
                "dataset": args.dataset,
                "n_tickers": len(tickers),
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

    base_config = ReCAPConfig(
        initial_cash=1_000_000.0,
        transaction_cost=1e-3,
        pretrain_timesteps=args.pretrain_timesteps,
        finetune_timesteps=args.finetune_timesteps,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gate_epochs=args.gate_epochs,
        min_segment_days=args.min_segment_days,
        cusum_threshold_scale=2.5,
        seed=args.seed,
        device=device,
        total_training_steps=None,
    )

    print(json.dumps({"phase": "fit_once_default_detector", "threshold_scale": 2.5}, indent=2))
    base_runner = ReCAP(config=base_config, tickers=tickers, use_vix=True)
    base_runner.fit(train_data)

    daily_train = base_runner.detector.build_daily_features(train_data)
    calibrations: dict[str, dict[str, object]] = {}
    for budget in budgets:
        scale, log, meta = calibrate_threshold_scale(
            daily_train,
            target_alarms_per_year=budget,
            min_segment_days=args.min_segment_days,
        )
        calibrations[f"budget_{budget:g}"] = {
            "target_alarms_per_year": budget,
            "threshold_scale": scale,
            "train_alarms": log.n_alarms,
            "train_alarms_per_year": log.alarms_per_year,
            "train_alarm_dates": log.alarm_dates,
            **meta,
        }

    variants: dict[str, float] = {"default_2.5": 2.5}
    scale_owner: dict[float, str] = {2.5: "default_2.5"}
    aliases: dict[str, str] = {}
    for name, payload in calibrations.items():
        scale = float(payload["threshold_scale"])
        # Deduplicate identical calibrated scales (one eval, aliases for others).
        match = next((s for s in scale_owner if abs(scale - s) < 1e-6), None)
        if match is not None:
            aliases[name] = scale_owner[match]
            payload["skipped_eval_duplicate_scale"] = True
            payload["alias_of"] = scale_owner[match]
            continue
        variants[name] = scale
        scale_owner[scale] = name
        payload["skipped_eval_duplicate_scale"] = False

    results: dict[str, object] = {}
    for name, scale in variants.items():
        print(json.dumps({"phase": "causal_eval", "variant": name, "threshold_scale": scale}, indent=2))
        runner = clone_fitted_runner(base_runner, threshold_scale=scale)
        results[name] = evaluate_causal(
            runner,
            test_data,
            online_update=True,
            adapt_on_alarm=True,
        )
    for alias_name, owner in aliases.items():
        results[alias_name] = dict(results[owner])
        results[alias_name]["alias_of"] = owner

    if args.no_adapt_control:
        runner = clone_fitted_runner(base_runner, threshold_scale=2.5)
        results["no_adapt_default"] = evaluate_causal(
            runner,
            test_data,
            online_update=False,
            adapt_on_alarm=False,
        )

    synthetic = None
    if not args.skip_synthetic:
        synthetic = run_synthetic_diagnostic({k: float(v) for k, v in variants.items()})

    out_dir = ROOT / "outputs" / "study"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"study_{args.profile}_{args.dataset.lower()}_seed{args.seed}_{stamp}.json"
    daily_dir = out_dir / f"daily_{stamp}"
    daily_dir.mkdir(parents=True, exist_ok=True)

    compact_results: dict[str, object] = {}
    for name, payload in results.items():
        daily = payload.get("daily", [])
        daily_path = daily_dir / f"{name}.json"
        daily_path.write_text(json.dumps(daily, indent=2), encoding="utf-8")
        compact = dict(payload)
        compact["daily_path"] = str(daily_path.relative_to(ROOT)).replace("\\", "/")
        compact.pop("daily", None)
        compact_results[name] = compact

    payload = {
        "mode": "false_alarm_calibrated_study",
        "protocol_notes": [
            "Policy library fitted once with default h=2.5σ on train.",
            "Threshold variants change test-time ARD only (isolates detector effect).",
            "Causal eval disables max(baseline, adapted) look-ahead selection.",
            "Train-only calibration; thresholds frozen before test eval.",
        ],
        "profile": args.profile,
        "dataset": args.dataset,
        "seed": args.seed,
        "device": device,
        "tickers": tickers,
        "dates": {
            "start": args.start_date,
            "train_end": args.train_end_date,
            "test_end": args.test_end_date,
        },
        "warning": (
            "Pilot-scale compute/profile may be below paper budgets. "
            "Do not claim paper Table 2 replication."
        ),
        "config": asdict(base_config),
        "calibrations": calibrations,
        "results": compact_results,
        "synthetic_diagnostic": synthetic,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    summary = {
        "saved": str(out_path),
        "calibrations": {
            name: {
                "target": payload["target_alarms_per_year"],
                "threshold_scale": payload["threshold_scale"],
                "train_alarms_per_year": payload["train_alarms_per_year"],
            }
            for name, payload in calibrations.items()
        },
        "variants": {
            name: {
                "CR": compact_results[name]["metrics"]["CR"],
                "SR": compact_results[name]["metrics"]["SR"],
                "MDD": compact_results[name]["metrics"]["MDD"],
                "alarms_per_year": compact_results[name]["detector"]["alarms_per_year"],
                "adaptations_per_year": compact_results[name]["adaptations_per_year"],
            }
            for name in compact_results
        },
    }
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
