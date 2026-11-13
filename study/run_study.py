"""Primary study runner: default vs FA-calibrated thresholds under causal eval.

Example (Colab micro):
  python study/run_study.py --profile micro --seed 0 --disable-local-cache
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
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

from recap.data import load_market_data
from recap.recap import ReCAP, ReCAPConfig
from recap.utils import split_by_date
from study.causal_eval import evaluate_causal
from study.diagnostics import run_synthetic_diagnostic
from study.market_tickers import get_study_tickers
from study.reduced_pilot import apply_profile, build_parser as build_pilot_parser
from study.streaming_ard import (
    calibrate_threshold_scale,
    count_alarms,
    default_null_rate,
    fit_null_model,
    simulate_null,
    with_threshold,
)


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


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
    parser.add_argument(
        "--ard-mode",
        choices=["paper", "repo"],
        default="paper",
        help="paper: one-sided CUSUM with mu0 re-estimation and historical reference (paper Eq.); "
        "repo: original release behaviour",
    )
    parser.add_argument("--null-cal-reps", type=int, default=100, help="No-change sims used to choose h")
    parser.add_argument("--null-holdout-reps", type=int, default=200, help="Held-out no-change sims for reporting")
    parser.add_argument("--null-horizon-days", type=int, default=1008, help="Post-reference days per simulation")
    parser.add_argument("--tag", default="", help="Experiment label added to the output filename and JSON")
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
    cloned.detector = with_threshold(base.detector, cfg.cusum_threshold_scale)
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
        ard_mode=args.ard_mode,
        seed=args.seed,
        device=device,
        total_training_steps=None,
    )

    print(json.dumps({"phase": "fit_once_default_detector", "threshold_scale": 2.5, "ard_mode": args.ard_mode}, indent=2))
    base_runner = ReCAP(config=base_config, tickers=tickers, use_vix=True)
    base_runner.fit(train_data)
    detector = base_runner.detector
    history = train_data[train_data["tic"].isin(base_runner.tickers)].copy()

    daily_train = detector.build_daily_features(history)
    train_reference_days = detector.reference_days_for(len(daily_train))
    null_model = fit_null_model(daily_train, train_reference_days)
    sim_days = train_reference_days + int(args.null_horizon_days)
    sim_reference_days = detector.reference_days_for(sim_days)
    print(json.dumps({"phase": "simulate_no_change", **null_model.describe()}, indent=2, default=float))
    calibration_sims = simulate_null(null_model, sim_days, args.null_cal_reps, seed=10_000 + args.seed)
    holdout_sims = simulate_null(null_model, sim_days, args.null_holdout_reps, seed=20_000 + args.seed)

    null_calibration: dict[str, object] = {
        "null_model": null_model.describe(),
        "sim_days": sim_days,
        "sim_reference_days": sim_reference_days,
        "calibration_reps": args.null_cal_reps,
        "holdout_reps": args.null_holdout_reps,
        "calibration_seed": 10_000 + args.seed,
        "holdout_seed": 20_000 + args.seed,
        "default_2.5": default_null_rate(detector, holdout_sims, sim_reference_days),
    }

    calibrations: dict[str, dict[str, object]] = {}
    for budget in budgets:
        print(json.dumps({"phase": "calibrate_h", "target_alarms_per_year": budget}, indent=2))
        scale, meta = calibrate_threshold_scale(
            detector,
            calibration_sims,
            holdout_sims,
            reference_days=sim_reference_days,
            target_alarms_per_year=budget,
        )
        calibrations[f"budget_{budget:g}"] = meta

    historical_train_alarms: dict[str, object] = {}
    for name, scale in {"default_2.5": 2.5, **{k: v["threshold_scale"] for k, v in calibrations.items()}}.items():
        log = count_alarms(with_threshold(detector, scale), history)
        historical_train_alarms[name] = {
            "threshold_scale": float(scale),
            "n_alarms": log.n_alarms,
            "alarms_per_year": log.alarms_per_year,
            "alarm_dates": log.alarm_dates,
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
            history=history,
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
            history=history,
        )

    synthetic = None
    if not args.skip_synthetic:
        print(json.dumps({"phase": "synthetic_shift_diagnostic"}, indent=2))
        synthetic = run_synthetic_diagnostic(
            detector,
            null_model,
            {k: float(v) for k, v in variants.items()},
            sim_days=sim_days,
            reference_days=sim_reference_days,
            seed=30_000 + args.seed,
        )

    out_dir = ROOT / "outputs" / "study"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = f"_{args.tag}" if args.tag else ""
    out_path = out_dir / f"study_{args.profile}{tag}_{args.dataset.lower()}_seed{args.seed}_{stamp}.json"
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
        "run_id": out_path.stem,
        "created_utc": stamp,
        "git_commit": git_commit(),
        "argv": sys.argv[1:],
        "protocol_notes": [
            f"ARD mode: {args.ard_mode} (see docs/DEVIATIONS.md).",
            "Policy library fitted once with default h=2.5σ on train.",
            "Threshold variants change test-time ARD h only; features, kappa, mu0 updates and resets unchanged.",
            "h calibrated on no-change simulations (VAR(1) fitted to the train reference window); "
            "achieved rate reported on separate held-out simulations.",
            "Historical (train/test) alarms are reported as alarms, not labelled false.",
            "Causal eval disables max(baseline, adapted) look-ahead selection; "
            "in paper mode train is streamed through the detector before test.",
            "Thresholds frozen before test eval.",
        ],
        "ard_mode": args.ard_mode,
        "tag": args.tag,
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
        "null_calibration": null_calibration,
        "calibrations": calibrations,
        "historical_train_alarms": historical_train_alarms,
        "results": compact_results,
        "synthetic_diagnostic": synthetic,
    }
    out_path.write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")

    summary = {
        "saved": str(out_path),
        "default_holdout_null_rate": null_calibration["default_2.5"]["holdout_sim_alarms_per_year"],
        "calibrations": {
            name: {
                "target": cal["target_alarms_per_year"],
                "threshold_scale": cal["threshold_scale"],
                "holdout_sim_alarms_per_year": cal["holdout_sim_alarms_per_year"],
                "within_tolerance": cal["within_tolerance"],
            }
            for name, cal in calibrations.items()
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
