"""Aggregate study JSONs across seeds into paper tables (mean ± sd).

Example:
  python study/aggregate.py outputs/study/study_micro_dow30_seed*.json --out results/tables
Uses the latest paper-mode run per seed.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import pandas as pd


def load_runs(patterns: list[str], ard_mode: str, tag: str | None = "") -> list[dict]:
    """Load latest paper/repo run per seed.

    tag=None means accept any tag (useful when seeds used different --tag values).
    tag="" means only untagged runs.
    """
    latest: dict[int, dict] = {}
    for path in sorted({p for pattern in patterns for p in glob.glob(pattern)}):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("ard_mode") != ard_mode or "results" not in payload:
            continue
        if tag is not None and payload.get("tag", "") != tag:
            continue
        seed = int(payload["seed"])
        if seed not in latest or payload["created_utc"] > latest[seed]["created_utc"]:
            latest[seed] = payload
    return [latest[s] for s in sorted(latest)]


def run_rows(runs: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    portfolio, detector, synthetic = [], [], []
    for run in runs:
        seed = run["seed"]
        null_rates = {"default_2.5": run["null_calibration"]["default_2.5"], **run["calibrations"]}
        for name, result in run["results"].items():
            portfolio.append(
                {
                    "seed": seed,
                    "variant": name,
                    "CR": result["metrics"]["CR"],
                    "SR": result["metrics"]["SR"],
                    "MDD": result["metrics"]["MDD"],
                    "adaptations_per_year": result["adaptations_per_year"],
                    "test_alarms_per_year": result["detector"]["alarms_per_year"],
                    "turnover_proxy": result.get("turnover_proxy"),
                    "library_size": result.get("library_size"),
                }
            )
        for name, cal in null_rates.items():
            detector.append(
                {
                    "seed": seed,
                    "variant": name,
                    "h_sigma": cal["threshold_scale"],
                    "holdout_sim_alarms_per_year": cal["holdout_sim_alarms_per_year"],
                    "historical_train_alarms_per_year": run["historical_train_alarms"][name]["alarms_per_year"],
                    "test_alarms_per_year": run["results"][name]["detector"]["alarms_per_year"],
                }
            )
        for row in (run.get("synthetic_diagnostic") or {}).get("results", []):
            synthetic.append({"seed": seed, **row})
    return pd.DataFrame(portfolio), pd.DataFrame(detector), pd.DataFrame(synthetic)


def mean_sd(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, sort=False)[values]
    mean, sd = grouped.mean(), grouped.std(ddof=1)
    out = pd.DataFrame(index=mean.index)
    for column in values:
        out[column] = [
            f"{m:.3f} ± {s:.3f}" if pd.notna(s) else f"{m:.3f}" for m, s in zip(mean[column], sd[column], strict=True)
        ]
    out["n_seeds"] = grouped.size()
    return out.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("patterns", nargs="+")
    parser.add_argument("--ard-mode", default="paper")
    parser.add_argument(
        "--tag",
        default="",
        help="Only runs with this experiment tag ('' = untagged; use --any-tag for all tags)",
    )
    parser.add_argument(
        "--any-tag",
        action="store_true",
        help="Ignore tag differences (keep latest paper/repo run per seed)",
    )
    parser.add_argument("--out", type=Path, default=Path("results/tables"))
    args = parser.parse_args()

    runs = load_runs(args.patterns, args.ard_mode, None if args.any_tag else args.tag)
    if not runs:
        raise SystemExit("No matching runs.")
    portfolio, detector, synthetic = run_rows(runs)
    args.out.mkdir(parents=True, exist_ok=True)
    portfolio.to_csv(args.out / "portfolio_per_seed.csv", index=False)
    detector.to_csv(args.out / "detector_per_seed.csv", index=False)
    synthetic.to_csv(args.out / "synthetic_per_seed.csv", index=False)

    tables = {
        "portfolio": mean_sd(
            portfolio, ["variant"], ["CR", "SR", "MDD", "adaptations_per_year", "test_alarms_per_year"]
        ),
        "detector": mean_sd(
            detector,
            ["variant"],
            ["h_sigma", "holdout_sim_alarms_per_year", "historical_train_alarms_per_year", "test_alarms_per_year"],
        ),
        "synthetic": mean_sd(
            synthetic, ["label", "shift_sigma"], ["detection_rate", "median_delay_days", "pre_shift_false_alarms_per_year"]
        ),
    }
    lines = [f"Seeds: {[r['seed'] for r in runs]}; commits: {sorted({r['git_commit'][:7] for r in runs})}", ""]
    for title, table in tables.items():
        lines += [f"## {title}", "", table.to_markdown(index=False), ""]
    (args.out / "tables.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
