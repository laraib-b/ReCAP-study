"""Consolidate ablation tables already produced by the study runners.

Ablations covered (from Milestone 2 / implemented protocol):
1. Primary: FA-budget threshold variants (default vs 0.5/1/2 alarms/yr)
2. Secondary: adaptation fine-tune budget (64 / 256 / 1024 steps) on micro DOW30
3. Diagnostic: paper-mode vs repo-mode ARD calibration feasibility (documented)

This script only aggregates verified CSVs; it does not invent new runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import sys
_IMPL = Path(__file__).resolve().parents[2]
if str(_IMPL) not in sys.path:
    sys.path.insert(0, str(_IMPL))
from paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / "results" / "tables_phase3"


def mean_sd(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    for variant, g in df.groupby("variant"):
        row = {"variant": variant, "n_seeds": len(g)}
        for c in cols:
            row[f"{c}_mean"] = float(g[c].mean())
            row[f"{c}_sd"] = float(g[c].std(ddof=1)) if len(g) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # Primary FA-budget ablation = reduced portfolio/detector tables
    reduced_p = pd.read_csv(ROOT / "results" / "tables_reduced" / "portfolio_per_seed.csv")
    reduced_d = pd.read_csv(ROOT / "results" / "tables_reduced" / "detector_per_seed.csv")
    primary = mean_sd(reduced_p, ["CR", "SR", "MDD", "adaptations_per_year", "test_alarms_per_year"])
    primary.to_csv(OUT / "ablation_primary_fa_budget.csv", index=False)

    # Fine-tune ablation
    frames = []
    for steps, folder in [(64, "tables"), (256, "tables_ft256"), (1024, "tables_ft1024")]:
        df = pd.read_csv(ROOT / "results" / folder / "portfolio_per_seed.csv")
        m = mean_sd(df, ["CR", "SR", "MDD"])
        m["finetune_timesteps"] = steps
        frames.append(m)
    ft = pd.concat(frames, ignore_index=True)
    ft.to_csv(OUT / "ablation_finetune_budget.csv", index=False)

    # Detector delay ablation from synthetic table
    syn = pd.read_csv(ROOT / "results" / "tables_reduced" / "synthetic_per_seed.csv")
    syn_m = (
        syn.groupby(["label", "shift_sigma"])
        .agg(
            detection_rate_mean=("detection_rate", "mean"),
            detection_rate_sd=("detection_rate", "std"),
            median_delay_mean=("median_delay_days", "mean"),
            median_delay_sd=("median_delay_days", "std"),
            pre_shift_fa_mean=("pre_shift_false_alarms_per_year", "mean"),
        )
        .reset_index()
        .rename(columns={"label": "variant"})
    )
    syn_m.to_csv(OUT / "ablation_detection_delay.csv", index=False)

    # Repo-mode calibration feasibility note from early summary
    repo_summary = ROOT / "results" / "runs" / "study_micro_dow30_seed0_20261112T185311Z_summary.json"
    repo_note = {
        "ablation": "ard_mode_repo_vs_paper",
        "evidence_file": str(repo_summary.relative_to(ROOT)).replace("\\", "/"),
        "finding": (
            "Under released repo-mode ARD (two-sided, fixed mu0), calibration could not "
            "reach 0.5–2 alarms/yr targets (budget_0.5 marked infeasible at h≈128σ). "
            "Paper-mode ARD enables feasible calibration for budgets 1 and 2; budget 0.5 "
            "often hits the h=64σ ceiling on reduced runs."
        ),
        "status": "documented_from_existing_run",
    }
    if repo_summary.exists():
        payload = json.loads(repo_summary.read_text(encoding="utf-8"))
        repo_note["calibrations"] = payload.get("calibrations")

    (OUT / "ablation_ard_mode_note.json").write_text(json.dumps(repo_note, indent=2), encoding="utf-8")

    def _md(df: pd.DataFrame) -> str:
        try:
            return df.to_markdown(index=False)
        except Exception:
            return "```\n" + df.to_string(index=False) + "\n```"

    # Combined markdown for report ingestion
    lines = [
        "# Ablation summary tables",
        "",
        "## Primary: false-alarm budget (reduced DOW30)",
        "",
        _md(primary),
        "",
        "## Secondary: fine-tune timesteps (micro DOW30)",
        "",
        _md(ft),
        "",
        "## Detector delay by shift magnitude (reduced)",
        "",
        _md(syn_m),
        "",
        "## ARD mode note",
        "",
        repo_note["finding"],
        "",
    ]
    (OUT / "ablation_tables.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote ablation tables under {OUT}")


if __name__ == "__main__":
    main()
