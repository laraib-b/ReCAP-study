"""Build Phase 3 comparison tables from verified experimental outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_IMPL = Path(__file__).resolve().parents[1]
import sys
if str(_IMPL) not in sys.path:
    sys.path.insert(0, str(_IMPL))
from paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
OUT = ROOT / "results" / "tables_phase3"


def fmt_mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    port = pd.read_csv(ROOT / "results" / "tables_reduced" / "portfolio_per_seed.csv")
    det = pd.read_csv(ROOT / "results" / "tables_reduced" / "detector_per_seed.csv")

    rows = []
    for variant, g in port.groupby("variant"):
        d = det[det["variant"] == variant] if variant in set(det["variant"]) else None
        row = {
            "variant": variant,
            "CR": fmt_mean_sd(g["CR"].mean(), g["CR"].std(ddof=1)),
            "SR": fmt_mean_sd(g["SR"].mean(), g["SR"].std(ddof=1)),
            "MDD": fmt_mean_sd(g["MDD"].mean(), g["MDD"].std(ddof=1)),
            "adaptations_per_year": fmt_mean_sd(
                g["adaptations_per_year"].mean(), g["adaptations_per_year"].std(ddof=1)
            ),
            "test_alarms_per_year": fmt_mean_sd(
                g["test_alarms_per_year"].mean(), g["test_alarms_per_year"].std(ddof=1)
            ),
            "n_seeds": len(g),
        }
        if d is not None and not d.empty:
            row["h_sigma"] = fmt_mean_sd(d["h_sigma"].mean(), d["h_sigma"].std(ddof=1))
            row["holdout_sim_alarms_per_year"] = fmt_mean_sd(
                d["holdout_sim_alarms_per_year"].mean(),
                d["holdout_sim_alarms_per_year"].std(ddof=1),
            )
        rows.append(row)
    main_tbl = pd.DataFrame(rows)
    main_tbl.to_csv(OUT / "table_main_reduced_comparison.csv", index=False)

    # Merge classical baselines if present
    base_csv = OUT / "market_baselines.csv"
    combined_rows = []
    for _, r in main_tbl.iterrows():
        combined_rows.append(
            {
                "method": r["variant"],
                "family": "study_RL",
                "CR": r["CR"],
                "SR": r["SR"],
                "MDD": r["MDD"],
                "notes": "causal FA-calibrated ReCAP pilot",
            }
        )
    if base_csv.exists():
        base = pd.read_csv(base_csv)
        for _, r in base.iterrows():
            combined_rows.append(
                {
                    "method": r["variant"],
                    "family": "classical",
                    "CR": f"{r['CR']:.3f}",
                    "SR": f"{r['SR']:.3f}",
                    "MDD": f"{r['MDD']:.3f}",
                    "notes": "same DOW28 test window; non-RL",
                }
            )
    combined = pd.DataFrame(combined_rows)
    combined.to_csv(OUT / "table_combined_baselines.csv", index=False)

    meta = {
        "primary_experiment": "reduced DOW30, 3 seeds, paper ARD, causal eval",
        "sources": {
            "portfolio": "results/tables_reduced/portfolio_per_seed.csv",
            "detector": "results/tables_reduced/detector_per_seed.csv",
            "market_baselines": "results/tables_phase3/market_baselines.csv",
        },
        "warning": "Pilot-scale PPO budgets; not paper Table 2 / NAS100 replication.",
    }
    (OUT / "table_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    md = [
        "# Phase 3 comparison tables",
        "",
        "## Table 1. Reduced DOW30 study variants",
        "",
        main_tbl.to_string(index=False),
        "",
        "## Table 2. Combined with classical baselines",
        "",
        combined.to_string(index=False),
        "",
    ]
    (OUT / "tables.md").write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote tables under {OUT}")


if __name__ == "__main__":
    main()
