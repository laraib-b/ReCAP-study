# Results index

Canonical artifacts for the false-alarm calibration study. Prefer these paths over `outputs/` (gitignored local scratch).

## What to open first

| Artifact | Use for |
|---|---|
| `tables_reduced/tables.md` | **Main pilot tables** (reduced DOW30, seeds 0–2) |
| `tables_reduced/portfolio_per_seed.csv` | Plot CR / SR / MDD by seed |
| `tables_reduced/detector_per_seed.csv` | Calibration + alarm rates |
| `tables_reduced/synthetic_per_seed.csv` | Injected-shift detector diagnostic |
| `tables/`, `tables_ft256/`, `tables_ft1024/` | Micro profile + fine-tune ablation aggregates |
| `runs/study_reduced_seed{0,1,2}_*_summary.json` | One-page human summaries |
| `runs/study_reduced_seed{0,1,2}_*.json` | **Full** reduced run records (metrics, alarms, segments) |
| `runs/study_micro_*.json` | Full micro / ft256 / ft1024 run records |

## Reduced DOW30 (primary)

- Profile: `reduced`, 28 tickers (WBA/DOW dropped), 2008-05-01 → 2025-04-29
- Protocol: `ard_mode=paper`, sim-calibrated \(h\), causal eval, library fit once at default \(h=2.5\sigma\)
- Seeds: 0 (`…T123541Z`), 1 (`…T131334Z`), 2 (`…T144354Z`)
- Discard: `study_reduced_seed0_20261113T095839Z_FAILED_summary.json` (pre–data-fix wipeout)

Mean CR: **no_adapt ≈ 0.65**; all adapt-on-alarm variants worse on average. Do **not** claim paper Table 2 replication.

## Micro + fine-tune ablation

| Folder | Meaning |
|---|---|
| `tables/` (= ft64) | micro baseline, `finetune_timesteps=64` |
| `tables_ft256/` | adaptation-budget ablation |
| `tables_ft1024/` | larger adaptation budget |

Same story directionally: no-adapt is strong; more fine-tune helps the loud detector somewhat on micro.

## Regenerating tables

```powershell
.\.venv\Scripts\python.exe study/aggregate.py "results/runs/study_reduced_seed*_2026*.json" --ard-mode paper --any-tag --out results/tables_reduced
.\.venv\Scripts\python.exe study/aggregate.py "results/runs/study_micro_dow30_seed*_20261113T08*.json" --ard-mode paper --out results/tables
.\.venv\Scripts\python.exe study/aggregate.py "results/runs/study_micro_ft256_*.json" --ard-mode paper --any-tag --out results/tables_ft256
.\.venv\Scripts\python.exe study/aggregate.py "results/runs/study_micro_ft1024_*.json" --ard-mode paper --any-tag --out results/tables_ft1024
```

Daily equity paths from runs live under local `outputs/study/daily_*` when a machine re-runs the study; they are not required to read the portfolio metrics already stored in each run JSON.
