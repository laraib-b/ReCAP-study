# Study package — False-Alarm-Calibrated Regime Detection for ReCAP

This folder extends the upstream minimal ReCAP implementation. It does **not** replace `recap/`.

## Goals

- Causal streaming evaluation (no `max(baseline, adapted)` look-ahead)
- Detector / adaptation instrumentation
- Train-only CUSUM threshold calibration to predeclared false-alarm budgets
- Synthetic-shift detector diagnostics
- Reproducible configs and saved outputs for the research write-up

Saved results for partners live under `results/` (see `results/README.md`), not under gitignored `outputs/`.

## Upstream pin

- Repo: https://github.com/Dumail/ReCAP
- Commit used as base: `68143bb`

## Run options

| Where | Best for | Notes |
|---|---|---|
| Local CLI | Micro / analysis | Python **3.10 or 3.11**; see `docs/REPRODUCE.md` |
| **molab** | Reduced / cloud reproduce | `notebooks/study_run_molab.py` — edit `JOBS` |
| Colab | Optional | `notebooks/study_run_colab.ipynb` only |

## Quick local smoke (after venv + deps)

```bash
python -m pytest tests/test_smoke.py tests/test_study_calibration.py -q
python study/smoke_run.py
```

## Study run (causal + FA calibration)

```bash
python study/run_study.py --profile micro --dataset DOW30 --seed 0 --disable-local-cache --no-adapt-control --ard-mode paper
python study/aggregate.py "results/runs/study_*.json" --ard-mode paper --any-tag --out results/tables
```

Full recipe list: `docs/REPRODUCE.md`. Protocol claims/limits: `docs/DEVIATIONS.md`. Partner sync: `docs/PARTNERS.md`.
