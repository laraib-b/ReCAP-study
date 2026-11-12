# Study package — False-Alarm-Calibrated Regime Detection for ReCAP

This folder extends the upstream minimal ReCAP implementation. It does **not** replace `recap/`.

## Goals

- Causal streaming evaluation (no `max(baseline, adapted)` look-ahead)
- Detector / adaptation instrumentation
- Train-only CUSUM threshold calibration to predeclared false-alarm budgets
- Synthetic-shift detector diagnostics
- Reproducible configs and saved outputs for the research write-up

## Upstream pin

- Repo: https://github.com/Dumail/ReCAP
- Commit used as base: `68143bb`

## Run options

| Where | Best for | Notes |
|---|---|---|
| Cursor / local CLI | Smoke tests, calibration, analysis | Prefer Python **3.10 or 3.11** (not 3.13) |
| Local Jupyter | Interactive debugging | Same venv as CLI |
| Google Colab | Heavier NAS100 / multi-seed runs | Use `notebooks/first_run_colab.ipynb` |

## Quick local smoke (after venv + deps)

```bash
python -m pytest tests/test_smoke.py tests/test_study_calibration.py -q
python study/smoke_run.py
```

## Study run (causal + FA calibration)

```bash
python study/run_study.py --profile micro --dataset DOW30 --seed 0 --disable-local-cache --no-adapt-control
python study/aggregate.py outputs/study/study_....json
```

Colab notebook: `notebooks/study_run_colab.ipynb`

See `docs/DEVIATIONS.md` for protocol claims and limits.
