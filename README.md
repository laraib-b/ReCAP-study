# ReCAP study fork — false-alarm–calibrated regime detection

Fork of the minimal [ReCAP](https://github.com/Dumail/ReCAP) implementation with a **causal, false-alarm–calibrated** study package.

**Question we answer:** keep ReCAP fixed; only change how often the ARD detector alarms (by calibrating threshold \(h\)); does a lower alarm rate help portfolio results?

**Short answer (pilot):** calibration works on no-change simulations; on reduced DOW30, **no adaptation beats every adapt-on-alarm setting**. See `docs/STATUS.md` and `results/`.

## If you already cloned an older copy

Read **`docs/PARTNERS.md`** first (removed notebooks, where results live, how to pull).

## Repo map (submission surface)

| Path | Role |
|---|---|
| `recap/` | Upstream ReCAP core + small study fixes (e.g. drop unusable tickers) |
| `study/` | Study protocol: paper-mode ARD, FA calibration, causal eval, diagnostics, aggregate |
| `results/` | **Canonical run JSONs + mean±sd tables** for write-up / plots |
| `docs/` | Protocol, reproduce recipes, partner sync, status |
| `notebooks/study_run_molab.py` | **Primary** cloud runner (all ablation recipes via `JOBS`) |
| `notebooks/study_run_colab.ipynb` | Optional Colab study notebook |
| `tests/` | Smoke + calibration + marimo static checks |
| `main.py` | Upstream entrypoint |

Removed on purpose: `notebooks/first_run_colab.ipynb`, `notebooks/first_run_kaggle.ipynb` (unused demos).

Not a paper Table 2 replication: training budgets are **pilot-scale** (`docs/DEVIATIONS.md`).

## Quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install pytest tabulate
.\.venv\Scripts\python.exe -m pytest tests/test_smoke.py tests/test_study_calibration.py tests/test_marimo_notebooks.py -q
```

## How experiments were run

See **`docs/REPRODUCE.md`** for the exact micro / ft256 / ft1024 / reduced commands.

Cloud re-run: upload `notebooks/study_run_molab.py` → edit `JOBS` → Run all.

## Results to use

Start here: **`results/README.md`**.

Headline reduced DOW30 (3 seeds, causal, paper ARD):

| Variant | mean CR |
|---|---:|
| no_adapt_default | **0.65** |
| default_2.5 | −0.01 |
| budget_0.5 | −0.34 |
| budget_1 | −0.20 |
| budget_2 | −0.20 |

## Branches

- `main` — partner-facing default (use this)
- `study-sunday` — kept equal to `main` for older molab clones

Upstream pin: Dumail/ReCAP (see `study/README.md`).
