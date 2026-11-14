# ReCAP study fork — false-alarm–calibrated regime detection

Fork of the minimal [ReCAP](https://github.com/Dumail/ReCAP) implementation with a **causal, false-alarm–calibrated** study package.

**Question we answer:** keep ReCAP fixed; only change how often the ARD detector alarms (by calibrating threshold \(h\)); does a lower alarm rate help portfolio results?

**Short answer (pilot):** calibration works on no-change simulations; on reduced DOW30, **no adaptation beats every adapt-on-alarm setting**. See `docs/STATUS.md` and `results/`.

## Repo map (what partners need)

| Path | Role |
|---|---|
| `recap/` | Upstream ReCAP core (PPO, ARD, gate, env) + small study fixes (e.g. drop unusable tickers) |
| `study/` | Study protocol: paper-mode ARD streaming, FA calibration, causal eval, diagnostics, aggregate |
| `results/` | **Canonical run JSONs + mean±sd tables** for write-up / plots |
| `docs/` | Protocol deviations, setup, status |
| `notebooks/study_run_molab.py` | Molab/marimo runner for reduced seeds |
| `notebooks/study_run_colab.ipynb` | Colab study notebook |
| `tests/` | Smoke + calibration unit tests |
| `main.py` | Upstream entrypoint (unchanged workflow) |

Not for submission as “paper replication”: training budgets here are **pilot-scale** (below paper Table 2). Label results as such (`docs/DEVIATIONS.md`).

## Quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install pytest tabulate
.\.venv\Scripts\python.exe -m pytest tests/test_smoke.py tests/test_study_calibration.py -q
```

Micro study (laptop, ~8–15 min/seed):

```powershell
.\.venv\Scripts\python.exe study/run_study.py --profile micro --dataset DOW30 --device cpu --seed 0 --disable-local-cache --no-adapt-control --ard-mode paper
```

Reduced DOW30 (prefer molab/Colab; ~15–60+ min/seed):

```powershell
.\.venv\Scripts\python.exe study/run_study.py --profile reduced --dataset DOW30 --device cpu --seed 0 --disable-local-cache --no-adapt-control --ard-mode paper --tag reduced_seed0
```

Aggregate:

```powershell
.\.venv\Scripts\python.exe study/aggregate.py "results/runs/study_reduced_seed*_2026*.json" --ard-mode paper --any-tag --out results/tables_reduced
```

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

- `main` — partner-facing study + results (default)
- `study-sunday` — same tip used during molab runs

Upstream pin: Dumail/ReCAP (see `study/README.md`).
