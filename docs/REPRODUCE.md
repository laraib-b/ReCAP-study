# How each ablation was run

All study runs use `study/run_study.py` with `--ard-mode paper`, `--disable-local-cache`, and `--no-adapt-control`.  
Canonical outputs are under `results/` (see `results/README.md`). You do **not** need the removed `first_run_*` notebooks.

## Front door for cloud re-runs

| Tool | File | Use |
|---|---|---|
| **molab (preferred)** | `notebooks/study_run_molab.py` | Edit the `JOBS` cell, Run all |
| Colab (optional) | `notebooks/study_run_colab.ipynb` | Same study CLI under Colab |
| Local CLI | commands below | Micro / analysis on a laptop |

Device note: force **CPU** for study jobs (`--device cpu`). SB3 MLP-PPO was slower on CUDA in our molab runs.

## Exact recipes used for the pilot

### A. Micro baseline (finetune = 64, default profile)

Laptop. Seeds 0–2. Aggregated in `results/tables/` (also mirrored as `results/tables_ft64/`).

```bash
python study/run_study.py --profile micro --dataset DOW30 --device cpu --seed 0 --disable-local-cache --no-adapt-control --ard-mode paper
python study/run_study.py --profile micro --dataset DOW30 --device cpu --seed 1 --disable-local-cache --no-adapt-control --ard-mode paper
python study/run_study.py --profile micro --dataset DOW30 --device cpu --seed 2 --disable-local-cache --no-adapt-control --ard-mode paper
```

### B. Micro ablation — more adaptation training (finetune = 256)

Laptop. Tagged runs; tables in `results/tables_ft256/`.

```bash
python study/run_study.py --profile micro --dataset DOW30 --device cpu --seed 0 --disable-local-cache --no-adapt-control --ard-mode paper --finetune-timesteps 256 --tag micro_ft256_seed0
# … seeds 1 and 2 with matching --tag …
```

### C. Micro ablation — even more adaptation training (finetune = 1024)

Laptop. Tables in `results/tables_ft1024/`.

```bash
python study/run_study.py --profile micro --dataset DOW30 --device cpu --seed 0 --disable-local-cache --no-adapt-control --ard-mode paper --finetune-timesteps 1024 --tag micro_ft1024_seed0
# … seeds 1 and 2 …
```

### D. Reduced DOW30 (primary pilot)

Molab via `notebooks/study_run_molab.py`, one seed at a time, CPU.  
28 tickers (WBA/DOW dropped). Tables in `results/tables_reduced/`.

```bash
python study/run_study.py --profile reduced --dataset DOW30 --device cpu --seed 0 --disable-local-cache --no-adapt-control --ard-mode paper --tag reduced_seed0
python study/run_study.py --profile reduced --dataset DOW30 --device cpu --seed 1 --disable-local-cache --no-adapt-control --ard-mode paper --tag reduced_seed1
python study/run_study.py --profile reduced --dataset DOW30 --device cpu --seed 2 --disable-local-cache --no-adapt-control --ard-mode paper --tag reduced_seed2
```

Equivalent molab `JOBS` entries (see notebook comments):

```python
("reduced", 0, None, "reduced_seed0")
("reduced", 1, None, "reduced_seed1")
("reduced", 2, None, "reduced_seed2")
```

## Aggregate after new runs

```bash
python study/aggregate.py "results/runs/study_reduced_seed*_2026*.json" --ard-mode paper --any-tag --out results/tables_reduced
python study/aggregate.py "results/runs/study_micro_dow30_seed*_20261113T08*.json" --ard-mode paper --out results/tables
python study/aggregate.py "results/runs/study_micro_ft256_*.json" --ard-mode paper --any-tag --out results/tables_ft256
python study/aggregate.py "results/runs/study_micro_ft1024_*.json" --ard-mode paper --any-tag --out results/tables_ft1024
```

## What we will not claim

- Exact paper Table 2 / NAS100 replication (pilot compute budgets).
- That historical market alarms are labelled false alarms.
- That synthetic-shift diagnostics are real market regimes.

See `docs/DEVIATIONS.md`.
