# Baseline / first-run log

## Environment (this device)

- OS: Windows 10
- CPU: Intel Core i7-8650U (no NVIDIA GPU)
- RAM: ~8 GB
- Python: 3.11.9 via `D:\quant\venvs\recap-study`
- Upstream commit: `68143bb`
- Core deps installed: numpy, pandas, torch, gymnasium, stable-baselines3, yfinance, pytest
- `finrl` not required for synthetic smoke; still needed for ticker lists in `main.py` / real-market CLI

## Patch applied for NumPy 2.x

`recap/data.py` `_calculate_turbulence`: copy covariance before in-place add (read-only array under NumPy 2).

## First successful run

- Command: `python study/smoke_run.py`
- Mode: synthetic 3-asset / ~80 business days, tiny PPO budgets
- Output: `outputs/smoke/smoke_metrics.json`
- Result: completed with CR/SR/MDD emitted (smoke metrics are not research findings)

## Next

1. Install `finrl` (or vendor tickers) for DOW30/NAS100 CLI
2. Colab notebook: `notebooks/first_run_colab.ipynb` for heavier runs
3. Causal eval + calibration under `study/`
