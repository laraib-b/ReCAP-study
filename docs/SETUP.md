# Setup guide

## Recommendation

| Goal | Path |
|---|---|
| Prove the pipeline works | Local smoke: `python study/smoke_run.py` |
| Reproduce / extend the study | CLI in `docs/REPRODUCE.md`, or molab notebook |
| Write-up from existing numbers | `results/` — no re-run required |
| Sync an old clone | `docs/PARTNERS.md` |

## Local environment (Windows)

1. Install **Python 3.11** (recommended).
2. From the repo root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install pytest tabulate
```

Study path does not require FinRL. Data: place caches under `datasets/` or allow yfinance on first run (`--disable-local-cache` forces download).

Micro seed on an 8 GB laptop: ~8–15 min.

## molab (preferred cloud runner)

1. Open [https://molab.new](https://molab.new).
2. Upload `notebooks/study_run_molab.py`.
3. Edit the `JOBS` cell (recipes in `docs/REPRODUCE.md`). Empty `JOBS` clones the repo only.
4. Run all. Leave the tab open.
5. Download JSON via the notebook button (or optional GitHub token push).

Force CPU inside the notebook — SB3 MLP-PPO was slower on CUDA in our runs.

## Colab (optional)

Use `notebooks/study_run_colab.ipynb` only if you prefer Colab. There is **no** `first_run_colab` / `first_run_kaggle` notebook anymore.

## Portability checklist

- [x] `requirements.txt` in git
- [x] Study runner + reproduce docs
- [x] Canonical results under `results/`
- [x] Protocol limits in `docs/DEVIATIONS.md`
