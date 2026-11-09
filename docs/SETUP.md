# Setup and first-run guide

## Recommendation on this hardware

This machine has ~8 GB RAM, Intel UHD only (no NVIDIA), and Python 3.13 by default.

| Goal | Recommended path |
|---|---|
| Prove the pipeline works | Local **smoke** (`study/smoke_run.py`) or Colab notebook |
| Reduced NAS100 pilot | Prefer **Google Colab** (more RAM) or a stronger PC |
| Analysis / calibration / plots | Local Cursor or Jupyter is fine |

Code is **not** tied to `D:\quant`. Clone the GitHub repo anywhere, recreate the venv, download data once.

## Local environment (Windows)

1. Install **Python 3.11** (recommended) from python.org or:
   `winget install Python.Python.3.11`
2. From the repo root:

```powershell
py -3.11 -m venv D:\quant\venvs\recap-study
D:\quant\venvs\recap-study\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
pip install pytest jupyter
```

3. Data: either place FinRL-style caches under `datasets/` or allow yfinance download on first run.

## Google Colab

Open `notebooks/first_run_colab.ipynb`, run all cells. It clones the upstream repo (or your fork), installs deps, and runs a **tiny** ReCAP pass so you can confirm the stack before scaling steps.

## Portability checklist

- [ ] `requirements.txt` (and later a lockfile) in git
- [ ] Config JSON/YAML for every experiment in `study/configs/`
- [ ] Data fetch documented (not assuming a local path)
- [ ] Outputs written under `outputs/` (gitignored except small examples)
- [ ] Hardware and seed count recorded in `docs/BASELINE.md`
