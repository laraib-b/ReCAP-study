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

1. Download the **latest** `notebooks/first_run_colab.ipynb` from GitHub (Colab does not auto-update an already-open notebook when you push).
2. Upload/open it in Colab → **Connect** → **Run all**.
3. Section 3 = synthetic smoke. Section 4 = reduced DOW30 pilot (slower, not paper-scale).
4. Repo URL in the notebook should be `https://github.com/laraib-b/ReCAP-study.git`.

CLI equivalent after clone:

```bash
python study/smoke_run.py
python study/reduced_pilot.py --dataset DOW30 --disable-local-cache
```

## Portability checklist

- [ ] `requirements.txt` (and later a lockfile) in git
- [ ] Config JSON/YAML for every experiment in `study/configs/`
- [ ] Data fetch documented (not assuming a local path)
- [ ] Outputs written under `outputs/` (gitignored except small examples)
- [ ] Hardware and seed count recorded in `docs/BASELINE.md`
