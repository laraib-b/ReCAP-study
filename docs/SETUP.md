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

### Lighter venv for the study (`micro` profile, no FinRL)

The study path does not import FinRL. A repo-local `.venv` (gitignored) is enough:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install "pandas==2.2.3" "gymnasium>=0.29" "stable-baselines3>=2.0" "yfinance>=0.2" "requests>=2.31" pytest
.\.venv\Scripts\python.exe study/run_study.py --profile micro --dataset DOW30 --device cpu --seed 0 --disable-local-cache --no-adapt-control
```

On this 8 GB laptop one micro seed takes ~8.5 min and reproduces the molab seed-0 result (same calibrated `h`, portfolio metrics equal to ~4 decimals).

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

## molab (recommended over free Colab for this project)

[molab](https://docs.marimo.io/guides/molab/) is marimo’s free hosted notebook cloud (~**4 CPU / 32 GB RAM**, optional GPU, up to 12h sessions). That RAM headroom is much better for ReCAP than free Colab.

1. Open [https://molab.new](https://molab.new) (or [https://molab.marimo.io](https://molab.marimo.io)).
2. Upload `notebooks/study_run_molab.py` (marimo format, not `.ipynb`).
3. Optionally attach a GPU from the notebook specs button.
4. Run all cells. It clones branch `study-sunday` and runs `study/run_study.py --profile micro`.
5. Download `outputs/study/*.json` from the file sidebar.

Note: molab uses **marimo** notebooks (`.py`), not Jupyter. Do **not** use the WebAssembly playground for this ML workload — use a full molab server notebook.

## Portability checklist

- [ ] `requirements.txt` (and later a lockfile) in git
- [ ] Config JSON/YAML for every experiment in `study/configs/`
- [ ] Data fetch documented (not assuming a local path)
- [ ] Outputs written under `outputs/` (gitignored except small examples)
- [ ] Hardware and seed count recorded in `docs/BASELINE.md`
