# Partner sync guide

If you already cloned an older copy of this repo (and maybe started docs from it), use this checklist so your local tree matches the **submission-ready** remote.

## 1. Pull the current tip

```bash
git fetch origin
git checkout main
git pull origin main
```

`main` and `study-sunday` are kept at the same tip. Prefer **`main`**.

Repo: https://github.com/laraib-b/ReCAP-study

## 2. What changed that may confuse older clones

| Old / confusing | Now |
|---|---|
| `notebooks/first_run_colab.ipynb` | **Removed** — unused upstream demo |
| `notebooks/first_run_kaggle.ipynb` | **Removed** — unused upstream demo |
| Only stub reduced summaries under `results/runs/` | **Full** reduced seed 0–2 JSONs + `results/tables_reduced/` |
| Molab notebook hardcoded to one leftover seed | Single front-door runner with documented JOB recipes |

If your notes or partner docs still mention `first_run_*`, point them at:

- **Results / write-up:** `results/README.md`, `docs/STATUS.md`, `docs/DEVIATIONS.md`
- **How experiments were run:** `docs/REPRODUCE.md`
- **Cloud re-run:** `notebooks/study_run_molab.py` (primary), `notebooks/study_run_colab.ipynb` (optional)

## 3. What you do *not* need to re-run

Canonical pilot results are already in git under `results/`. Re-running micro/reduced is only for reproduction checks or new seeds — not required to draft the paper from the existing tables.

## 4. Suggested local layout for your write-up

Keep analysis docs **outside** or clearly separate from upstream code if you prefer, but when syncing code/results:

```text
ReCAP-study/          # this git repo (pull often)
  results/            # use these numbers
  docs/               # protocol + partner notes
  study/              # study code
your-paper-notes/     # optional: your shared Google Docs / Overleaf exports
```

## 5. Quick sanity after pull

```bash
ls results/tables_reduced
ls notebooks   # expect study_run_molab.py + study_run_colab.ipynb only
python -m pytest tests/test_smoke.py tests/test_study_calibration.py tests/test_marimo_notebooks.py -q
```
