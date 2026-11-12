# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "marimo",
#   "numpy<2.5",
#   "pandas==2.2.3",
#   "torch",
#   "gymnasium>=0.29",
#   "stable-baselines3>=2.0",
#   "yfinance>=0.2",
#   "requests>=2.31",
# ]
# ///

"""ReCAP false-alarm study runner for molab (marimo).

Open on https://molab.new, upload this file, optionally enable GPU, then run.
"""

import marimo

__generated_with = "0.14.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import torch

    return Path, json, mo, os, pd, subprocess, sys, torch


@app.cell
def _(mo):
    mo.md(
        r"""
        # ReCAP study on molab

        Protocol-correct pilot:
        - fit once at default h=2.5σ
        - calibrate train-only FA budgets 0.5 / 1 / 2 alarms/year
        - causal eval (no look-ahead max selection)
        - synthetic-shift diagnostic

        Uses **micro** profile first.
        """
    )
    return


@app.cell
def _():
    REPO_URL = "https://github.com/laraib-b/ReCAP-study.git"
    REPO_DIR = "/tmp/ReCAP-study"
    BRANCH = "study-sunday"
    return BRANCH, REPO_DIR, REPO_URL


@app.cell
def _(BRANCH, Path, REPO_DIR, REPO_URL, mo, os, subprocess, sys):
    os.environ["YF_DISABLE_CURL_CFFI"] = "1"
    repo = Path(REPO_DIR)
    if not repo.exists():
        subprocess.check_call(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO_DIR])
    else:
        subprocess.check_call(["git", "-C", REPO_DIR, "fetch", "origin", BRANCH])
        subprocess.check_call(["git", "-C", REPO_DIR, "reset", "--hard", f"origin/{BRANCH}"])

    os.chdir(REPO_DIR)
    if REPO_DIR not in sys.path:
        sys.path.insert(0, REPO_DIR)

    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    mo.md(f"**cwd:** `{os.getcwd()}`  \n**commit:** `{commit}`")
    return (commit,)


@app.cell
def _(mo, torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mo.md(f"**Device:** `{device}` | cuda available = `{torch.cuda.is_available()}`")
    return (device,)


@app.cell
def _(device, mo, subprocess):
    cmd = [
        "python",
        "study/run_study.py",
        "--profile",
        "micro",
        "--dataset",
        "DOW30",
        "--device",
        device,
        "--seed",
        "0",
        "--disable-local-cache",
        "--no-adapt-control",
    ]
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
    print(completed.stdout[-4000:] if completed.stdout else "")
    if completed.returncode != 0:
        print(completed.stderr[-4000:] if completed.stderr else "")
        raise RuntimeError(f"study run failed with code {completed.returncode}")
    mo.md("Study run finished.")
    return (completed,)


@app.cell
def _(Path, json, mo):
    study_files = sorted(Path("outputs/study").glob("study_*.json"))
    assert study_files, "No study JSON found under outputs/study"
    latest = study_files[-1]
    payload = json.loads(latest.read_text(encoding="utf-8"))

    rows = []
    for name, result in payload.get("results", {}).items():
        m = result["metrics"]
        d = result["detector"]
        rows.append(
            {
                "variant": name,
                "CR": m["CR"],
                "SR": m["SR"],
                "MDD": m["MDD"],
                "alarms/yr": d["alarms_per_year"],
                "adapt/yr": result["adaptations_per_year"],
            }
        )

    mo.md(f"**Latest artifact:** `{latest}`")
    return latest, payload, rows, study_files


@app.cell
def _(mo, pd, rows):
    results_df = pd.DataFrame(rows)
    mo.ui.table(results_df)
    return (results_df,)


@app.cell
def _(latest, mo):
    mo.md(
        f"""
        ## Next
        1. Download `{latest}` from the molab file browser.
        2. Optional: change `--seed` to 1 and 2 for a 3-seed pilot.
        3. If micro succeeds and RAM is fine, try `--profile reduced`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
