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
    # Prefer a download button. Molab FILES often does not show runtime-created folders.
    study_dir = Path("/tmp/ReCAP-study/outputs/study")
    if not study_dir.exists():
        study_dir = Path("outputs/study")
    study_files = sorted(study_dir.glob("study_*.json"))
    assert study_files, f"No study JSON found under {study_dir}"
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

    download = mo.download(
        data=latest.read_bytes(),
        filename=latest.name,
        mimetype="application/json",
        label=f"Download {latest.name}",
    )
    mo.md(f"**Artifact on disk:** `{latest}`")
    return download, latest, payload, rows, study_files


@app.cell
def _(download, mo, pd, rows):
    results_df = pd.DataFrame(rows)
    mo.vstack([download, mo.ui.table(results_df)])
    return (results_df,)


@app.cell
def _(mo):
    github_token = mo.ui.text(
        label="GitHub token (repo write). Leave empty to skip push.",
        kind="password",
    )
    github_token
    return (github_token,)


@app.cell
def _(Path, github_token, latest, mo, subprocess):
    if not github_token.value:
        mo.md("Skipping GitHub push (no token). Use the Download button above instead.")
        push_result = None
    else:
        cmd = [
            "python",
            "study/push_artifact.py",
            str(latest),
            "--token",
            github_token.value,
            "--branch",
            "study-sunday",
        ]
        completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
        print(completed.stdout)
        if completed.returncode != 0:
            print(completed.stderr)
            raise RuntimeError("GitHub push failed")
        push_result = completed.stdout
        mo.md("Pushed to GitHub branch `study-sunday` under `outputs/study/`.")
    return (push_result,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Next
        1. Prefer **Download** button, or push with a GitHub token above.
        2. After push, get the file from:
           `https://github.com/laraib-b/ReCAP-study/tree/study-sunday/outputs/study`
        3. Optional later: seeds 1 and 2.
        """
    )
    return


if __name__ == "__main__":
    app.run()
