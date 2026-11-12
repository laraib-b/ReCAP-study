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
        - ARD in paper mode (one-sided CUSUM, μ0 re-estimated after alarms, train streamed before test)
        - fit once at default h=2.5σ
        - calibrate h on no-change simulations for 0.5 / 1 / 2 alarms/year; report held-out sim rate
        - causal eval (no look-ahead max selection)
        - injected-shift diagnostic on held-out simulations

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
    _repo = Path(REPO_DIR)
    if not _repo.exists():
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
    _cmd = [
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
    print("Running:", " ".join(_cmd))
    study_run = subprocess.run(_cmd, check=False, text=True, capture_output=True)
    print(study_run.stdout[-4000:] if study_run.stdout else "")
    if study_run.returncode != 0:
        print(study_run.stderr[-4000:] if study_run.stderr else "")
        raise RuntimeError(f"study run failed with code {study_run.returncode}")
    mo.md("Study run finished.")
    return (study_run,)


@app.cell
def _(Path, json, mo, study_run):
    assert study_run.returncode == 0
    # Prefer a download button. Molab FILES often does not show runtime-created folders.
    _study_dir = Path("/tmp/ReCAP-study/outputs/study")
    if not _study_dir.exists():
        _study_dir = Path("outputs/study")
    _study_files = sorted(_study_dir.glob("study_*.json"))
    assert _study_files, f"No study JSON found under {_study_dir}"
    latest = _study_files[-1]
    _payload = json.loads(latest.read_text(encoding="utf-8"))

    rows = []
    for _name, _result in _payload.get("results", {}).items():
        rows.append(
            {
                "variant": _name,
                "CR": _result["metrics"]["CR"],
                "SR": _result["metrics"]["SR"],
                "MDD": _result["metrics"]["MDD"],
                "alarms/yr": _result["detector"]["alarms_per_year"],
                "adapt/yr": _result["adaptations_per_year"],
                "alias_of": _result.get("alias_of"),
            }
        )

    _null = _payload.get("null_calibration", {})
    _hist = _payload.get("historical_train_alarms", {})
    cal_rows = [
        {
            "variant": "default_2.5",
            "target/yr (no-change sims)": None,
            "h/σ": 2.5,
            "held-out sim alarms/yr": _null.get("default_2.5", {}).get("holdout_sim_alarms_per_year"),
            "within tolerance": None,
            "historical train alarms/yr": _hist.get("default_2.5", {}).get("alarms_per_year"),
        }
    ]
    for _name, _cal in _payload.get("calibrations", {}).items():
        cal_rows.append(
            {
                "variant": _name,
                "target/yr (no-change sims)": _cal.get("target_alarms_per_year"),
                "h/σ": _cal.get("threshold_scale"),
                "held-out sim alarms/yr": _cal.get("holdout_sim_alarms_per_year"),
                "within tolerance": _cal.get("within_tolerance"),
                "historical train alarms/yr": _hist.get(_name, {}).get("alarms_per_year"),
            }
        )

    _local_copy = Path("latest_study.json")
    _local_copy.write_text(latest.read_text(encoding="utf-8"), encoding="utf-8")

    download = mo.download(
        data=latest.read_bytes(),
        filename=latest.name,
        mimetype="application/json",
        label=f"Download {latest.name}",
    )
    preview = json.dumps(
        {
            "run_id": _payload.get("run_id"),
            "git_commit": _payload.get("git_commit"),
            "ard_mode": _payload.get("ard_mode"),
            "null_calibration": _payload.get("null_calibration"),
            "calibrations": _payload.get("calibrations"),
            "historical_train_alarms": _payload.get("historical_train_alarms"),
            "metrics": {
                _k: {
                    "metrics": _v.get("metrics"),
                    "detector": {
                        "threshold_scale": _v.get("detector", {}).get("threshold_scale"),
                        "alarms_per_year": _v.get("detector", {}).get("alarms_per_year"),
                    },
                    "adaptations_per_year": _v.get("adaptations_per_year"),
                    "alias_of": _v.get("alias_of"),
                }
                for _k, _v in _payload.get("results", {}).items()
            },
            "synthetic_diagnostic": _payload.get("synthetic_diagnostic"),
        },
        indent=2,
        default=float,
    )
    mo.md(f"**Artifact on disk:** `{latest}`")
    return cal_rows, download, latest, preview, rows


@app.cell
def _(cal_rows, download, mo, pd, preview, rows):
    mo.vstack(
        [
            download,
            mo.md("### h calibration (no-change simulations) and historical train alarms"),
            mo.ui.table(pd.DataFrame(cal_rows)),
            mo.md("### Causal portfolio results"),
            mo.ui.table(pd.DataFrame(rows)),
            mo.md("### Compact JSON (copy if download fails)"),
            mo.ui.text_area(value=preview, rows=16),
        ]
    )
    return


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
        _msg = mo.md("Skipping GitHub push (no token). Use the Download button above instead.")
    else:
        _push = subprocess.run(
            [
                "python",
                "study/push_artifact.py",
                str(latest),
                "--token",
                github_token.value,
                "--branch",
                "study-sunday",
            ],
            check=False,
            text=True,
            capture_output=True,
        )
        print(_push.stdout)
        if _push.returncode != 0:
            print(_push.stderr)
            raise RuntimeError("GitHub push failed")
        _msg = mo.md("Pushed to GitHub branch `study-sunday` under `results/runs/`.")
    _msg
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Next
        1. Prefer **Download** button, or push with a GitHub token above.
        2. After push, get the file from:
           `https://github.com/laraib-b/ReCAP-study/tree/study-sunday/results/runs`
        3. Optional later: seeds 1 and 2.
        """
    )
    return


if __name__ == "__main__":
    app.run()
