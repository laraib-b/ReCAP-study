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
    # Edit this list to choose which seeds to run (seed 0 is already recorded).
    SEEDS = [1, 2]
    return BRANCH, REPO_DIR, REPO_URL, SEEDS


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
def _(Path, SEEDS, device, mo, subprocess):
    _study_dir = Path("/tmp/ReCAP-study/outputs/study")
    study_files = []
    for _seed in SEEDS:
        _before = set(_study_dir.glob("study_*.json")) if _study_dir.exists() else set()
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
            str(_seed),
            "--disable-local-cache",
            "--no-adapt-control",
        ]
        print("Running:", " ".join(_cmd), flush=True)
        _run = subprocess.run(_cmd, check=False, text=True, capture_output=True)
        print(_run.stdout[-2000:] if _run.stdout else "")
        if _run.returncode != 0:
            print(_run.stderr[-4000:] if _run.stderr else "")
            raise RuntimeError(f"seed {_seed}: study run failed with code {_run.returncode}")
        _new = sorted(set(_study_dir.glob("study_*.json")) - _before)
        assert _new, f"seed {_seed}: no study JSON written"
        study_files.append(_new[-1])
    mo.md(f"Finished seeds {SEEDS}: " + ", ".join(f"`{p.name}`" for p in study_files))
    return (study_files,)


@app.cell
def _(json, mo, study_files):
    rows = []
    cal_rows = []
    _compact = []
    _full = {}
    for _path in study_files:
        _payload = json.loads(_path.read_text(encoding="utf-8"))
        _full[_payload.get("run_id", _path.stem)] = _payload
        _seed = _payload.get("seed")
        for _name, _result in _payload.get("results", {}).items():
            rows.append(
                {
                    "seed": _seed,
                    "variant": _name,
                    "CR": _result["metrics"]["CR"],
                    "SR": _result["metrics"]["SR"],
                    "MDD": _result["metrics"]["MDD"],
                    "test alarms/yr": _result["detector"]["alarms_per_year"],
                    "adapt/yr": _result["adaptations_per_year"],
                    "library": _result.get("library_size"),
                }
            )

        _null = _payload.get("null_calibration", {})
        _hist = _payload.get("historical_train_alarms", {})
        cal_rows.append(
            {
                "seed": _seed,
                "variant": "default_2.5",
                "target/yr (no-change sims)": None,
                "h/σ": 2.5,
                "held-out sim alarms/yr": _null.get("default_2.5", {}).get("holdout_sim_alarms_per_year"),
                "within tolerance": None,
                "historical train alarms/yr": _hist.get("default_2.5", {}).get("alarms_per_year"),
            }
        )
        for _name, _cal in _payload.get("calibrations", {}).items():
            cal_rows.append(
                {
                    "seed": _seed,
                    "variant": _name,
                    "target/yr (no-change sims)": _cal.get("target_alarms_per_year"),
                    "h/σ": _cal.get("threshold_scale"),
                    "held-out sim alarms/yr": _cal.get("holdout_sim_alarms_per_year"),
                    "within tolerance": _cal.get("within_tolerance"),
                    "historical train alarms/yr": _hist.get(_name, {}).get("alarms_per_year"),
                }
            )

        _compact.append(
            {
                "run_id": _payload.get("run_id"),
                "git_commit": _payload.get("git_commit"),
                "ard_mode": _payload.get("ard_mode"),
                "seed": _seed,
                "null_calibration": _payload.get("null_calibration"),
                "calibrations": _payload.get("calibrations"),
                "historical_train_alarms": _payload.get("historical_train_alarms"),
                "metrics": {
                    _k: {
                        "metrics": _v.get("metrics"),
                        "detector": {
                            "threshold_scale": _v.get("detector", {}).get("threshold_scale"),
                            "alarms_per_year": _v.get("detector", {}).get("alarms_per_year"),
                            "alarm_dates": _v.get("detector", {}).get("alarm_dates"),
                        },
                        "adaptations_per_year": _v.get("adaptations_per_year"),
                        "library_size": _v.get("library_size"),
                        "n_test_segments": _v.get("n_test_segments"),
                    }
                    for _k, _v in _payload.get("results", {}).items()
                },
                "synthetic_diagnostic": _payload.get("synthetic_diagnostic"),
            }
        )

    preview = json.dumps(_compact, indent=2, default=float)
    download = mo.download(
        data=json.dumps(_full, indent=2, default=float).encode("utf-8"),
        filename="study_runs_this_session.json",
        mimetype="application/json",
        label=f"Download all {len(study_files)} run(s)",
    )
    return cal_rows, download, preview, rows


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
def _(github_token, mo, study_files, subprocess):
    if not github_token.value:
        _msg = mo.md("Skipping GitHub push (no token). Use the Download button above instead.")
    else:
        for _path in study_files:
            _push = subprocess.run(
                [
                    "python",
                    "study/push_artifact.py",
                    str(_path),
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
                raise RuntimeError(f"GitHub push failed for {_path.name}")
        _msg = mo.md(f"Pushed {len(study_files)} file(s) to `study-sunday` under `results/runs/`.")
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
        3. To run other seeds, edit `SEEDS` in the config cell and re-run.
        """
    )
    return


if __name__ == "__main__":
    app.run()
