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

Upload this file to https://molab.new, enable GPU if available, then Run all.
Leave the browser tab open until the job finishes.
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
    import time
    from pathlib import Path

    import marimo as mo
    import pandas as pd
    import torch

    return Path, json, mo, os, pd, subprocess, sys, time, torch


@app.cell
def _(mo):
    mo.md(
        r"""
        # ReCAP study on molab

        **Leave this tab open until the run finishes.** Closing it can kill the job.

        What this run does:
        - paper-mode ARD (one-sided CUSUM, μ₀ re-estimated after alarms)
        - calibrate \(h\) on no-change simulations (0.5 / 1 / 2 alarms/year)
        - causal portfolio eval (no look-ahead max selection)
        - injected-shift diagnostic

        Default job below is **reduced profile, seed 0** — a timing/memory check
        for the fuller DOW30 setting (longer dates, all tickers, larger PPO).
        Micro pilots already finished on the laptop.
        """
    )
    return


@app.cell
def _():
    # ---- edit only this cell when you want a different job ----
    REPO_URL = "https://github.com/laraib-b/ReCAP-study.git"
    REPO_DIR = "/tmp/ReCAP-study"
    BRANCH = "study-sunday"

    # Reduced seed 0 is done (usable). Run ONE seed at a time so timing is visible.
    # Expected: ~13 min per seed on GPU. If >25 min with live logs stalled, stop.
    # Each item: (profile, seed, finetune_timesteps_or_None, tag)
    JOBS = [
        ("reduced", 1, None, "reduced_seed1"),
        # After seed 1 finishes cleanly, switch to:
        # ("reduced", 2, None, "reduced_seed2"),
    ]
    return BRANCH, JOBS, REPO_DIR, REPO_URL


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
    mo.md(f"**cwd:** `{os.getcwd()}`  \n**commit:** `{commit}`  \nExpect this to match the latest `study-sunday` push.")
    return (commit,)


@app.cell
def _(JOBS, mo, torch):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "n/a"
    mo.md(
        f"**Device:** `{device}` | cuda = `{torch.cuda.is_available()}` | GPU = `{_gpu_name}`  \n"
        f"**Jobs queued:** `{JOBS}`  \n"
        "Expected ~**13 min/seed on GPU**. Live logs should keep moving. "
        "If silent >5 min or total >25 min for one seed, stop the cell."
    )
    return (device,)


@app.cell
def _(JOBS, Path, device, mo, subprocess, time):
    _study_dir = Path("/tmp/ReCAP-study/outputs/study")
    study_files = []
    job_timings = []

    for _profile, _seed, _ft, _tag in JOBS:
        _before = set(_study_dir.glob("study_*.json")) if _study_dir.exists() else set()
        _cmd = [
            "python",
            "-u",
            "study/run_study.py",
            "--profile",
            str(_profile),
            "--dataset",
            "DOW30",
            "--device",
            device,
            "--seed",
            str(_seed),
            "--disable-local-cache",
            "--no-adapt-control",
            "--ard-mode",
            "paper",
            "--tag",
            str(_tag),
        ]
        if _ft is not None:
            _cmd.extend(["--finetune-timesteps", str(_ft)])

        print("Running:", " ".join(_cmd), flush=True)
        print(f"Expect ~13 min on GPU for {_tag}. Streaming logs...", flush=True)
        _t0 = time.time()
        # Stream output live (old capture_output=True hid progress for the whole job).
        _proc = subprocess.Popen(
            _cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        _lines = []
        assert _proc.stdout is not None
        for _line in _proc.stdout:
            _lines.append(_line)
            print(_line, end="", flush=True)
        _code = _proc.wait()
        _minutes = (time.time() - _t0) / 60.0
        if _code != 0:
            raise RuntimeError(
                f"job profile={_profile} seed={_seed} tag={_tag} failed "
                f"(code {_code}) after {_minutes:.1f} min"
            )
        _new = sorted(set(_study_dir.glob("study_*.json")) - _before)
        assert _new, f"job {_tag}: no study JSON written"
        study_files.append(_new[-1])
        job_timings.append(
            {
                "tag": _tag,
                "profile": _profile,
                "seed": _seed,
                "minutes": round(_minutes, 1),
                "file": _new[-1].name,
            }
        )
        print(f"FINISHED {_tag} in {_minutes:.1f} min -> {_new[-1].name}", flush=True)

    mo.md(
        "**Finished.** Times: "
        + ", ".join(f"`{r['tag']}` {r['minutes']} min" for r in job_timings)
    )
    return job_timings, study_files


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
        _tag = _payload.get("tag", "")
        for _name, _result in _payload.get("results", {}).items():
            rows.append(
                {
                    "tag": _tag,
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
                "tag": _tag,
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
                    "tag": _tag,
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
                "tag": _tag,
                "profile": _payload.get("profile"),
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
def _(cal_rows, download, job_timings, mo, pd, preview, rows):
    mo.vstack(
        [
            download,
            mo.md("### Job timings (minutes)"),
            mo.ui.table(pd.DataFrame(job_timings)),
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
        ## After this run

        1. Note the **minutes** in the timing table (that tells us how long seeds 1–2 will take).
        2. Prefer **Download**, or push with a GitHub token.
        3. Paste the compact JSON here (or tell me the GitHub path) so I can update the paper tables.
        4. If it OOM / crashes: paste the error — we may need fewer tickers or a smaller PPO budget.
        """
    )
    return


if __name__ == "__main__":
    app.run()
