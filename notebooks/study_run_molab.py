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

This is the only cloud front door for reproducing study ablations.
Upload to https://molab.new, edit the JOBS cell, Run all.
Leave the browser tab open until the job finishes.

Canonical pilot results are already in git under results/ — re-run only to reproduce.
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

        What each job does:
        - paper-mode ARD (one-sided CUSUM, μ₀ re-estimated after alarms)
        - calibrate \(h\) on no-change simulations (0.5 / 1 / 2 alarms/year)
        - causal portfolio eval (no look-ahead max selection)
        - injected-shift diagnostic

        Edit **only** the `JOBS` cell. Recipes for micro / ft256 / ft1024 / reduced
        are documented in `docs/REPRODUCE.md`. Pilot results already live in `results/`.
        """
    )
    return


@app.cell
def _():
    # ---- edit only this cell when you want a different job ----
    REPO_URL = "https://github.com/laraib-b/ReCAP-study.git"
    REPO_DIR = "/tmp/ReCAP-study"
    BRANCH = "main"  # same tip as study-sunday

    # SB3 MLP-PPO is slower on GPU than CPU — force CPU for study jobs.
    # Each item: (profile, seed, finetune_timesteps_or_None, tag)
    #
    # Historical recipes (uncomment one block; run one seed at a time on molab):
    # JOBS = [("micro", 0, 64, "micro_ft64_seed0")]
    # JOBS = [("micro", 0, 256, "micro_ft256_seed0")]
    # JOBS = [("micro", 0, 1024, "micro_ft1024_seed0")]
    # JOBS = [("reduced", 0, None, "reduced_seed0")]
    #
    # Empty JOBS = clone + show instructions only (no long run).
    FORCE_DEVICE = "cpu"
    JOBS = []
    return BRANCH, FORCE_DEVICE, JOBS, REPO_DIR, REPO_URL


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
    mo.md(f"**cwd:** `{os.getcwd()}`  \n**commit:** `{commit}`  \nExpect this to match the latest `main` push.")
    return (commit,)


@app.cell
def _(FORCE_DEVICE, JOBS, mo, torch):
    device = FORCE_DEVICE
    _gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "n/a"
    _jobs_msg = (
        f"**Jobs queued:** `{JOBS}`"
        if JOBS
        else "**Jobs queued:** none — edit the JOBS cell (see `docs/REPRODUCE.md`), then re-run."
    )
    mo.md(
        f"**Study device:** `{device}` (forced — SB3 MLP-PPO is slow on CUDA)  \n"
        f"GPU present: `{torch.cuda.is_available()}` (`{_gpu_name}`) but unused for PPO.  \n"
        f"{_jobs_msg}  \n"
        "Watch live phase logs. One reduced seed is often ~15–60+ min on CPU."
    )
    return (device,)


@app.cell
def _(JOBS, Path, device, mo, subprocess, time):
    _study_dir = Path("/tmp/ReCAP-study/outputs/study")
    study_files = []
    job_timings = []

    if not JOBS:
        mo.md(
            "No jobs configured. Set `JOBS` to a list of "
            "`(profile, seed, finetune_or_None, tag)` tuples and re-run this cell."
        )
        return job_timings, study_files

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
        print(f"Streaming logs for {_tag} (CPU; reduced often 15–60+ min)...", flush=True)
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
    if not study_files:
        _msg = mo.md("No study files from this session — nothing to push.")
    elif not github_token.value:
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
                    "main",
                ],
                check=False,
                text=True,
                capture_output=True,
            )
            print(_push.stdout)
            if _push.returncode != 0:
                print(_push.stderr)
                raise RuntimeError(f"GitHub push failed for {_path.name}")
        _msg = mo.md(f"Pushed {len(study_files)} file(s) to `main` under `results/runs/`.")
    _msg
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## After this run

        1. Prefer **Download**, or push with a GitHub token.
        2. Copy new JSONs into `results/runs/` on your clone and re-aggregate (`docs/REPRODUCE.md`).
        3. For write-up numbers, prefer the canonical tables already in `results/tables_*`.
        4. Partner sync notes: `docs/PARTNERS.md`.
        """
    )
    return


if __name__ == "__main__":
    app.run()
