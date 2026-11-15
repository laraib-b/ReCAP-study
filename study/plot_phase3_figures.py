"""Generate the seven Phase 3 report figures from saved experiment outputs.

IMPORTANT — provenance:
These visualization scripts were added separately to process already-saved
CSV/JSON experiment outputs for the Phase 3 report. They were NOT part of the
original training / evaluation pipeline. They do not retrain models and do not
fabricate metrics; they only plot verified tables and run JSON under results/.

Usage (from Milestone3 parent, or any folder where results/ sits beside
implementation/):

    python implementation/study/plot_phase3_figures.py

Figures are written to results/figures/ as fig01…fig07 (.png and .pdf).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_IMPL = Path(__file__).resolve().parents[1]
if str(_IMPL) not in sys.path:
    sys.path.insert(0, str(_IMPL))
from paths import PROJECT_ROOT

ROOT = PROJECT_ROOT
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables_phase3"

VARIANT_ORDER = ["default_2.5", "budget_2", "budget_1", "budget_0.5", "no_adapt_default"]
VARIANT_LABELS = {
    "default_2.5": "Default h=2.5σ",
    "budget_2": "FA budget 2/yr",
    "budget_1": "FA budget 1/yr",
    "budget_0.5": "FA budget 0.5/yr",
    "no_adapt_default": "No adaptation",
    "buy_and_hold": "Buy & hold",
    "equal_weight_daily": "Equal-weight daily",
    "equal_weight_monthly": "Equal-weight monthly",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 200,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _mean_sd_table(path: Path) -> pd.DataFrame:
    """Load a per-seed CSV so means/sds can be recomputed honestly."""
    return pd.read_csv(path)


def fig_portfolio_comparison() -> None:
    """Fig 1 — Reduced DOW30 portfolio CR / SR / MDD by detector variant."""
    df = _mean_sd_table(ROOT / "results" / "tables_reduced" / "portfolio_per_seed.csv")
    metrics = ["CR", "SR", "MDD"]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))
    xlabels = [VARIANT_LABELS.get(v, v) for v in VARIANT_ORDER]
    x = np.arange(len(VARIANT_ORDER))
    for ax, metric in zip(axes, metrics):
        means, sds = [], []
        for v in VARIANT_ORDER:
            sub = df[df["variant"] == v][metric]
            means.append(sub.mean())
            sds.append(sub.std(ddof=1) if len(sub) > 1 else 0.0)
        ax.bar(x, means, yerr=sds, capsize=3, color="#4C78A8")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=35, ha="right")
        ax.set_title(metric)
        ax.axhline(0.0, color="black", linewidth=0.8)
    fig.suptitle("Figure 1. Reduced DOW30 portfolio metrics (mean ± sd, 3 seeds)", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG / "fig01_reduced_portfolio_comparison.png", bbox_inches="tight")
    fig.savefig(FIG / "fig01_reduced_portfolio_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_detector_calibration() -> None:
    """Fig 2 — Hold-out simulated false-alarm rates vs target FA budgets."""
    df = _mean_sd_table(ROOT / "results" / "tables_reduced" / "detector_per_seed.csv")
    order = ["default_2.5", "budget_2", "budget_1", "budget_0.5"]
    targets = {"default_2.5": None, "budget_2": 2.0, "budget_1": 1.0, "budget_0.5": 0.5}
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    achieved, sds, labels = [], [], []
    for v in order:
        sub = df[df["variant"] == v]["holdout_sim_alarms_per_year"]
        achieved.append(sub.mean())
        sds.append(sub.std(ddof=1) if len(sub) > 1 else 0.0)
        labels.append(VARIANT_LABELS[v])
    x = np.arange(len(order))
    ax.bar(x, achieved, yerr=sds, capsize=3, color="#72B7B2", label="Hold-out sim alarms/yr")
    for i, v in enumerate(order):
        t = targets[v]
        if t is not None:
            ax.plot([i - 0.35, i + 0.35], [t, t], color="#E45756", linewidth=2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Alarms per year")
    ax.set_title("Figure 2. False-alarm calibration on no-change simulations")
    ax.legend(["Target budget", "Achieved hold-out rate"])
    fig.tight_layout()
    fig.savefig(FIG / "fig02_detector_calibration.png", bbox_inches="tight")
    fig.savefig(FIG / "fig02_detector_calibration.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_fa_delay_tradeoff() -> None:
    """Fig 3 — Trade-off between false-alarm rate and detection delay (+2σ shifts)."""
    syn = _mean_sd_table(ROOT / "results" / "tables_reduced" / "synthetic_per_seed.csv")
    sub = syn[syn["shift_sigma"] == 2].copy()
    det = _mean_sd_table(ROOT / "results" / "tables_reduced" / "detector_per_seed.csv")
    rows = []
    for v in ["default_2.5", "budget_2", "budget_1", "budget_0.5"]:
        fa = det[det["variant"] == v]["holdout_sim_alarms_per_year"].mean()
        delay = sub[sub["label"] == v]["median_delay_days"].mean()
        rate = sub[sub["label"] == v]["detection_rate"].mean()
        rows.append({"variant": v, "fa": fa, "delay": delay, "detection_rate": rate})
    plot_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    sc = ax.scatter(
        plot_df["fa"],
        plot_df["delay"],
        s=80,
        c=plot_df["detection_rate"],
        cmap="viridis",
        vmin=0.9,
        vmax=1.0,
    )
    for _, r in plot_df.iterrows():
        ax.annotate(
            VARIANT_LABELS[r["variant"]],
            (r["fa"], r["delay"]),
            textcoords="offset points",
            xytext=(6, 4),
        )
    ax.set_xlabel("Hold-out sim false alarms / year")
    ax.set_ylabel("Median detection delay (days), +2σ shift")
    ax.set_title("Figure 3. False-alarm vs detection-delay trade-off")
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Detection rate")
    fig.tight_layout()
    fig.savefig(FIG / "fig03_fa_delay_tradeoff.png", bbox_inches="tight")
    fig.savefig(FIG / "fig03_fa_delay_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_finetune_ablation() -> None:
    """Fig 4 — Ablation of fine-tune budget (64 / 256 / 1024) on micro DOW30 CR."""
    frames = []
    for steps, folder in [(64, "tables"), (256, "tables_ft256"), (1024, "tables_ft1024")]:
        df = pd.read_csv(ROOT / "results" / folder / "portfolio_per_seed.csv")
        g = df.groupby("variant")["CR"].agg(["mean", "std"]).reset_index()
        g["finetune_steps"] = steps
        frames.append(g)
    all_df = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for v in VARIANT_ORDER:
        sub = all_df[all_df["variant"] == v].sort_values("finetune_steps")
        ax.errorbar(
            sub["finetune_steps"],
            sub["mean"],
            yerr=sub["std"].fillna(0.0),
            marker="o",
            capsize=3,
            label=VARIANT_LABELS[v],
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 256, 1024])
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: str(int(x))))
    ax.set_xlabel("Fine-tune timesteps per adaptation")
    ax.set_ylabel("Cumulative return (mean ± sd)")
    ax.set_title("Figure 4. Ablation: adaptation training budget (micro DOW30)")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(FIG / "fig04_finetune_ablation.png", bbox_inches="tight")
    fig.savefig(FIG / "fig04_finetune_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_alarm_timeline() -> None:
    """Fig 5 — Test-period detector alarm dates for reduced DOW30 seed 0."""
    run = ROOT / "results" / "runs" / "study_reduced_seed0_20261113T123541Z.json"
    payload = json.loads(run.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    y = 0
    yticks, ylabels = [], []
    for name in ["default_2.5", "budget_2", "budget_1", "budget_0.5"]:
        dates = pd.to_datetime(payload["results"][name]["detector"]["alarm_dates"])
        ax.scatter(dates, np.full(len(dates), y), marker="|", s=120)
        yticks.append(y)
        ylabels.append(VARIANT_LABELS[name])
        y += 1
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_title("Figure 5. Test-period detector alarm dates (seed 0, reduced DOW30)")
    ax.set_xlabel("Date")
    fig.tight_layout()
    fig.savefig(FIG / "fig05_alarm_timeline_seed0.png", bbox_inches="tight")
    fig.savefig(FIG / "fig05_alarm_timeline_seed0.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_risk_return_scatter() -> None:
    """Fig 6 — Risk–return scatter (MDD vs Sharpe) for reduced DOW30 means."""
    df = pd.read_csv(ROOT / "results" / "tables_reduced" / "portfolio_per_seed.csv")
    means = df.groupby("variant")[["CR", "SR", "MDD"]].mean().reset_index()
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for _, r in means.iterrows():
        ax.scatter(r["MDD"], r["SR"], s=70)
        ax.annotate(
            VARIANT_LABELS.get(r["variant"], r["variant"]),
            (r["MDD"], r["SR"]),
            textcoords="offset points",
            xytext=(5, 4),
        )
    ax.set_xlabel("Maximum drawdown")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Figure 6. Risk–return summary (reduced DOW30 means)")
    fig.tight_layout()
    fig.savefig(FIG / "fig06_risk_return.png", bbox_inches="tight")
    fig.savefig(FIG / "fig06_risk_return.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_per_seed_cr() -> None:
    """Fig 7 — Per-seed cumulative returns for reduced DOW30 variants."""
    df = pd.read_csv(ROOT / "results" / "tables_reduced" / "portfolio_per_seed.csv")
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    width = 0.22
    seeds = sorted(df["seed"].unique())
    x = np.arange(len(VARIANT_ORDER))
    for i, seed in enumerate(seeds):
        vals = []
        for v in VARIANT_ORDER:
            row = df[(df["seed"] == seed) & (df["variant"] == v)]
            vals.append(float(row["CR"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + (i - 1) * width, vals, width=width, label=f"seed {seed}")
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[v] for v in VARIANT_ORDER], rotation=25, ha="right")
    ax.set_ylabel("Cumulative return")
    ax.set_title("Figure 7. Per-seed cumulative returns (reduced DOW30)")
    ax.legend()
    ax.axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "fig07_per_seed_cr.png", bbox_inches="tight")
    fig.savefig(FIG / "fig07_per_seed_cr.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    _style()
    # Seven report figures only (no optional classical-baseline plots).
    fig_portfolio_comparison()
    fig_detector_calibration()
    fig_fa_delay_tradeoff()
    fig_finetune_ablation()
    fig_alarm_timeline()
    fig_risk_return_scatter()
    fig_per_seed_cr()
    print(f"Wrote seven report figures under {FIG}")


if __name__ == "__main__":
    main()
