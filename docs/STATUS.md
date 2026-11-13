# Where the study stands (2026-11-13)

## One-sentence goal

Keep ReCAP fixed, only change how often the detector alarms (by calibrating threshold \(h\)), and check whether fewer alarms still give good portfolio results.

## What we already finished (laptop, ~8 GB RAM, CPU)

All of this used the **micro** profile: 8 tickers, 2019→2021, tiny PPO. Paper-mode ARD. 3 seeds each.

| Experiment | Fine-tune steps | Status |
|---|---:|---|
| Baseline study | 64 | Done |
| Ablation: more adaptation training | 256 | Done |
| Ablation: even more adaptation training | 1024 | Done |

### Portfolio means across 3 seeds (CR = cumulative return)

| Variant | 64 steps | 256 steps | 1024 steps |
|---|---:|---:|---:|
| default (many alarms) | 0.24 | 0.41 | **0.55** |
| budget 0.5 (fewest alarms) | 0.28 | 0.34 | 0.33 |
| budget 1 | 0.35 | 0.29 | 0.41 |
| budget 2 | 0.12 | 0.11 | 0.09 |
| no adaptation | **0.48** | **0.48** | 0.48 |

### What this means in plain English

1. **Calibration works.** We can set \(h\) so that on “nothing changed” simulations the detector fires about 0.5 / 1 / 2 times per year. That part of the paper story is solid.
2. **Fewer alarms ≠ better on this tiny setup.** The strictest budgets did not clearly beat the default.
3. **Giving adaptation more training helps the default a lot.** At 1024 steps, default finally beats “no adaptation” on average return (0.55 vs 0.48), though Sharpe is still a bit lower and noisy.
4. **Micro is a pilot, not the paper claim.** 8 stocks + 1 test year + tiny PPO is too small to conclude. We need a fuller run.

## Reduced DOW30 on molab (GPU)

Protocol: paper ARD, sim-calibrated \(h\), causal eval. 28 tickers (WBA/DOW dropped). Dates 2008-05-01 → 2025-04-29. Commit `2dcdaae`.

| Seed | Status | Notes |
|---:|---|---|
| 0 (bad run, CR≈−1) | **Discard** | WBA zeros / DOW IPO wipeout |
| 0 (re-run `…T123541Z`) | **Done / usable** | Calibration OK for budgets 1 & 2; portfolio CR sane |
| 1 | **Next** | ~13 min GPU |
| 2 | **Next** | ~13 min GPU |

### Seed 0 portfolio (causal)

| Variant | CR | SR | MDD | test alarms/yr |
|---|---:|---:|---:|---:|
| no_adapt_default | **0.66** | **0.75** | 0.22 | 4.82 |
| default_2.5 | 0.27 | 0.36 | 0.32 | 4.82 |
| budget_0.5 | −0.12 | −0.06 | 0.38 | 2.21 |
| budget_1 | −0.43 | −0.48 | 0.57 | 3.01 |
| budget_2 | −0.12 | −0.03 | 0.44 | 3.01 |

Calibration: budget_1 / budget_2 within tolerance; budget_0.5 hits search ceiling at \(h=64\sigma\) (holdout ≈0.69/yr vs target 0.5).

Plain English so far (1 seed only): **no adaptation beats every adapt-on-alarm setting**; stricter false-alarm budgets do not help portfolio CR on this seed.

## What is still needed for the paper

1. Reduced seeds **1–2** on molab (~13 min each GPU; leave tab open). Notebook `JOBS` already pointed at those seeds after this update.
2. Aggregate 3-seed means + write-up (detector mismatch, sim calibration, portfolios, limitations).

## Hardware rules (for the human)

| Where | Use for | Typical time |
|---|---|---|
| **Laptop (Cursor open)** | debug, micro runs, aggregation | minutes to ~15 min |
| **molab** | anything that needs lots of RAM or >30 min | leave the browser tab open until done |

I will always warn before starting anything that needs the laptop open for long or that must go to molab.
