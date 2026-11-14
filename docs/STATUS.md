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
| 0 (re-run `…T123541Z`) | **Done / usable** | ~13 min CUDA |
| 1 (`…T131334Z`) | **Done / usable** | 23.3 min CUDA (MLP-PPO on GPU is slow) |
| 2 | **Next** | force **CPU** in notebook (`dd7d687`); ~15–30 min |

### Portfolio CR (causal) — seeds 0 and 1

| Variant | seed 0 | seed 1 |
|---|---:|---:|
| no_adapt_default | **0.66** | **0.65** |
| default_2.5 | 0.27 | −0.05 |
| budget_0.5 | −0.12 | −0.47 |
| budget_1 | −0.43 | 0.12 |
| budget_2 | −0.12 | −0.21 |

Calibration (both seeds): budget_1 / budget_2 within tolerance; budget_0.5 hits \(h=64\sigma\) ceiling.

Plain English so far: **no adaptation (~0.65 CR) beats adapt-on-alarm on both seeds**; fewer alarms do not clearly help.

## What is still needed for the paper

1. Reduced seed **2** on molab (CPU; leave tab open). Notebook already queues it.
2. Aggregate 3-seed means + write-up (detector mismatch, sim calibration, portfolios, limitations).

## Hardware rules (for the human)

| Where | Use for | Typical time |
|---|---|---|
| **Laptop (Cursor open)** | debug, micro runs, aggregation | minutes to ~15 min |
| **molab** | anything that needs lots of RAM or >30 min | leave the browser tab open until done |

I will always warn before starting anything that needs the laptop open for long or that must go to molab.
