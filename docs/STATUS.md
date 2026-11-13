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

## What is still needed for the paper

1. **Fuller profile on molab** (`reduced`). First seed-0 run finished in **12.9 min on GPU** but portfolio metrics were **invalid** (CR≈−1): Yahoo returned all-zero prices for delisted **WBA**, and **DOW** either zero-filled pre-IPO or truncated the panel to 2019. Fix: drop WBA/DOW from the study ticker list + `drop_unusable_tickers` (zeros and short histories) before date alignment. **Re-run reduced seed 0** on molab (~13 min GPU) after pulling this commit.
2. If that re-run is healthy, seeds 1–2 on molab (~13 min each).
3. Write-up: detector mismatch (repo vs paper), simulation calibration, portfolio tables, limitations.

## Hardware rules (for the human)

| Where | Use for | Typical time |
|---|---|---|
| **Laptop (Cursor open)** | debug, micro runs, aggregation | minutes to ~15 min |
| **molab** | anything that needs lots of RAM or >30 min | leave the browser tab open until done |

I will always warn before starting anything that needs the laptop open for long or that must go to molab.
