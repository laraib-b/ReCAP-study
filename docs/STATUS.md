# Where the study stands (2026-11-14)

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

## Reduced DOW30 on molab — **3 seeds done**

Protocol: paper ARD, sim-calibrated \(h\), causal eval. 28 tickers (WBA/DOW dropped). Dates 2008-05-01 → 2025-04-29. Pilot PPO budgets (pretrain 1024 / finetune 256).

| Seed | Status | Device / time |
|---:|---|---|
| 0 (bad CR≈−1) | **Discard** | WBA/DOW wipeout |
| 0 re-run `…T123541Z` | **Usable** | CUDA ~13 min |
| 1 `…T131334Z` | **Usable** | CUDA 23.3 min |
| 2 `…T144354Z` | **Usable** | CPU ~60+ min |

### Portfolio CR (causal)

| Variant | seed 0 | seed 1 | seed 2 | **mean** |
|---|---:|---:|---:|---:|
| no_adapt_default | 0.66 | 0.65 | 0.64 | **0.65** |
| default_2.5 | 0.27 | −0.05 | −0.27 | −0.01 |
| budget_0.5 | −0.12 | −0.47 | −0.44 | −0.34 |
| budget_1 | −0.43 | 0.12 | −0.29 | −0.20 |
| budget_2 | −0.12 | −0.21 | −0.28 | −0.20 |

Calibration: budgets 1 & 2 within tolerance on all seeds; budget 0.5 often hits \(h=64\sigma\) ceiling (seed 2 barely within tolerance at ceiling).

### Plain English (reduced pilot)

1. **Calibration works** for ~1–2 alarms/yr on no-change sims.
2. **No adaptation (~0.65 CR) clearly beats every adapt-on-alarm setting** across 3 seeds.
3. **Fewer alarms do not fix adaptation** — stricter budgets stay negative on average.
4. Still **pilot-scale** (not paper Table 2 / NAS100 budgets). Do not overclaim.

## What is still needed for the paper

1. ~~Reduced seeds 0–2~~ **Done**
2. **Write-up:** detector mismatch (repo vs paper), sim calibration, 3-seed portfolio table, limitations (pilot compute, budget_0.5 ceiling, MLP-PPO device notes).

## Hardware rules (for the human)

| Where | Use for | Typical time |
|---|---|---|
| **Laptop (Cursor open)** | debug, micro runs, aggregation, write-up | minutes to ~15 min |
| **molab** | reduced seeds (done) | leave tab open until done |

I will always warn before starting anything that needs the laptop open for long or that must go to molab.
