# Seed-0 micro study results (molab)

**Run id:** `study_micro_dow30_seed0_20261112T180251Z`  
**Profile:** micro · DOW30 · 8 tickers · 2019-01-01 → 2021-06-01 · CPU · causal eval  
**Warning:** pilot-scale only; not paper Table 2 replication.

## Portfolio + detector table

| Variant | CR | SR | MDD | Test alarms/yr | Adaptations/yr | Library |
|---|---:|---:|---:|---:|---:|---:|
| default_2.5 | 0.5302 | 2.196 | 0.1063 | 12.0 | 11.09 | 12 |
| budget_0.5 / 1 / 2* | 0.5331 | 2.180 | 0.1010 | 10.0 | 10.04 | 13 |
| no_adapt_default | 0.5321 | 2.202 | 0.1057 | 12.0 | 0.00 | 12 |

\*All three calibrated budgets collapsed to the **same** threshold (`h ≈ 8.0σ`) because calibration could not reach the target rates (see below).

## Calibration status (important)

| Target alarms/yr | Chosen h/σ | Achieved train alarms/yr | Status |
|---:|---:|---:|---|
| 0.5 | 8.0 | 10.65 | **Missed** (hit search ceiling) |
| 1.0 | 8.0 | 10.65 | **Missed** |
| 2.0 | 8.0 | 10.65 | **Missed** |

So this run does **not** yet answer H1/H2 for 0.5/1/2 budgets. It only compares default `2.5σ` vs a maxed-out `~8σ` detector.

## What we can say from this run

1. **Pipeline works** on molab under causal protocol.
2. Raising threshold from 2.5 → ~8 slightly cut test alarms (12 → 10) and adaptations (11.1 → 10.0).
3. Portfolio metrics barely moved (CR ~0.53, SR ~2.18–2.20). `no_adapt` is almost identical on this micro window.
4. **Synthetic diagnostic is clearer:** higher threshold cut false alarms before the injected shift (often 8–10 → 0) with modest delay increase — this supports the detector trade-off story even though market calibration targets were not hit.

## Next fix before more seeds

Calibration search must:
- raise the upper bound on `threshold_scale` (and/or use absolute `h`),
- report **infeasible** when `min_segment_days` forces a high alarm floor,
- avoid emitting three identical “budget” variants when they map to one scale.
