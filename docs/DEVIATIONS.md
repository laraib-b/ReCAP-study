# Paper vs code vs study protocol

## Upstream minimal ReCAP (`68143bb` base)

- Two-sided CUSUM; paper formula is one-sided.
- After alarm, statistic resets but \(\mu_0\) is **not** re-estimated from a new window.
- Stock `evaluate()` segments the full test set then picks `max(baseline, adapted)` per segment (look-ahead).
- Training budgets in this repo/pilot are below paper \(10^4\)-per-task / 10-seed settings.

## Study protocol (this project)

- **Causal eval:** disables `max(baseline, adapted)` selection; each segment uses a fixed rule (adapt or not).
- **Calibration:** threshold scale chosen on **train** feature series only to match predeclared alarms/year (0.5, 1, 2); frozen before test.
- **Isolation:** policy library fitted **once** at default \(h=2.5\sigma\); variants change **test-time** detector threshold only.
- **Metrics:** portfolio CR/SR/MDD plus test alarm rate and adaptation rate; synthetic-shift delay/miss kept separate.
- **Compute:** `micro` profile (few tickers, short dates, tiny PPO steps) is for runnable pilots on Colab — label clearly in the paper; scale up if hardware allows before Sunday cutoff.

## What we will not claim

- Exact replication of paper NAS100 Table 2 numbers.
- Cross-market generalization.
- That synthetic-shift diagnostics are real market regimes.
