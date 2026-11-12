# Paper vs code vs study protocol

## ARD: paper specification vs released code

Paper (Sec. ARD): \(S_t=\max(0,S_{t-1}+(u_t-\mu_0)-\kappa)\); \(\mu_0\) from a historical reference window; alarm when \(S_t>h\); after each change point \(S_t\) resets and \(\mu_0\) is re-estimated on the subsequent reference window; run per feature, union of change points. Defaults \(\kappa=0.5\sigma\), \(h=2.5\sigma\).

`h` is exposed in code as `cusum_threshold_scale` (\(h = \text{scale}\cdot\hat\sigma\), \(\hat\sigma\) from the reference window). \(\kappa = 0.5\hat\sigma\) is fixed.

Mismatches in the released `recap/ard.py` (kept verbatim as `ard_mode="repo"`):

| # | Paper | Released code (`repo`) | Corrected (`paper`) |
|---|---|---|---|
| 1 | One-sided statistic | Two-sided (\(S^+\) and \(S^-\)) | One-sided upward, as in the equation |
| 2 | \(\mu_0\) re-estimated after each change point | \(\mu_0\) fixed for the whole series; only \(S\) resets | After an alarm at \(t\), \(\mu_0\) = mean of \(u_{t+1..t+W}\); detection resumes at \(t+W+1\) |
| 3 | Reference window is historical | Reference = first \(\min(252, n/2)\) days of whatever series is passed; at test time that is the test series itself, so early test alarms use later test data | Detector streams train then test; reference window pinned to the first \(\min(252, n_\text{train}/2)\) train days |

Choices the paper leaves open, fixed here and stated as ours:
- Post-alarm reference window length \(W\) = `min_segment_days` (the code already forbids segments shorter than this, so no new parameter). \(\hat\sigma\) (hence \(\kappa\), \(h\)) stays from the initial reference window; only \(\mu_0\) is re-estimated, as the paper states.
- No alarms are raised inside a reference window.

Consequence observed in the repo-mode runs (`results/runs/*185311Z*`): with \(\mu_0\) never re-estimated, a persistent level change re-triggers every `min_segment_days`, so even \(h=128\sigma\) produced 2.1 train alarms/yr and could not reach 0.5–2/yr. This is reported as a property of the released code, not of the paper's method.

Features (`vix, turbulence, boll_ub, boll_lb, zd_5, rsi_30`, cross-sectional daily mean) and the multi-feature union / vote / min-segment rules are unchanged in both modes.

## Other upstream differences

- Stock `evaluate()` segments the full test set then picks `max(baseline, adapted)` per segment (look-ahead).
- Training budgets in this repo/pilot are below paper \(10^4\)-per-task / 10-seed settings.

## Study protocol (this project)

- **Detector:** `ard_mode="paper"` for both fitting the policy library and test-time detection. Repo-mode runs are kept as the reference for the released code.
- **Only \(h\) is calibrated.** Features, \(\kappa\), \(\mu_0\) updates and resets are unchanged.
- **Calibration on no-change simulations.** A VAR(1) with correlated Gaussian innovations is fitted to the detector's initial train reference window (spectral radius capped at 0.995 if needed). \(h\) is chosen by bisection on one set of simulations (seed 10000+seed) to hit 0.5 / 1 / 2 alarms/yr; the achieved rate and a Poisson 95% CI are reported on a separate held-out set (seed 20000+seed). The default \(h=2.5\sigma\) is reported on the same held-out set.
- **Historical alarms are not labelled false.** Train and test alarm counts/dates are reported as alarms; market data has no known change dates.
- **Injected-shift diagnostic:** steps of +1, +2, +3, −2 reference-\(\sigma\) added to all ARD features in another held-out simulation set (seed 30000+seed); detection rate within 60 days, median delay, and pre-shift false-alarm rate (true false alarms, since the change date is known). −2σ documents that the one-sided detector does not respond to decreases.
- **Causal eval:** disables `max(baseline, adapted)`; train is streamed through the detector before test, so each alarm at \(t\) uses data up to \(t\).
- **Isolation:** policy library fitted **once** at default \(h=2.5\sigma\); variants change test-time \(h\) only.
- **Compute:** `micro` profile is for runnable pilots; label clearly in the paper.

## What we will not claim

- Exact replication of paper NAS100 Table 2 numbers.
- Cross-market generalization.
- That synthetic-shift diagnostics are real market regimes.
- That historical market alarms are false alarms.
