# Research and Development Plan

## False-Alarm-Calibrated Regime Detection for ReCAP

**Working title:** *How Reliable Must a Regime Detector Be for Continual Portfolio Adaptation? A False-Alarm-Calibrated Evaluation of ReCAP*

**Study type:** Focused detector ablation and evaluation study  
**Primary scope:** One dataset (NAS100), one continual portfolio framework (ReCAP), one changed component (CUSUM alarm-threshold selection)

---

## 1. Executive summary

ReCAP uses a CUSUM-based Adaptive Regime Detection module (ARD) to identify market changes and trigger policy adaptation. The paper already evaluates regime detection through portfolio outcomes, an ARD ablation, threshold sensitivity, and an HMM alternative. A focused follow-up is to quantify how false alarms and detection delays affect adaptation events and resulting portfolio performance, and whether controlling alarm frequency preserves performance.

This study will keep ReCAP's portfolio model and continual-learning setup fixed while calibrating the detector's CUSUM threshold on training data to meet predeclared false-alarm budgets. Thresholds will be frozen before evaluation. The primary evaluation will use unmodified NAS100 market data; a separate controlled diagnostic with injected shifts will measure detection delay and missed-shift behavior where change dates are known.

The work is feasible as a CPU-first prototype if the public implementation runs successfully. One week is a reasonable target for a pilot, not a guarantee of publishable findings. The public repository describes itself as a minimal runnable implementation with fewer training steps than the paper's full experiments, so reproduction differences must be reported.

## 2. Research problem and gap

### Problem statement

In regime-aware continual portfolio management, a detector's output has operational consequences: an alarm triggers adaptation and may update the policy library. Too many false alarms can cause unnecessary updates; delayed or missed alarms can postpone adaptation after genuine changes. Portfolio performance alone does not reveal whether a detector's alarm behavior is responsible for a gain or a loss.

### Research gap

ReCAP already evaluates its detector using portfolio outcomes, an ARD ablation, threshold sensitivity, and an HMM alternative. However, those evaluations do not directly quantify how false alarms and detection delays affect adaptation events and the resulting portfolio performance. This study evaluates that relationship by calibrating the CUSUM threshold to predeclared false-alarm budgets using training data only. The claim is deliberately limited to this detector-and-adaptation relationship; the study does not test cross-market policy transfer or dynamic asset-universe handling.

### Research question

**When ReCAP's CUSUM threshold is calibrated to a pre-set false-alarm budget using training data only, does it reduce unnecessary adaptation while preserving out-of-sample portfolio performance?**

### Hypotheses

- **H1 — Detector behavior:** False-alarm-budget calibration reduces false adaptation triggers relative to ReCAP's default CUSUM threshold.
- **H2 — Portfolio behavior:** Calibrated detection preserves or improves out-of-sample risk-adjusted returns and drawdown relative to default ReCAP.
- **H3 — Trade-off:** Lower false-alarm budgets may reduce unnecessary updates while increasing detection delay. The study reports this trade-off rather than selecting a threshold using test returns.

These are hypotheses to evaluate, not expected or guaranteed findings.

## 3. Proposed method

### False-alarm-budgeted CUSUM

1. Reproduce the repository's default ReCAP detector and record alarms and adaptation events.
2. Keep the selected detector feature, drift parameter, reset behavior, and the rest of ReCAP fixed.
3. Calibrate the alarm threshold using training data only, targeting predeclared rates such as **0.5, 1, and 2 alarms per 252 trading days**.
4. Freeze each threshold before evaluation on the test period.
5. Run the same ReCAP setup at each threshold and compare against default ReCAP.

The paper's reported CUSUM settings are the initial default reference. Before implementing calibration, inspect the public code and document its actual feature choice, data flow, threshold scaling, reset logic, and alarm timing. The method should be described according to the code actually run, especially if it differs from the paper.

### Calibration safeguards

- Use chronological data splits. Threshold selection must not inspect test returns or test labels.
- Keep all non-detector settings constant across variants.
- Declare target alarm rates before running final test comparisons.
- Save configurations and raw daily outputs for every run.
- If a separate calibration period is carved out of the paper's training window, use the same reduced training data for every comparison and report the protocol deviation.

## 4. Experimental design

### Dataset and scope

Start with **NAS100**, which the paper uses for its primary performance visualization. Keep the main study to one market so the detector question remains tractable. Cross-market policy transfer and dynamic asset-universe handling are outside this study's scope; do not claim that this experiment establishes either form of generalization.

Use the paper's chronological evaluation where the available data and implementation support it. If a training/calibration split is needed, define it chronologically and apply it consistently to all methods. Record dataset dates, asset universe, missing-data handling, feature-generation rules, and transaction-cost assumptions.

### Comparison methods

1. **Default ReCAP:** the repository's original CUSUM configuration.
2. **Calibrated ReCAP:** one run per predeclared false-alarm budget.
3. **Fixed-window retraining:** a fixed-frequency adaptation baseline, if it can be implemented with matched data and training budget.
4. **No test-time adaptation:** a diagnostic baseline to estimate whether adaptation helps over the test period.

The first two are the primary comparison. Treat the other controls as secondary and omit them if they cannot be implemented fairly within the available time.

### Matching requirements

Hold constant across comparisons:

- Policy architecture, features, optimizer, and RL algorithm.
- Training window and update budget, except where the protocol explicitly requires a common shortened training split.
- Transaction costs, initial portfolio value, and evaluation dates.
- Random seeds, using paired seeds across methods.

Do not silently compare a small repository run with the paper's full experimental budget as though they were identical.

## 5. Evaluation measures

### Detector behavior

Report:

- **False adaptation events per year.**
- **Detection delay** after a known shift.
- **Missed-shift rate** within a predeclared detection window.

Use a separate controlled diagnostic for known-date detection scoring: inject shifts of known date and magnitude into held-out feature sequences and score detector alarms against the injected dates. Keep this diagnostic separate from the main portfolio evaluation on unmodified market data. State the injection procedure and magnitudes so it can be reproduced. Do not describe injected-data returns as real-market performance.

Sequential change detection has a false-alarm versus detection-delay trade-off; see Warner and Fellouris, [“CuSum for Sequential Change Diagnosis”](https://par.nsf.gov/servlets/purl/10382479).

### Portfolio outcomes

Report:

- Net cumulative return.
- Annualized Sharpe ratio.
- Maximum drawdown.
- Turnover and transaction costs.
- Number of adaptation events and policy-library size.
- Performance around detected shifts, when it can be computed consistently.

Keep detector metrics and portfolio metrics separate, then analyze their relationship. For example: did fewer alarms yield fewer updates and similar portfolio outcomes, or did they delay adaptation and harm performance?

## 6. Statistical analysis and reporting

- Run a small number of seeds first to confirm the pipeline, then run ten seeds if time and compute permit, matching the paper's stated practice where possible.
- Report means and standard deviations across seeds, plus paired differences against default ReCAP.
- Use block-bootstrap confidence intervals on daily returns for performance differences to preserve some short-range time dependence.
- Predeclare all alarm budgets, show results for every tested budget, and account for multiple comparisons.
- Do not tune the threshold or select the reported result using test performance.
- Keep a log of all attempted variants, including unsuccessful ones.

Backtest selection can overstate evidence when many strategy variants are explored. See Bailey et al., [“The Probability of Backtest Overfitting”](https://escholarship.org/uc/item/4w1110bb). This study should report the full selection process and avoid presenting the best test variant as if it had been preselected.

### Evidence for a positive finding

A persuasive positive result would show that calibration:

1. Reduces false alarms and adaptation events;
2. Preserves or improves out-of-sample risk-adjusted return and drawdown; and
3. Does so with uncertainty estimates that support the comparison.

If calibration reduces false alarms but also harms returns through increased delay, report that trade-off. A null or negative result is still informative if the protocol is sound and the implementation is reproducible.

## 7. Development plan

### Phase A — Inspect and establish a baseline

- Clone and inspect the official ReCAP repository.
- Record dependencies, data source/cache behavior, date ranges, detector implementation, training steps, and available metrics.
- Confirm whether NAS100 data is present or must be obtained, and document the source and any data adjustments.
- Run the default NAS100 configuration once and retain logs, configuration, model output, and daily portfolio series.
- Compare implementation settings with the paper and make deviations explicit.

### Phase B — Add detector instrumentation

- Record detector statistic, threshold, alarm date, reset date, and whether an alarm triggers adaptation.
- Record adaptation events, update steps, and policy-library changes.
- Save daily portfolio weights, returns, turnover, and transaction costs.
- Use a structured configuration file to make each experiment reproducible.

### Phase C — Add threshold calibration

- Add an isolated threshold-calibration routine using training data only.
- Support the predeclared alarm budgets.
- Freeze and save each chosen threshold before test evaluation.
- Preserve a default mode to ensure direct comparison with ReCAP.

### Phase D — Run comparisons and diagnostics

- Run a small pilot to catch protocol and data problems.
- Run the primary default-versus-calibrated comparison with paired seeds.
- Add fixed-window and no-adaptation controls only if they can be matched fairly.
- Run known-shift detector diagnostics separately from unmodified market portfolio tests.

### Phase E — Produce research outputs

- Create a detector table with false alarms, delays, and missed shifts.
- Create a portfolio table with metrics and uncertainty estimates.
- Plot portfolio value with alarm/adaptation dates marked.
- Plot false-alarm budget against detection delay and portfolio outcomes.
- Write the protocol, data handling, deviations, limitations, and all attempted variants.

## 8. Reproducibility and artifact checklist

The completed study should retain:

- Exact source repository URL and commit hash.
- Dataset identity, date range, asset universe, and data processing notes.
- Environment/dependency list and hardware description.
- Configurations, random seeds, and logs for every run.
- Raw daily returns, weights, costs, detector statistics, alarms, and adaptation events.
- Scripts that regenerate tables and figures from saved outputs.
- A limitations statement describing single-market scope, repository training-budget differences, data limitations, and the fact that historical backtests do not establish live performance.

## 9. Feasibility and risks

### Time and hardware

This is CPU-first: detector calibration and analysis are lightweight. Whether full runs need a GPU depends on the repository's actual PPO setup, run time, and number of seeds. First establish that the existing implementation works before arranging hardware. One week is a reasonable target for a prototype or pilot, not for guaranteeing a publishable paper.

### Main risks

- **Baseline does not run or data are unavailable:** prioritize a documented minimal reproduction and reduce scope before adding method changes.
- **Repository differs from paper:** report the difference and frame results as an evaluation of the available implementation, not exact replication.
- **Threshold calibration is contaminated by test data:** use training-only calibration and freeze the threshold.
- **No reliable ground-truth regimes:** use known-date synthetic shifts only for detector diagnostics; do not claim those dates are actual market truth.
- **Too many variants or seeds for the deadline:** prioritize default ReCAP versus calibrated CUSUM; label secondary comparisons as future work.
- **No statistically clear improvement:** report the full trade-off and avoid overstating performance.

## 10. References

1. Pan, C. et al. (2026). **Regime-Adaptive Continual Learning for Portfolio Management.** KDD 2026. [arXiv paper](https://arxiv.org/abs/2606.00143).
2. Pan et al. **ReCAP minimal implementation.** [Official GitHub repository](https://github.com/Dumail/ReCAP).
3. Warner, A. and Fellouris, G. (2022). **CuSum for Sequential Change Diagnosis.** ISIT 2022. [Paper PDF](https://par.nsf.gov/servlets/purl/10382479).
4. Bailey, D. et al. (2017). **The Probability of Backtest Overfitting.** *Journal of Computational Finance*. [Paper record and full text](https://escholarship.org/uc/item/4w1110bb).
