# ReCAP Minimal Implementation

The current repository is the smallest runnable code repository for ReCAP (Regime-Adaptive Continual Learning for Portfolio Management, KDD 2026), which implements the main process and core mechanism in the paper:

- PPO base policy pretraining
- Adaptive Regime Detection (ARD)
- Dynamic Regime-Gate Module (RGM)
- DOW30、NAS100、SP500、NIKKEI30、COMMODITY_ETF `CR`、`SR`、
  `MDD` evaluation.

## Requirements

Suggest using the current `finrl` environment of the project, or installing it yourself:

```bash
pip install -r code/requirements.txt
```

## Data

- By default, the local cache under `datasets/` is read first.
- If there is no local cache, it will fall back to `yfinance` download.
- You can use `--data-dir` to specify other cache directories.
- After passing in `--disable-local-cache`, the 'read local cache first' behavior will be disabled.

## Quick Start

The default command is executed within the time range of DOW30 in the paper:

```bash
python main.py --dataset DOW30
```

## Notes

- Due to the particularity of the application field of the paper, the current goal of this repository is to provide a minimum runnable implementation, rather than a complete replication of the experimental framework. Updates will be made in the future.
- The current default observation feature is a 26 dimensional feature set. When `vix` or `turbulence` is unavailable, the gate and ARD will automatically revert to other market features.
- Training steps and training processes are significantly less than the complete experimental settings in the paper, so that regression verification can be completed within a limited time.
