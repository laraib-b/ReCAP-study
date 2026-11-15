# Baseline / environment log

## Environment (author laptop)

- OS: Windows 10
- CPU: Intel Core i7-8650U (no NVIDIA GPU)
- RAM: ~8 GB
- Python: 3.11
- Upstream commit pin: `68143bb` (Dumail/ReCAP)
- Study cloud: molab (CPU forced for PPO)

## First successful smoke

- Command: `python study/smoke_run.py`
- Mode: synthetic 3-asset / ~80 business days, tiny PPO budgets
- Output: `outputs/smoke/smoke_metrics.json` (local; gitignored)
- Result: pipeline runs; smoke metrics are not research findings

## Study path (current)

1. Micro + finetune ablations on laptop (`docs/REPRODUCE.md`)
2. Reduced DOW30 seeds on molab (`notebooks/study_run_molab.py`)
3. Canonical numbers in `results/`; status in `docs/STATUS.md`
