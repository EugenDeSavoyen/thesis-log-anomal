# Event Log Anomaly Detection

This repository contains the code for a thesis project on anomaly detection in process and system event logs.

The pipeline parses raw logs into templates, builds chronological or block-aware windows, scores suspicious regions with statistical tail models, evaluates classical baselines, and prepares bounded review packets for semantic triage.

## Repository Layout

- `src/thesis_log_anomaly/` - reusable pipeline modules.
- `scripts/` - experiment and evaluation entry points.
- `configs/` - reproducible configuration files.
- `prompts/` - prompt templates used by optional triage experiments.
- `tests/` - smoke and unit tests.

Generated data, reports, model weights, and local caches are intentionally not versioned.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,retrieval]"
```

The core pipeline expects datasets under `data/raw/`. Large public datasets and generated outputs should be kept outside version control.

## Example Commands

Run a configured pipeline:

```powershell
python scripts\run_pipeline.py configs\base.yaml
```

Run classical baseline evaluation:

```powershell
python scripts\run_classical_triage_baseline.py
```

Run tests:

```powershell
python -m pytest
```

Some integration tests require local datasets in `data/raw/`; without those files, only data-free tests are expected to run.
