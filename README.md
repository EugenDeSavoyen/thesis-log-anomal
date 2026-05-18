# Event Log Anomaly Detection

This repository contains the code for a thesis project on anomaly detection in process and system event logs.

The pipeline parses raw logs into templates, builds chronological or block-aware windows, scores suspicious regions with statistical tail models, evaluates classical baselines, and prepares bounded review packets for semantic triage.

The LLM stage is evaluated as a bounded incident reviewer rather than a raw-stream classifier. The repository includes prompt templates and a repeated-run stability script for checking whether inspect/ignore decisions change across uncached LLM runs.

## AI Tool Disclosure

Large language model assistants, including OpenAI ChatGPT/Codex-family tools, were used in a supporting role during this project. Their use included literature search support and preliminary analysis of related work, coding assistance, script refactoring, consistency checks between reported metrics and generated artifacts, table/summary preparation, and English proofreading.

All research questions, system design choices, experiment protocols, interpretation of results, and final wording remain the author's responsibility. AI tools were not used to fabricate experimental data, invent citations, replace evaluation scripts, or substitute for the author's analytical contribution. The local LLM used in experiments is treated separately as an evaluated component of the anomaly-detection pipeline.

## Repository Layout

- `src/thesis_log_anomaly/` - reusable pipeline modules.
- `scripts/` - experiment and evaluation entry points.
- `configs/` - reproducible configuration files.
- `prompts/` - prompt templates used by optional triage experiments.
- `outputs/reports/streaming_llm_stability_*` - compact stability summaries for the incident-level LLM detector.
- `tests/` - smoke and unit tests.

Large generated data, full per-run LLM outputs, model weights, and local caches are intentionally not versioned. Small stability summaries are kept because they document the repeated-run evidence used in the report.

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

Evaluate incident-level LLM stability over a fixed packet set:

```powershell
python scripts\run_streaming_llm_stability_experiment.py --n-runs 10 --output-dir outputs\reports\streaming_llm_stability_gap100_v3_n10
```

The stability runner expects previously generated streaming incident packets, region files, and local Ollama access to the configured model. The committed summaries report 10 repeated runs for the default temperature plus two temperature stress checks.

Run tests:

```powershell
python -m pytest
```

Some integration tests require local datasets in `data/raw/`; without those files, only data-free tests are expected to run.
