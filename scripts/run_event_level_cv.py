from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_log_anomaly.baselines.event_level import (
    build_event_candidate_rows,
    run_event_level_cross_validation,
    write_event_candidate_rows,
)
from thesis_log_anomaly.config import load_config
from thesis_log_anomaly.stages.evt import score_windows_with_evt
from thesis_log_anomaly.stages.load import load_logs
from thesis_log_anomaly.stages.template import parse_templates
from thesis_log_anomaly.stages.window import build_windows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run block-aware CV for zero-LLM event-level baselines inside GEV-suspicious windows."
    )
    parser.add_argument("config", nargs="?", default="configs/bgl_multiblock_sample.yaml")
    parser.add_argument("--min-validation-recall", type=float, default=0.90)
    parser.add_argument(
        "--output-report",
        default="outputs/reports/event_level_cv_bgl_multiblock.json",
    )
    parser.add_argument(
        "--output-table",
        default="outputs/reports/event_level_cv_bgl_multiblock.csv",
    )
    parser.add_argument(
        "--output-models",
        default="outputs/models/event_level_cv_bgl_multiblock.joblib",
    )
    parser.add_argument(
        "--output-candidates",
        default="data/processed/event_level_candidates_bgl_multiblock.csv",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    records = load_logs(config)
    parsed_stream = parse_templates(records, config)
    windows = build_windows(parsed_stream, config)
    windows_after_gev = score_windows_with_evt(windows, config)
    report = run_event_level_cross_validation(
        windows_after_gev,
        parsed_stream["stream_records"],
        min_validation_recall=args.min_validation_recall,
        output_report=args.output_report,
        output_table=args.output_table,
        output_models=args.output_models,
    )
    write_event_candidate_rows(
        build_event_candidate_rows(windows_after_gev),
        args.output_candidates,
    )
    print(
        json.dumps(
            {
                "best": report["summary"][0] if report.get("summary") else None,
                "report": args.output_report,
                "table": args.output_table,
                "models": args.output_models,
                "candidates": args.output_candidates,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
