from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_log_anomaly.config import load_config
from thesis_log_anomaly.stages.evaluation import evaluate_candidates
from thesis_log_anomaly.stages.evt import score_windows_with_evt
from thesis_log_anomaly.stages.load import load_logs
from thesis_log_anomaly.stages.pot import refine_with_pot
from thesis_log_anomaly.stages.template import parse_templates
from thesis_log_anomaly.stages.window import build_windows


WINDOW_GRID = [
    {"size": 40, "stride": 20},
    {"size": 60, "stride": 30},
    {"size": 80, "stride": 40},
    {"size": 100, "stride": 50},
    {"size": 120, "stride": 60},
]

ALPHA_GRID = [0.80, 0.85, 0.90, 0.95]

ENSEMBLE_CANDIDATES = [
    "unique_templates",
    "template_entropy",
    "historically_rare_template_ratio",
    "unseen_in_history_ratio",
    "mean_event_novelty_score",
    "max_event_novelty_score",
    "mean_event_rarity_score",
    "rarity_score_variance",
]

ENSEMBLE_SUBSETS = [
    [
        "unique_templates",
        "template_entropy",
        "historically_rare_template_ratio",
        "unseen_in_history_ratio",
    ],
    [
        "unique_templates",
        "template_entropy",
        "mean_event_novelty_score",
        "max_event_novelty_score",
    ],
    [
        "historically_rare_template_ratio",
        "unseen_in_history_ratio",
        "mean_event_rarity_score",
        "rarity_score_variance",
    ],
    [
        "unique_templates",
        "template_entropy",
        "historically_rare_template_ratio",
        "unseen_in_history_ratio",
        "mean_event_novelty_score",
    ],
    [
        "unique_templates",
        "template_entropy",
        "historically_rare_template_ratio",
        "unseen_in_history_ratio",
        "mean_event_novelty_score",
        "mean_event_rarity_score",
    ],
]


def subset_stream(parsed_stream: dict, start: int, end: int) -> dict:
    return {
        "history_records": parsed_stream.get("history_records", []),
        "history_template_summary": parsed_stream.get("history_template_summary", {}),
        "stream_records": parsed_stream["stream_records"][start:end],
        "stream_summary": parsed_stream.get("stream_summary", {}),
        "history_size": parsed_stream.get("history_size", 0),
        "stream_size": max(0, end - start),
    }


def score_candidate(base_config: dict, parsed_stream: dict, windowing: dict, alpha: float, statistics: list[str]) -> dict:
    config = deepcopy(base_config)
    config["windowing"].update(windowing)
    config["evt"].update(
        {
            "ensemble_enabled": True,
            "ensemble_statistics": ",".join(statistics),
            "significance_level": alpha,
        }
    )
    config["pot"].update(
        {
            "enabled": False,
            "per_window": True,
            "score_target": "event_novelty",
            "threshold_quantile": 0.70,
            "tail_alpha": 0.20,
            "min_exceedances": 1,
        }
    )

    windows = build_windows(parsed_stream, config)
    windows_after_gev = score_windows_with_evt(windows, config)
    windows_after_pot = refine_with_pot(windows_after_gev, config)
    report = evaluate_candidates(
        {
            "parsed_stream": parsed_stream,
            "windows_after_gev": windows_after_gev,
            "windows_after_pot": windows_after_pot,
        },
        {**config, "evaluation": {"enabled": False}},
    )
    return {
        "windowing": windowing,
        "alpha": alpha,
        "statistics": statistics,
        "report": report,
        "validation_score": rank_candidate(report),
    }


def rank_candidate(report: dict) -> float:
    gev = report["post_gev"]
    recall = gev.get("recall") or 0.0
    precision = gev.get("precision") or 0.0
    load = gev.get("event_reduction_ratio")
    load_penalty = load if load is not None else 1.0
    # Emphasize recall first, then precision, then lower forwarded load.
    return (0.60 * recall) + (0.25 * precision) + (0.15 * (1.0 - load_penalty))


def main() -> None:
    base_config = load_config("configs/base.yaml")
    logs = load_logs(base_config)
    parsed_stream = parse_templates(logs, base_config)

    stream_size = parsed_stream["stream_size"]
    val_start = 0
    val_end = stream_size // 2
    test_start = val_end
    test_end = stream_size

    validation_stream = subset_stream(parsed_stream, val_start, val_end)
    test_stream = subset_stream(parsed_stream, test_start, test_end)

    validation_results = [
        score_candidate(base_config, validation_stream, windowing, alpha, statistics)
        for windowing in WINDOW_GRID
        for alpha in ALPHA_GRID
        for statistics in ENSEMBLE_SUBSETS
    ]
    validation_results.sort(key=lambda item: item["validation_score"], reverse=True)
    best = validation_results[0]

    test_result = score_candidate(base_config, test_stream, best["windowing"], best["alpha"], best["statistics"])

    output = {
        "selection_strategy": "validation_split_on_stream_second_half",
        "validation_candidates": validation_results,
        "best_validation_setting": {
            "windowing": best["windowing"],
            "alpha": best["alpha"],
            "statistics": best["statistics"],
            "validation_score": best["validation_score"],
            "report": best["report"],
        },
        "test_result": test_result,
    }

    output_path = ROOT / "outputs" / "reports" / "ensemble_window_tuning.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"Wrote tuning report to {output_path}")
    print("Best validation setting:")
    print(json.dumps(output["best_validation_setting"], indent=2))
    print("Held-out test result:")
    print(json.dumps(test_result, indent=2))


if __name__ == "__main__":
    main()
