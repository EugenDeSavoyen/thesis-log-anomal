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


BEST_UPSTREAM = {
    "name": "historically_rare_ratio_100_50",
    "windowing": {"size": 100, "stride": 50},
    "evt": {"target_statistic": "historically_rare_template_ratio", "significance_level": 0.90},
}

POT_GRID = [
    {"threshold_quantile": 0.60, "tail_alpha": 0.30, "min_exceedances": 1},
    {"threshold_quantile": 0.65, "tail_alpha": 0.30, "min_exceedances": 1},
    {"threshold_quantile": 0.70, "tail_alpha": 0.30, "min_exceedances": 1},
    {"threshold_quantile": 0.70, "tail_alpha": 0.20, "min_exceedances": 1},
    {"threshold_quantile": 0.75, "tail_alpha": 0.20, "min_exceedances": 1},
    {"threshold_quantile": 0.80, "tail_alpha": 0.20, "min_exceedances": 1},
    {"threshold_quantile": 0.80, "tail_alpha": 0.10, "min_exceedances": 1},
    {"threshold_quantile": 0.85, "tail_alpha": 0.10, "min_exceedances": 1},
    {"threshold_quantile": 0.90, "tail_alpha": 0.10, "min_exceedances": 1},
    {"threshold_quantile": 0.70, "tail_alpha": 0.20, "min_exceedances": 3},
    {"threshold_quantile": 0.80, "tail_alpha": 0.20, "min_exceedances": 3},
]


def run_once(base_config: dict, parsed_stream: dict, pot_params: dict) -> dict:
    config = deepcopy(base_config)
    config["windowing"].update(BEST_UPSTREAM["windowing"])
    config["evt"].update(BEST_UPSTREAM["evt"])
    config["pot"].update(
        {
            "enabled": True,
            "score_target": "event_novelty",
            **pot_params,
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

    score = score_result(report)
    return {
        "pot": config["pot"],
        "report": report,
        "ranking_score": score,
    }


def score_result(report: dict) -> float:
    pot = report["post_pot"]
    precision = pot.get("precision") or 0.0
    recall = pot.get("recall") or 0.0
    reduction = pot.get("event_reduction_ratio")
    reduction_bonus = (1.0 - reduction) if reduction is not None else 0.0
    # Emphasize anomaly retention first, then precision, then candidate reduction.
    return (recall * 0.6) + (precision * 0.25) + (reduction_bonus * 0.15)


def main() -> None:
    base_config = load_config("configs/base.yaml")
    logs = load_logs(base_config)
    parsed_stream = parse_templates(logs, base_config)

    results = [run_once(base_config, parsed_stream, pot_params) for pot_params in POT_GRID]
    results.sort(key=lambda item: item["ranking_score"], reverse=True)

    output_path = ROOT / "outputs" / "reports" / "best_pot_tuning.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} tuning results to {output_path}")

    best = results[0]
    print("Best setting:")
    print(json.dumps(best, indent=2))

    print("\nSummary:")
    for item in results:
        pot = item["report"]["post_pot"]
        print(
            f"q={item['pot']['threshold_quantile']:.2f}",
            f"tail={item['pot']['tail_alpha']:.2f}",
            f"min_exc={item['pot']['min_exceedances']}",
            f"cand={pot['candidate_events']}",
            f"prec={pot['precision']}",
            f"rec={pot['recall']}",
            f"f1={pot['f1']}",
            f"score={item['ranking_score']:.4f}",
        )


if __name__ == "__main__":
    main()
