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


REGIMES = [
    {
        "name": "gev_ensemble_100_50",
        "windowing": {"size": 100, "stride": 50},
        "evt": {
            "ensemble_enabled": True,
            "ensemble_statistics": "unique_templates,template_entropy,historically_rare_template_ratio,unseen_in_history_ratio",
            "target_statistic": "historically_rare_template_ratio",
            "significance_level": 0.90,
        },
    },
    {
        "name": "baseline_max_cluster_100_50",
        "windowing": {"size": 100, "stride": 50},
        "evt": {"target_statistic": "max_cluster_size", "significance_level": 0.90},
    },
    {
        "name": "max_cluster_ratio_100_50",
        "windowing": {"size": 100, "stride": 50},
        "evt": {"target_statistic": "max_cluster_ratio", "significance_level": 0.90},
    },
    {
        "name": "unique_templates_100_50",
        "windowing": {"size": 100, "stride": 50},
        "evt": {"target_statistic": "unique_templates", "significance_level": 0.90},
    },
    {
        "name": "unique_template_ratio_100_50",
        "windowing": {"size": 100, "stride": 50},
        "evt": {"target_statistic": "unique_template_ratio", "significance_level": 0.90},
    },
    {
        "name": "template_entropy_100_50",
        "windowing": {"size": 100, "stride": 50},
        "evt": {"target_statistic": "template_entropy", "significance_level": 0.90},
    },
    {
        "name": "new_template_ratio_100_50",
        "windowing": {"size": 100, "stride": 50},
        "evt": {"target_statistic": "new_template_ratio", "significance_level": 0.90},
    },
    {
        "name": "historically_rare_ratio_100_50",
        "windowing": {"size": 100, "stride": 50},
        "evt": {"target_statistic": "historically_rare_template_ratio", "significance_level": 0.90},
    },
    {
        "name": "unseen_in_history_ratio_100_50",
        "windowing": {"size": 100, "stride": 50},
        "evt": {"target_statistic": "unseen_in_history_ratio", "significance_level": 0.90},
    },
    {
        "name": "novelty_magnitude_combo_100_50",
        "windowing": {"size": 100, "stride": 50},
        "evt": {"target_statistic": "novelty_magnitude_score", "significance_level": 0.90},
    },
    {
        "name": "baseline_max_cluster_200_100",
        "windowing": {"size": 200, "stride": 100},
        "evt": {"target_statistic": "max_cluster_size", "significance_level": 0.90},
    },
]

POT_VARIANTS = [
    {
        "name": "pot_event_novelty",
        "pot": {"score_target": "event_novelty"},
    },
    {
        "name": "pot_historical_rarity",
        "pot": {"score_target": "historical_rarity"},
    },
    {
        "name": "pot_template_burst",
        "pot": {"score_target": "template_burst"},
    },
]


def run_regime(base_config: dict, parsed_stream: dict, regime: dict, pot_variant: dict) -> dict:
    config = deepcopy(base_config)
    config["windowing"].update(regime.get("windowing", {}))
    regime_evt = regime.get("evt", {})
    if "ensemble_enabled" in regime_evt:
        config["evt"]["ensemble_enabled"] = regime_evt["ensemble_enabled"]
    else:
        config["evt"]["ensemble_enabled"] = False
        config["evt"]["ensemble_statistics"] = ""
    config["evt"].update(regime_evt)
    config["pot"].update(pot_variant.get("pot", {}))

    windows = build_windows(parsed_stream, config)
    windows_after_gev = score_windows_with_evt(windows, config)
    windows_after_pot = refine_with_pot(windows_after_gev, config)
    report = evaluate_candidates(
        {
            "parsed_stream": parsed_stream,
            "windows_after_gev": windows_after_gev,
            "windows_after_pot": windows_after_pot,
        },
        {
            **config,
            "evaluation": {"enabled": False},
        },
    )
    return {
        "name": f"{regime['name']}__{pot_variant['name']}",
        "base_regime": regime["name"],
        "pot_variant": pot_variant["name"],
        "windowing": config["windowing"],
        "evt": config["evt"],
        "pot": config["pot"],
        "report": report,
    }


def main() -> None:
    base_config = load_config("configs/base.yaml")
    logs = load_logs(base_config)
    parsed_stream = parse_templates(logs, base_config)

    focus_regimes = [
        regime
        for regime in REGIMES
        if regime["name"] in {
            "gev_ensemble_100_50",
            "unique_templates_100_50",
            "template_entropy_100_50",
            "historically_rare_ratio_100_50",
            "unseen_in_history_ratio_100_50",
            "novelty_magnitude_combo_100_50",
        }
    ]
    results = [
        run_regime(base_config, parsed_stream, regime, pot_variant)
        for regime in focus_regimes
        for pot_variant in POT_VARIANTS
    ]
    output_path = ROOT / "outputs" / "reports" / "pre_llm_regime_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} regime results to {output_path}")

    for result in results:
        gev = result["report"]["post_gev"]
        pot = result["report"]["post_pot"]
        print(
            result["name"],
            f"GEV_f1={gev['f1']}",
            f"GEV_recall={gev['recall']}",
            f"GEV_forward={gev['event_load_to_next_stage']}",
            f"POT_recall={pot['recall']}",
            f"POT_candidates={pot['candidate_events']}",
        )


if __name__ == "__main__":
    main()
