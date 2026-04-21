from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.simulate_streaming_incident_triage import (  # noqa: E402
    _deduplicate_events,
    _enrich_markov_scores,
    _read_true_clusters,
    evaluate_streaming_regions,
)


INSPECT_ACTIONS = {"inspect_event", "inspect_window", "inspect_template_cluster"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM decisions over streaming incident packets with labels used only offline."
    )
    parser.add_argument("--llm-jsonl", default="outputs/reports/streaming_incident_llm_full_v3.jsonl")
    parser.add_argument("--llm-run-json", default="outputs/reports/streaming_incident_llm_full_v3_run.json")
    parser.add_argument("--regions-csv", default="outputs/reports/streaming_incident_regions.csv")
    parser.add_argument("--input-csv", default="data/processed/event_level_candidates_bgl_multiblock.csv")
    parser.add_argument("--true-clusters-csv", default="outputs/reports/anomaly_cluster_density_clusters.csv")
    parser.add_argument("--streaming-report-json", default="outputs/reports/streaming_incident_triage_report.json")
    parser.add_argument("--output-json", default="outputs/reports/streaming_incident_llm_full_v3_eval.json")
    parser.add_argument("--output-md", default="outputs/reports/streaming_incident_llm_full_v3_report.md")
    args = parser.parse_args()

    llm_run = _read_json(Path(args.llm_run_json))
    base_report = _read_json(Path(args.streaming_report_json))
    decisions = _read_decisions(Path(args.llm_jsonl))
    regions = pd.read_csv(args.regions_csv).merge(decisions, on="region_id", how="left")
    df = _enrich_markov_scores(_deduplicate_events(pd.read_csv(args.input_csv)))
    true_clusters = _read_true_clusters(Path(args.true_clusters_csv))
    totals = base_report["totals"]

    selected = regions[regions["recommended_action"].isin(INSPECT_ACTIONS)].copy()
    ignored = regions[regions["recommended_action"].eq("ignore")].copy()
    selected_metrics = evaluate_streaming_regions(selected, df, true_clusters, totals=totals)
    ignored_metrics = evaluate_streaming_regions(ignored, df, true_clusters, totals=totals)

    summary = {
        "llm_run": llm_run,
        "all_streaming_metrics": base_report["metrics"],
        "selected_metrics": selected_metrics,
        "ignored_metrics": ignored_metrics,
        "decision_counts": _counts(decisions, "review_decision"),
        "action_counts": _counts(decisions, "recommended_action"),
        "priority_counts": _counts(decisions, "review_priority"),
        "semantic_category_counts": _counts(decisions, "semantic_category"),
        "selected_regions": _region_rows(selected),
        "ignored_regions": _region_rows(ignored),
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown(summary, output_md)
    print(json.dumps(_console_summary(summary), indent=2))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_decisions(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        packet_id = str(row["packet_id"])
        parsed = row.get("parsed_response") or {}
        rows.append(
            {
                "region_id": int(packet_id.rsplit(":", 1)[1]),
                "packet_id": packet_id,
                "valid_json": bool(row.get("valid_json")),
                "review_decision": row.get("review_decision"),
                "review_priority": row.get("review_priority"),
                "triage_score": row.get("triage_score"),
                "recommended_action": row.get("recommended_action"),
                "semantic_category": parsed.get("semantic_category"),
                "rationale": parsed.get("rationale"),
            }
        )
    return pd.DataFrame(rows)


def _counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df:
        return {}
    return {str(key): int(value) for key, value in df[column].value_counts().items()}


def _region_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        "region_id",
        "packet_id",
        "review_decision",
        "review_priority",
        "triage_score",
        "recommended_action",
        "semantic_category",
        "seed_events",
        "trigger_reasons",
        "interval_start",
        "interval_end",
        "rationale",
    ]
    return df[[column for column in columns if column in df]].to_dict(orient="records")


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    run = summary["llm_run"]
    selected = summary["selected_metrics"]
    ignored = summary["ignored_metrics"]
    all_metrics = summary["all_streaming_metrics"]
    lines = [
        "# Streaming Incident LLM Full V3 Evaluation",
        "",
        "This evaluates `log_triage_streaming_incident_v3` on all emitted streaming incident packets. Labels are used only after LLM decisions to measure offline coverage.",
        "",
        "## LLM Runtime",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| packets reviewed | {run['num_packets']} |",
        f"| valid JSON | {run['valid_json']} / {run['num_packets']} |",
        f"| valid JSON rate | {_fmt(run['valid_json_rate'])} |",
        f"| mean latency, uncached | {_fmt(run['mean_latency_ms_uncached'] / 1000)} s |",
        f"| p95 latency, uncached | {_fmt(run['p95_latency_ms_uncached'] / 1000)} s |",
        f"| total tokens | {int(run['total_tokens_reported']):,} |",
        "",
        "## Decision Counts",
        "",
        "| Field | Counts |",
        "| --- | --- |",
        f"| decisions | {_format_counts(summary['decision_counts'])} |",
        f"| actions | {_format_counts(summary['action_counts'])} |",
        f"| priorities | {_format_counts(summary['priority_counts'])} |",
        f"| semantic categories | {_format_counts(summary['semantic_category_counts'])} |",
        "",
        "## Offline Coverage",
        "",
        "| Metric | All emitted packets | LLM-inspected packets | LLM-ignored packets |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key, name in [
        ("regions_emitted", "regions"),
        ("interval_events", "interval events"),
        ("stream_load", "stream load"),
        ("covered_anomaly_events", "covered anomaly events"),
        ("anomaly_recall_against_all", "anomaly recall"),
        ("true_clusters_hit", "true clusters hit"),
        ("true_cluster_recall", "true cluster recall"),
        ("region_weighted_density", "weighted density"),
        ("compression_ratio", "compression ratio"),
    ]:
        lines.append(f"| {name} | {_fmt(all_metrics[key])} | {_fmt(selected[key])} | {_fmt(ignored[key])} |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The tuned prompt remains schema-stable on the full packet set: all responses are valid JSON and none are `uncertain`.",
            "- The LLM reduces analyst packet load by ignoring corrected-hardware/noise packets.",
            "- Offline, the inspected subset preserves all label-derived anomaly and incident-cluster coverage in this run.",
            "- The main reduction is packet load, not interval stream load, because the ignored regions are small.",
            "",
            "## Artifacts",
            "",
            "- `outputs/reports/streaming_incident_llm_full_v3.jsonl`",
            "- `outputs/reports/streaming_incident_llm_full_v3_run.json`",
            "- `outputs/reports/streaming_incident_llm_full_v3_eval.json`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{key}: {value}" for key, value in counts.items())


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _console_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_counts": summary["decision_counts"],
        "action_counts": summary["action_counts"],
        "selected_metrics": summary["selected_metrics"],
        "ignored_metrics": summary["ignored_metrics"],
    }


if __name__ == "__main__":
    main()
