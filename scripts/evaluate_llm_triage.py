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

from scripts.run_llm_triage import read_jsonl


INSPECT_ACTIONS = {"inspect_event", "inspect_window", "inspect_template_cluster"}
DEFAULT_BASELINE_METHOD = "template_burst"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate LLM triage outputs offline by merging with labels from the candidate table."
    )
    parser.add_argument(
        "--llm-jsonl",
        default="outputs/reports/llm_triage_bgl_multiblock.jsonl",
    )
    parser.add_argument(
        "--packet-jsonl",
        default=None,
        help="Optional source packet JSONL. Required for accurate template-cluster expansion.",
    )
    parser.add_argument(
        "--evaluation-mode",
        default="auto",
        choices=["auto", "event", "cluster"],
        help="Use event rows directly or expand cluster packet decisions to member events.",
    )
    parser.add_argument(
        "--candidate-csv",
        default="data/processed/event_level_candidates_bgl_multiblock.csv",
    )
    parser.add_argument(
        "--baseline-csv",
        default="outputs/reports/classical_triage_baseline_summary.csv",
    )
    parser.add_argument("--baseline-method", default=DEFAULT_BASELINE_METHOD)
    parser.add_argument(
        "--event-cv-json",
        default="outputs/reports/event_level_cv_bgl_multiblock.json",
        help="Optional report used to infer stream/anomaly totals.",
    )
    parser.add_argument("--total-stream-events", type=int, default=None)
    parser.add_argument("--total-anomalies", type=int, default=None)
    parser.add_argument(
        "--top-k",
        default="100,250,500",
        help="Comma-separated top-K budgets to evaluate within the LLM output.",
    )
    parser.add_argument(
        "--output-summary-csv",
        default="outputs/reports/llm_triage_evaluation_summary.csv",
    )
    parser.add_argument(
        "--output-event-csv",
        default="outputs/reports/llm_triage_evaluation_events.csv",
    )
    parser.add_argument(
        "--output-report-json",
        default="outputs/reports/llm_triage_evaluation.json",
    )
    parser.add_argument(
        "--output-report-md",
        default="outputs/reports/llm_triage_evaluation.md",
    )
    args = parser.parse_args()

    packet_rows = read_jsonl(Path(args.packet_jsonl)) if args.packet_jsonl else None
    llm_df = load_llm_outputs(
        Path(args.llm_jsonl),
        packet_rows=packet_rows,
        evaluation_mode=args.evaluation_mode,
    )
    candidates = pd.read_csv(args.candidate_csv)
    totals = infer_totals(
        candidates,
        event_cv_json=Path(args.event_cv_json),
        total_stream_events=args.total_stream_events,
        total_anomalies=args.total_anomalies,
    )
    evaluated = merge_llm_with_labels(llm_df, candidates)
    top_k_values = parse_int_list(args.top_k)
    summary = build_llm_summary(evaluated, top_k_values=top_k_values, totals=totals)
    baseline = load_baseline(Path(args.baseline_csv), method=args.baseline_method, top_k_values=top_k_values)
    comparison = build_baseline_comparison(summary, baseline)

    output_summary = Path(args.output_summary_csv)
    output_events = Path(args.output_event_csv)
    output_json = Path(args.output_report_json)
    output_md = Path(args.output_report_md)
    for path in [output_summary, output_events, output_json, output_md]:
        path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(output_summary, index=False)
    evaluated.to_csv(output_events, index=False)
    report = build_report(
        summary,
        comparison,
        evaluated,
        totals=totals,
        llm_jsonl=args.llm_jsonl,
        candidate_csv=args.candidate_csv,
        baseline_csv=args.baseline_csv,
        baseline_method=args.baseline_method,
        outputs={
            "summary_csv": str(output_summary),
            "event_csv": str(output_events),
            "report_json": str(output_json),
            "report_md": str(output_md),
        },
    )
    output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(summary, comparison, report, output_md)
    print(json.dumps(report, indent=2))


def load_llm_outputs(
    path: Path,
    *,
    packet_rows: list[dict[str, Any]] | None = None,
    evaluation_mode: str = "auto",
) -> pd.DataFrame:
    rows = read_jsonl(path)
    packet_by_id = {row.get("packet_id"): row for row in packet_rows or []}
    mode = infer_evaluation_mode(rows, packet_by_id, evaluation_mode)
    flattened = []
    for row in rows:
        if mode == "cluster":
            packet = packet_by_id.get(row.get("packet_id"))
            flattened.extend(_expand_cluster_row(row, packet))
        else:
            flattened.append(_flatten_llm_row(row))
    df = pd.DataFrame(flattened)
    if df.empty:
        return df
    df["event_id"] = df["event_id"].astype("Int64")
    df["selection_rank"] = df["selection_rank"].astype("Int64")
    sort_columns = [column for column in ["selection_rank", "cluster_rank", "event_id"] if column in df.columns]
    return df.sort_values(sort_columns, na_position="last")


def infer_evaluation_mode(
    rows: list[dict[str, Any]],
    packet_by_id: dict[str, dict[str, Any]],
    requested_mode: str,
) -> str:
    if requested_mode != "auto":
        return requested_mode
    for row in rows:
        packet = packet_by_id.get(row.get("packet_id"))
        if packet and packet.get("packet_schema_version") == "llm_template_cluster_packet_v1":
            return "cluster"
    return "event"


def _flatten_llm_row(row: dict[str, Any]) -> dict[str, Any]:
    parsed = row.get("parsed_response") if isinstance(row.get("parsed_response"), dict) else {}
    return {
        "packet_id": row.get("packet_id"),
        "packet_type": "event",
        "event_id": row.get("event_id"),
        "selection_rank": row.get("selection_rank"),
        "valid_json": bool(row.get("valid_json")),
        "review_decision": row.get("review_decision") or parsed.get("review_decision"),
        "recommended_action": row.get("recommended_action") or parsed.get("recommended_action"),
        "confidence": row.get("confidence") if row.get("confidence") is not None else parsed.get("confidence"),
        "triage_score": row.get("triage_score") if row.get("triage_score") is not None else parsed.get("triage_score"),
        "review_priority": row.get("review_priority") or parsed.get("review_priority"),
        "severity": parsed.get("severity"),
        "reason_codes": json.dumps(parsed.get("reason_codes", []), ensure_ascii=False),
        "latency_ms": row.get("latency_ms"),
        "prompt_tokens": row.get("prompt_tokens"),
        "completion_tokens": row.get("completion_tokens"),
        "total_tokens": row.get("total_tokens"),
        "cache_hit": bool(row.get("cache_hit")),
        "retry_count": row.get("retry_count"),
        "validation_errors": json.dumps(row.get("validation_errors", []), ensure_ascii=False),
        "rationale": parsed.get("rationale"),
    }


def _expand_cluster_row(row: dict[str, Any], packet: dict[str, Any] | None) -> list[dict[str, Any]]:
    base = _flatten_llm_row(row)
    base["packet_type"] = "cluster"
    if not packet:
        return [base]
    cluster = packet.get("cluster", {})
    event_ids = cluster.get("member_event_ids") or [
        item.get("event_id") for item in packet.get("representative_events", []) if item.get("event_id") is not None
    ]
    selection_ranks = cluster.get("member_selection_ranks") or [
        item.get("selection_rank")
        for item in packet.get("representative_events", [])
        if item.get("selection_rank") is not None
    ]
    rows = []
    for offset, event_id in enumerate(event_ids):
        expanded = dict(base)
        expanded["event_id"] = event_id
        expanded["selection_rank"] = selection_ranks[offset] if offset < len(selection_ranks) else None
        expanded["cluster_packet_id"] = row.get("packet_id")
        expanded["cluster_rank"] = cluster.get("cluster_rank")
        expanded["cluster_event_count"] = cluster.get("event_count")
        rows.append(expanded)
    return rows


def merge_llm_with_labels(llm_df: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    if llm_df.empty:
        return llm_df
    label_columns = [
        column
        for column in [
            "event_id",
            "label",
            "template",
            "block_id",
            "template_burst_score",
            "template_count_deviation_score",
            "novelty_score",
            "rarity_score",
            "message",
        ]
        if column in candidates.columns
    ]
    labels = candidates[label_columns].drop_duplicates(subset=["event_id"])
    merged = llm_df.merge(labels, on="event_id", how="left", validate="many_to_one")
    merged["label"] = merged["label"].fillna(0).astype(int)
    merged["llm_likely_anomaly"] = (merged["valid_json"]) & (merged["review_decision"] == "likely_anomaly")
    merged["llm_review_action"] = (merged["valid_json"]) & (merged["recommended_action"].isin(INSPECT_ACTIONS))
    merged["llm_uncertain_or_anomaly"] = (merged["valid_json"]) & (
        merged["review_decision"].isin(["likely_anomaly", "uncertain"])
    )
    if "triage_score" in merged:
        merged["triage_score"] = pd.to_numeric(merged["triage_score"], errors="coerce")
    return merged


def build_llm_summary(
    evaluated: pd.DataFrame,
    *,
    top_k_values: list[int],
    totals: dict[str, int],
) -> pd.DataFrame:
    rows = []
    for top_k in top_k_values:
        subset = evaluated.head(top_k).copy()
        for policy, column in [
            ("likely_anomaly", "llm_likely_anomaly"),
            ("inspect_action", "llm_review_action"),
            ("uncertain_or_anomaly", "llm_uncertain_or_anomaly"),
        ]:
            selected = subset[subset[column]].copy() if column in subset else subset.iloc[0:0].copy()
            rows.append(
                {
                    "policy": policy,
                    "top_k": top_k,
                    "llm_input_events": int(len(subset)),
                    **selection_metrics(selected, subset, evaluated, totals),
                    **resource_metrics(subset),
                }
            )
        if "triage_score" in evaluated:
            scored = evaluated[evaluated["valid_json"] & evaluated["triage_score"].notna()].copy()
            if not scored.empty:
                score_sort_columns = [
                    column
                    for column in ["triage_score", "confidence", "selection_rank", "event_id"]
                    if column in scored.columns
                ]
                ascending = [False if column in {"triage_score", "confidence"} else True for column in score_sort_columns]
                selected = scored.sort_values(score_sort_columns, ascending=ascending).head(top_k).copy()
            else:
                selected = evaluated.iloc[0:0].copy()
            rows.append(
                {
                    "policy": "triage_score_rank",
                    "top_k": top_k,
                    "llm_input_events": int(len(evaluated)),
                    **selection_metrics(selected, evaluated, evaluated, totals),
                    **resource_metrics(evaluated),
                }
            )
    return pd.DataFrame(rows)


def selection_metrics(
    selected: pd.DataFrame,
    llm_budget_subset: pd.DataFrame,
    all_llm_outputs: pd.DataFrame,
    totals: dict[str, int],
) -> dict[str, Any]:
    events = int(len(selected))
    positive_events = int(selected["label"].sum()) if "label" in selected else 0
    total_anomalies = int(totals["total_anomalies"])
    total_stream_events = int(totals["total_stream_events"])
    llm_budget_anomalies = int(llm_budget_subset["label"].sum()) if "label" in llm_budget_subset else 0
    all_llm_anomalies = int(all_llm_outputs["label"].sum()) if "label" in all_llm_outputs else 0
    precision = safe_ratio(positive_events, events)
    recall_all = safe_ratio(positive_events, total_anomalies)
    f1 = f1_score(precision, recall_all)

    selected_templates = set(selected["template"].dropna()) if "template" in selected else set()
    all_templates = set(all_llm_outputs["template"].dropna()) if "template" in all_llm_outputs else set()
    all_positive_templates = (
        set(all_llm_outputs.loc[all_llm_outputs["label"] == 1, "template"].dropna())
        if {"label", "template"}.issubset(all_llm_outputs.columns)
        else set()
    )
    selected_positive_templates = (
        set(selected.loc[selected["label"] == 1, "template"].dropna())
        if {"label", "template"}.issubset(selected.columns)
        else set()
    )

    return {
        "selected_events": events,
        "positive_events": positive_events,
        "negative_events": events - positive_events,
        "precision": precision,
        "recall_against_all_anomalies": recall_all,
        "recall_against_llm_budget_anomalies": safe_ratio(positive_events, llm_budget_anomalies),
        "recall_against_all_llm_output_anomalies": safe_ratio(positive_events, all_llm_anomalies),
        "f1": f1,
        "event_load_ratio_against_stream": safe_ratio(events, total_stream_events),
        "event_load_ratio_against_llm_budget": safe_ratio(events, len(llm_budget_subset)),
        "unique_templates": len(selected_templates),
        "template_load_ratio_against_llm_outputs": safe_ratio(len(selected_templates), len(all_templates)),
        "positive_templates": len(selected_positive_templates),
        "positive_template_recall_against_llm_outputs": safe_ratio(
            len(selected_positive_templates),
            len(all_positive_templates),
        ),
    }


def resource_metrics(subset: pd.DataFrame) -> dict[str, Any]:
    packet_subset = subset.drop_duplicates(subset=["packet_id"]) if "packet_id" in subset else subset
    valid_json = int(packet_subset["valid_json"].sum()) if "valid_json" in packet_subset else 0
    uncached = packet_subset[~packet_subset["cache_hit"]] if "cache_hit" in packet_subset else packet_subset
    latencies = [int(value) for value in uncached.get("latency_ms", pd.Series(dtype=float)).dropna().tolist()]
    total_tokens = int(packet_subset.get("total_tokens", pd.Series(dtype=float)).fillna(0).sum())
    completion_tokens = int(packet_subset.get("completion_tokens", pd.Series(dtype=float)).fillna(0).sum())
    prompt_tokens = int(packet_subset.get("prompt_tokens", pd.Series(dtype=float)).fillna(0).sum())
    return {
        "llm_input_packets": int(len(packet_subset)),
        "valid_json": valid_json,
        "invalid_json": int(len(packet_subset) - valid_json),
        "valid_json_rate": safe_ratio(valid_json, len(packet_subset)),
        "cache_hits": int(packet_subset["cache_hit"].sum()) if "cache_hit" in packet_subset else 0,
        "mean_latency_ms_uncached": None if not latencies else sum(latencies) / len(latencies),
        "p95_latency_ms_uncached": percentile(latencies, 0.95),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def load_baseline(path: Path, *, method: str, top_k_values: list[int]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    baseline = pd.read_csv(path)
    baseline = baseline[(baseline["method"] == method) & (baseline["top_k"].isin(top_k_values))].copy()
    baseline["policy"] = "classical_rank"
    return baseline


def build_baseline_comparison(summary: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if baseline.empty:
        return pd.DataFrame()
    baseline_cols = [
        "top_k",
        "precision",
        "recall_against_all_anomalies",
        "f1",
        "event_load_ratio_against_stream",
        "unique_templates",
        "positive_template_recall",
    ]
    available = [column for column in baseline_cols if column in baseline.columns]
    comparison = summary.merge(
        baseline[available],
        on="top_k",
        how="left",
        suffixes=("", "_classical"),
    )
    for metric in ["precision", "recall_against_all_anomalies", "f1", "event_load_ratio_against_stream"]:
        classical = f"{metric}_classical"
        if classical in comparison.columns:
            comparison[f"{metric}_delta_vs_classical"] = comparison[metric] - comparison[classical]
    return comparison


def build_report(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    evaluated: pd.DataFrame,
    *,
    totals: dict[str, int],
    llm_jsonl: str,
    candidate_csv: str,
    baseline_csv: str,
    baseline_method: str,
    outputs: dict[str, str],
) -> dict[str, Any]:
    best_precision = best_row(summary, "precision")
    best_f1 = best_row(summary, "f1")
    decisions = value_counts(evaluated.get("review_decision", pd.Series(dtype=str)).dropna().tolist())
    actions = value_counts(evaluated.get("recommended_action", pd.Series(dtype=str)).dropna().tolist())
    packet_count = int(evaluated["packet_id"].nunique()) if "packet_id" in evaluated else int(len(evaluated))
    valid_packet_count = (
        int(evaluated.drop_duplicates(subset=["packet_id"])["valid_json"].sum())
        if {"packet_id", "valid_json"}.issubset(evaluated.columns)
        else int(evaluated["valid_json"].sum())
        if "valid_json" in evaluated
        else 0
    )
    return {
        "llm_jsonl": llm_jsonl,
        "candidate_csv": candidate_csv,
        "baseline_csv": baseline_csv,
        "baseline_method": baseline_method,
        "totals": totals,
        "num_llm_outputs": int(len(evaluated)),
        "valid_json_rate": safe_ratio(int(evaluated["valid_json"].sum()), len(evaluated)) if not evaluated.empty else None,
        "decision_counts": decisions,
        "action_counts": actions,
        "best_precision_row": best_precision,
        "best_f1_row": best_f1,
        "comparison_rows": comparison.to_dict("records") if not comparison.empty else [],
        "outputs": outputs,
        "num_llm_packets": packet_count,
        "valid_packet_rate": safe_ratio(valid_packet_count, packet_count),
        "notes": [
            "Labels are merged only in this offline evaluator.",
            "Policy likely_anomaly counts only explicit likely_anomaly decisions.",
            "Policy inspect_action counts recommended inspect_event/window/template_cluster actions.",
            "Policy uncertain_or_anomaly treats uncertain as review-worthy.",
        ],
    }


def write_markdown_report(
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    report: dict[str, Any],
    path: Path,
) -> None:
    lines = [
        "# LLM Triage Evaluation",
        "",
        "This report evaluates LLM triage outputs offline. Labels are merged only after LLM responses are produced.",
        "",
        "## Run Summary",
        "",
        f"- LLM outputs: `{report['num_llm_outputs']}`",
        f"- Valid JSON rate: `{_format_optional(report['valid_json_rate'])}`",
        f"- Decisions: `{report['decision_counts']}`",
        f"- Actions: `{report['action_counts']}`",
        "",
        "## Policy Metrics",
        "",
        "| Policy | Top-K input | Selected | Precision | Recall all anomalies | F1 | Stream load | Valid JSON | Total tokens |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            "| {policy} | {top_k} | {selected} | {precision} | {recall} | {f1} | {stream_load} | {valid_json_rate} | {tokens} |".format(
                policy=row["policy"],
                top_k=int(row["top_k"]),
                selected=int(row["selected_events"]),
                precision=_format_optional(row["precision"]),
                recall=_format_optional(row["recall_against_all_anomalies"]),
                f1=_format_optional(row["f1"]),
                stream_load=_format_optional(row["event_load_ratio_against_stream"]),
                valid_json_rate=_format_optional(row["valid_json_rate"]),
                tokens=int(row["total_tokens"]),
            )
        )
    if not comparison.empty:
        lines.extend(
            [
                "",
                "## Classical Comparison",
                "",
                "| Policy | Top-K | Precision delta | Recall delta | F1 delta | Stream-load delta |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in comparison.to_dict("records"):
            lines.append(
                "| {policy} | {top_k} | {precision_delta} | {recall_delta} | {f1_delta} | {load_delta} |".format(
                    policy=row["policy"],
                    top_k=int(row["top_k"]),
                    precision_delta=_format_optional(row.get("precision_delta_vs_classical")),
                    recall_delta=_format_optional(row.get("recall_against_all_anomalies_delta_vs_classical")),
                    f1_delta=_format_optional(row.get("f1_delta_vs_classical")),
                    load_delta=_format_optional(row.get("event_load_ratio_against_stream_delta_vs_classical")),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this report to compare whether the LLM improves review prioritization or explanation under the same input budget. Do not treat it as evidence that the LLM is the primary detector.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def infer_totals(
    candidates: pd.DataFrame,
    *,
    event_cv_json: Path,
    total_stream_events: int | None,
    total_anomalies: int | None,
) -> dict[str, int]:
    sidecar = read_sidecar_totals(event_cv_json)
    stream_events = total_stream_events or sidecar.get("total_stream_events") or event_id_total(candidates)
    anomalies = total_anomalies or sidecar.get("total_anomalies") or int(candidates["label"].sum())
    return {
        "total_stream_events": int(stream_events),
        "total_anomalies": int(anomalies),
    }


def read_sidecar_totals(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    report = json.loads(path.read_text(encoding="utf-8"))
    summary = report.get("total_event_summary", {})
    totals = {}
    if summary.get("total_labeled_events") is not None:
        totals["total_stream_events"] = int(summary["total_labeled_events"])
    if summary.get("total_anomalous_events") is not None:
        totals["total_anomalies"] = int(summary["total_anomalous_events"])
    return totals


def event_id_total(df: pd.DataFrame) -> int:
    for column in ["event_order", "event_id"]:
        if column in df and not df.empty:
            return int(df[column].max()) + 1
    return int(len(df))


def parse_int_list(raw: str) -> list[int]:
    return sorted({int(item.strip()) for item in raw.split(",") if item.strip() and int(item.strip()) > 0})


def safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def f1_score(precision: float | None, recall: float | None) -> float | None:
    if precision is None or recall is None or precision + recall == 0:
        return None
    return 2 * precision * recall / (precision + recall)


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def best_row(df: pd.DataFrame, metric: str) -> dict[str, Any] | None:
    if df.empty or metric not in df:
        return None
    ranked = df.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if ranked.empty:
        return None
    return ranked.iloc[0].to_dict()


def value_counts(values: list[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _format_optional(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
