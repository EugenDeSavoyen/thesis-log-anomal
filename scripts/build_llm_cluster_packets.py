from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_llm_review_packets import LABEL_LIKE_KEYS, assert_no_label_leakage
from scripts.run_classical_triage_baseline import DEFAULT_METHODS, rank_events


CLUSTER_PACKET_SCHEMA_VERSION = "llm_template_cluster_packet_v1"
DEFAULT_NUMERIC_SUMMARY_COLUMNS = [
    "classical_score",
    "template_burst_score",
    "template_count_deviation_score",
    "template_count_past_z",
    "novelty_score",
    "rarity_score",
    "historical_count_log",
    "local_sequence_context_score",
    "suspicious_window_count",
    "max_window_gev_score",
    "max_window_gev_excess",
    "max_window_template_entropy",
    "max_window_rare_template_ratio",
    "max_window_unseen_template_ratio",
    "max_window_new_template_ratio",
    "max_window_mean_event_novelty_score",
    "max_template_count_in_window",
    "max_template_ratio_in_window",
]
REPRESENTATIVE_EVENT_COLUMNS = [
    "event_id",
    "event_order",
    "block_id",
    "classical_score",
    "template_burst_score",
    "template_count_deviation_score",
    "novelty_score",
    "rarity_score",
    "local_sequence_context_score",
    "message",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build label-free JSONL template-cluster packets for local LLM review."
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/event_level_candidates_bgl_multiblock.csv",
    )
    parser.add_argument("--dataset", default="bgl_multiblock")
    parser.add_argument("--selection-method", default="template_burst", choices=DEFAULT_METHODS)
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument("--max-events-per-cluster", type=int, default=8)
    parser.add_argument("--prompt-strategy", default="template_cluster_v1")
    parser.add_argument(
        "--output-jsonl",
        default="outputs/reports/llm_cluster_packets_bgl_multiblock.jsonl",
    )
    parser.add_argument(
        "--output-report",
        default="outputs/reports/llm_cluster_packets_bgl_multiblock.json",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    packets = build_template_cluster_packets(
        df,
        dataset=args.dataset,
        selection_method=args.selection_method,
        top_k=args.top_k,
        max_clusters=args.max_clusters,
        max_events_per_cluster=args.max_events_per_cluster,
        prompt_strategy=args.prompt_strategy,
    )

    output_jsonl = Path(args.output_jsonl)
    output_report = Path(args.output_report)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(packets, output_jsonl)
    report = build_cluster_packet_report(
        packets,
        input_csv=args.input_csv,
        output_jsonl=str(output_jsonl),
        dataset=args.dataset,
        selection_method=args.selection_method,
        top_k=args.top_k,
        max_clusters=args.max_clusters,
        max_events_per_cluster=args.max_events_per_cluster,
        prompt_strategy=args.prompt_strategy,
    )
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_template_cluster_packets(
    df: pd.DataFrame,
    *,
    dataset: str,
    selection_method: str,
    top_k: int,
    max_clusters: int | None = None,
    max_events_per_cluster: int = 8,
    prompt_strategy: str = "template_cluster_v1",
) -> list[dict[str, Any]]:
    ranked = rank_events(df, selection_method).head(top_k).copy()
    ranked["selection_rank"] = range(1, len(ranked) + 1)
    if "template" not in ranked:
        raise ValueError("Input candidate table must include a template column.")

    cluster_rows = []
    for template, group in ranked.groupby("template", sort=False):
        cluster_rows.append(
            {
                "template": template,
                "group": group.sort_values("selection_rank"),
                "best_rank": int(group["selection_rank"].min()),
                "event_count": int(len(group)),
                "max_score": float(group["classical_score"].max()) if "classical_score" in group else 0.0,
            }
        )
    cluster_rows = sorted(cluster_rows, key=lambda item: (item["best_rank"], -item["event_count"], -item["max_score"]))
    if max_clusters is not None:
        cluster_rows = cluster_rows[:max_clusters]

    packets = []
    for cluster_rank, item in enumerate(cluster_rows, start=1):
        packet = _cluster_packet(
            item["group"],
            template=str(item["template"]),
            dataset=dataset,
            selection_method=selection_method,
            top_k=top_k,
            cluster_rank=cluster_rank,
            max_events_per_cluster=max_events_per_cluster,
            prompt_strategy=prompt_strategy,
        )
        assert_no_label_leakage(packet)
        packets.append(packet)
    return packets


def write_jsonl(packets: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def build_cluster_packet_report(
    packets: list[dict[str, Any]],
    *,
    input_csv: str,
    output_jsonl: str,
    dataset: str,
    selection_method: str,
    top_k: int,
    max_clusters: int | None,
    max_events_per_cluster: int,
    prompt_strategy: str,
) -> dict[str, Any]:
    event_counts = [int(packet["cluster"]["event_count"]) for packet in packets]
    return {
        "packet_schema_version": CLUSTER_PACKET_SCHEMA_VERSION,
        "input_csv": input_csv,
        "output_jsonl": output_jsonl,
        "dataset": dataset,
        "selection_method": selection_method,
        "top_k": top_k,
        "max_clusters": max_clusters,
        "max_events_per_cluster": max_events_per_cluster,
        "prompt_strategy": prompt_strategy,
        "num_packets": len(packets),
        "covered_events": sum(event_counts),
        "max_cluster_events": max(event_counts) if event_counts else 0,
        "packet_sha256": _sha256_json(packets),
        "label_policy": "labels and evaluation metrics are excluded from LLM cluster packets",
        "notes": [
            "Each packet represents one parsed-template cluster within a ranked event budget.",
            "Use prompts/log_triage_cluster_v1.md with scripts/run_llm_triage.py.",
            "Cluster packets are intended to reduce repeated event-level LLM calls.",
        ],
    }


def _cluster_packet(
    group: pd.DataFrame,
    *,
    template: str,
    dataset: str,
    selection_method: str,
    top_k: int,
    cluster_rank: int,
    max_events_per_cluster: int,
    prompt_strategy: str,
) -> dict[str, Any]:
    group = group.sort_values("selection_rank")
    event_ids = [_as_int(value) for value in group["event_id"].dropna().tolist()] if "event_id" in group else []
    selection_ranks = [_as_int(value) for value in group["selection_rank"].dropna().tolist()]
    block_ids = [_clean_value(value) for value in group["block_id"].dropna().unique().tolist()] if "block_id" in group else []
    packet_id = f"{dataset}:{selection_method}:top{top_k}:cluster:{cluster_rank:04d}"
    return {
        "packet_schema_version": CLUSTER_PACKET_SCHEMA_VERSION,
        "packet_id": packet_id,
        "dataset": dataset,
        "selection_method": selection_method,
        "top_k_budget": top_k,
        "prompt_strategy": prompt_strategy,
        "cluster": {
            "cluster_rank": cluster_rank,
            "template": template,
            "event_count": int(len(group)),
            "member_event_ids": event_ids,
            "member_selection_ranks": selection_ranks,
            "event_id_range": _range_summary(event_ids),
            "selection_rank_range": _range_summary(selection_ranks),
            "block_ids": block_ids,
            "unique_blocks": len(block_ids),
        },
        "score_summary": _numeric_summary(group, DEFAULT_NUMERIC_SUMMARY_COLUMNS),
        "representative_events": _representative_events(group, max_events=max_events_per_cluster),
    }


def _representative_events(group: pd.DataFrame, *, max_events: int) -> list[dict[str, Any]]:
    if max_events <= 0:
        return []
    if len(group) <= max_events:
        chosen = group
    else:
        head_count = max_events // 2
        tail_count = max_events - head_count
        chosen = pd.concat([group.head(head_count), group.tail(tail_count)]).drop_duplicates(subset=["event_id"])
    events = []
    for _, row in chosen.iterrows():
        event = {"selection_rank": _as_int(row.get("selection_rank"))}
        for column in REPRESENTATIVE_EVENT_COLUMNS:
            if column in row.index:
                value = _clean_value(row.get(column))
                if value is not None:
                    event[column] = value
        _assert_no_label_like_keys(event)
        events.append(event)
    return events


def _numeric_summary(group: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, Any]]:
    summary = {}
    for column in columns:
        if column not in group:
            continue
        values = pd.to_numeric(group[column], errors="coerce").dropna()
        if values.empty:
            continue
        summary[column] = {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }
    return summary


def _range_summary(values: list[int | None]) -> dict[str, int | None]:
    clean = [int(value) for value in values if value is not None]
    if not clean:
        return {"min": None, "max": None}
    return {"min": min(clean), "max": max(clean)}


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


def _as_int(value: Any) -> int | None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    return int(cleaned)


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _assert_no_label_like_keys(value: dict[str, Any]) -> None:
    for key in value:
        normalized = str(key).lower()
        if normalized in LABEL_LIKE_KEYS or normalized.startswith("label_"):
            raise ValueError(f"Label-like key found in representative event: {key}")


if __name__ == "__main__":
    main()
