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

from scripts.run_classical_triage_baseline import DEFAULT_METHODS, rank_events


PACKET_SCHEMA_VERSION = "llm_event_packet_v1"
DEFAULT_SCORE_COLUMNS = [
    "template_burst_score",
    "template_count_deviation_score",
    "template_count_past_z",
    "novelty_score",
    "rarity_score",
    "historical_count_log",
    "unseen_in_history",
    "historically_rare",
    "is_new_template",
    "local_sequence_context_score",
    "markov_bigram_surprise",
    "markov_trigram_surprise",
]
DEFAULT_WINDOW_CONTEXT_COLUMNS = [
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
DEFAULT_LOCAL_CONTEXT_COLUMNS = [
    "prev_event_novelty_score",
    "next_event_novelty_score",
    "neighbor_max_novelty_score",
    "neighbor_mean_novelty_score",
    "local_unseen_count_radius2",
    "local_new_template_count_radius2",
    "local_template_switch_count_radius2",
    "prev_template_same",
    "next_template_same",
    "relative_position_in_window",
    "distance_to_window_edge_ratio",
]
LABEL_LIKE_KEYS = {
    "label",
    "labels",
    "is_anomaly",
    "anomaly",
    "ground_truth",
    "positive_events",
    "precision",
    "recall",
    "f1",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build label-free JSONL packets for local LLM review of event-level candidates."
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/event_level_candidates_bgl_multiblock.csv",
    )
    parser.add_argument("--dataset", default="bgl_multiblock")
    parser.add_argument(
        "--selection-method",
        default="template_burst",
        choices=DEFAULT_METHODS,
    )
    parser.add_argument("--top-k", type=int, default=500)
    parser.add_argument(
        "--prompt-strategy",
        default="event_explanation_v1",
        help="Logical strategy id to store in each packet.",
    )
    parser.add_argument(
        "--same-template-examples",
        type=int,
        default=2,
        help="Number of earlier same-template examples to include without labels.",
    )
    parser.add_argument(
        "--neighbor-events",
        type=int,
        default=1,
        help="Number of previous and next candidate-table events to include without labels.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="outputs/reports/llm_review_packets_bgl_multiblock.jsonl",
    )
    parser.add_argument(
        "--output-report",
        default="outputs/reports/llm_review_packets_bgl_multiblock.json",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    packets = build_review_packets(
        df,
        dataset=args.dataset,
        selection_method=args.selection_method,
        top_k=args.top_k,
        prompt_strategy=args.prompt_strategy,
        same_template_examples=args.same_template_examples,
        neighbor_events=args.neighbor_events,
    )

    output_jsonl = Path(args.output_jsonl)
    output_report = Path(args.output_report)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(packets, output_jsonl)
    report = build_packet_report(
        packets,
        input_csv=args.input_csv,
        output_jsonl=str(output_jsonl),
        dataset=args.dataset,
        selection_method=args.selection_method,
        top_k=args.top_k,
        prompt_strategy=args.prompt_strategy,
    )
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_review_packets(
    df: pd.DataFrame,
    *,
    dataset: str,
    selection_method: str,
    top_k: int,
    prompt_strategy: str,
    same_template_examples: int = 2,
    neighbor_events: int = 1,
) -> list[dict[str, Any]]:
    ranked = rank_events(df, selection_method).head(top_k).copy()
    packets = []
    for selection_rank, (index, row) in enumerate(ranked.iterrows(), start=1):
        packet = _packet_from_row(
            df,
            row,
            source_index=index,
            dataset=dataset,
            selection_method=selection_method,
            selection_rank=selection_rank,
            prompt_strategy=prompt_strategy,
            same_template_examples=same_template_examples,
            neighbor_events=neighbor_events,
        )
        assert_no_label_leakage(packet)
        packets.append(packet)
    return packets


def write_jsonl(packets: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def build_packet_report(
    packets: list[dict[str, Any]],
    *,
    input_csv: str,
    output_jsonl: str,
    dataset: str,
    selection_method: str,
    top_k: int,
    prompt_strategy: str,
) -> dict[str, Any]:
    templates = {packet.get("template") for packet in packets}
    blocks = {packet.get("block_id") for packet in packets}
    return {
        "packet_schema_version": PACKET_SCHEMA_VERSION,
        "input_csv": input_csv,
        "output_jsonl": output_jsonl,
        "dataset": dataset,
        "selection_method": selection_method,
        "top_k": top_k,
        "prompt_strategy": prompt_strategy,
        "num_packets": len(packets),
        "unique_templates": len({item for item in templates if item is not None}),
        "unique_blocks": len({item for item in blocks if item is not None}),
        "packet_sha256": _sha256_json(packets),
        "label_policy": "labels and evaluation metrics are excluded from LLM packets",
        "notes": [
            "Packets are intended for local no-thinking LLM triage.",
            "Labels remain available only in the source table for offline evaluation.",
            "Use prompts/log_triage_event_v1.md for the first event-level experiment.",
        ],
    }


def assert_no_label_leakage(value: Any, path: str = "packet") -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            normalized = str(key).lower()
            if normalized in LABEL_LIKE_KEYS or normalized.startswith("label_"):
                raise ValueError(f"Label-like key found in packet at {path}.{key}")
            assert_no_label_leakage(nested, f"{path}.{key}")
    elif isinstance(value, list):
        for offset, nested in enumerate(value):
            assert_no_label_leakage(nested, f"{path}[{offset}]")


def _packet_from_row(
    df: pd.DataFrame,
    row: pd.Series,
    *,
    source_index: Any,
    dataset: str,
    selection_method: str,
    selection_rank: int,
    prompt_strategy: str,
    same_template_examples: int,
    neighbor_events: int,
) -> dict[str, Any]:
    event_id = _as_int(row.get("event_id"))
    packet_id = f"{dataset}:{selection_method}:{selection_rank:06d}:event:{event_id}"
    return {
        "packet_schema_version": PACKET_SCHEMA_VERSION,
        "packet_id": packet_id,
        "dataset": dataset,
        "selection_method": selection_method,
        "selection_rank": selection_rank,
        "classical_score": _clean_value(row.get("classical_score")),
        "event_id": event_id,
        "event_order": _as_int(row.get("event_order")),
        "block_id": _clean_value(row.get("block_id")),
        "template": _clean_text(row.get("template")),
        "message": _clean_text(row.get("message")),
        "scores": _pick_columns(row, DEFAULT_SCORE_COLUMNS),
        "window_context": _pick_columns(row, DEFAULT_WINDOW_CONTEXT_COLUMNS),
        "local_context": _pick_columns(row, DEFAULT_LOCAL_CONTEXT_COLUMNS),
        "related_events": _related_events(
            df,
            row,
            source_index=source_index,
            same_template_examples=same_template_examples,
            neighbor_events=neighbor_events,
        ),
        "prompt_strategy": prompt_strategy,
    }


def _related_events(
    df: pd.DataFrame,
    row: pd.Series,
    *,
    source_index: Any,
    same_template_examples: int,
    neighbor_events: int,
) -> dict[str, list[dict[str, Any]]]:
    ordered = _ordered_df(df)
    neighbors = _neighbor_examples(ordered, source_index, radius=neighbor_events)
    same_template = _same_template_examples(
        ordered,
        row,
        source_index=source_index,
        limit=same_template_examples,
    )
    return {
        "neighbor_events": neighbors,
        "same_template_examples": same_template,
    }


def _ordered_df(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ["block_id", "event_order", "event_id"] if column in df.columns]
    if not sort_columns:
        return df
    return df.sort_values(sort_columns)


def _neighbor_examples(ordered: pd.DataFrame, source_index: Any, *, radius: int) -> list[dict[str, Any]]:
    if radius <= 0 or source_index not in ordered.index:
        return []
    positions = pd.Series(range(len(ordered)), index=ordered.index)
    position = int(positions.loc[source_index])
    start = max(0, position - radius)
    end = min(len(ordered), position + radius + 1)
    examples = []
    for index, row in ordered.iloc[start:end].iterrows():
        if index == source_index:
            continue
        examples.append(_compact_event(row, relation="neighbor"))
    return examples


def _same_template_examples(
    ordered: pd.DataFrame,
    row: pd.Series,
    *,
    source_index: Any,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or "template" not in ordered.columns:
        return []
    template = row.get("template")
    same_template = ordered[(ordered["template"] == template) & (ordered.index != source_index)]
    if "event_order" in same_template.columns and pd.notna(row.get("event_order")):
        earlier = same_template[same_template["event_order"] < row.get("event_order")]
        if not earlier.empty:
            same_template = earlier
    same_template = same_template.tail(limit)
    return [_compact_event(example, relation="same_template") for _, example in same_template.iterrows()]


def _compact_event(row: pd.Series, *, relation: str) -> dict[str, Any]:
    return {
        "relation": relation,
        "event_id": _as_int(row.get("event_id")),
        "event_order": _as_int(row.get("event_order")),
        "block_id": _clean_value(row.get("block_id")),
        "template": _clean_text(row.get("template")),
        "message": _clean_text(row.get("message")),
        "scores": _pick_columns(row, ["template_burst_score", "novelty_score", "rarity_score"]),
    }


def _pick_columns(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    values = {}
    for column in columns:
        if column in row.index:
            value = _clean_value(row.get(column))
            if value is not None:
                values[column] = value
    return values


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


def _clean_text(value: Any) -> str:
    cleaned = _clean_value(value)
    if cleaned is None:
        return ""
    return str(cleaned)


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    main()
