from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_log_retrieval_index import DEFAULT_MODEL_PATH
from scripts.build_llm_review_packets import assert_no_label_leakage
from scripts.run_llm_triage import read_jsonl


RETRIEVAL_CONTEXT_SCHEMA_VERSION = "retrieval_context_v1"
SCORE_COLUMNS = [
    "template_burst_score",
    "template_count_deviation_score",
    "novelty_score",
    "rarity_score",
    "historical_count_log",
    "local_sequence_context_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Augment LLM cluster packets with retrieval-lite examples from a local embedding index."
    )
    parser.add_argument(
        "--input-jsonl",
        default="outputs/reports/llm_cluster_packets_bgl_multiblock.jsonl",
    )
    parser.add_argument(
        "--index-meta",
        default="outputs/reports/log_retrieval_index_meta.csv",
    )
    parser.add_argument(
        "--index-embeddings",
        default="outputs/models/log_retrieval_index_embeddings.npy",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--nearest-k", type=int, default=4)
    parser.add_argument("--same-template-k", type=int, default=4)
    parser.add_argument("--low-score-k", type=int, default=4)
    parser.add_argument("--max-query-events", type=int, default=6)
    parser.add_argument(
        "--max-representative-events",
        type=int,
        default=4,
        help="Maximum representative events retained in the augmented packet.",
    )
    parser.add_argument(
        "--max-member-ids",
        type=int,
        default=1000,
        help="Maximum member event ids/ranks retained for evaluation expansion.",
    )
    parser.add_argument(
        "--max-message-chars",
        type=int,
        default=240,
        help="Maximum characters retained per log message in LLM-facing packet fields.",
    )
    parser.add_argument(
        "--drop-nearest-examples",
        action="store_true",
        help="Omit general nearest-neighbor examples and keep only template/score contrast.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="outputs/reports/llm_cluster_packets_bgl_multiblock_retrieval.jsonl",
    )
    parser.add_argument(
        "--output-report",
        default="outputs/reports/llm_cluster_packets_bgl_multiblock_retrieval.json",
    )
    args = parser.parse_args()

    packets = read_jsonl(Path(args.input_jsonl))
    meta = pd.read_csv(args.index_meta)
    embeddings = np.load(args.index_embeddings).astype(np.float32)
    if len(meta) != len(embeddings):
        raise ValueError(f"Index metadata rows ({len(meta)}) do not match embeddings ({len(embeddings)}).")

    query_texts = [build_cluster_query_text(packet, max_query_events=args.max_query_events) for packet in packets]
    query_embeddings = encode_queries(query_texts, model_path=args.model_path)
    augmented = []
    for packet, query_embedding in zip(packets, query_embeddings, strict=True):
        updated = compact_cluster_packet(
            packet,
            max_representative_events=args.max_representative_events,
            max_member_ids=args.max_member_ids,
            max_message_chars=args.max_message_chars,
        )
        updated["retrieved_context"] = retrieve_context_for_packet(
            packet,
            meta,
            embeddings,
            query_embedding,
            nearest_k=args.nearest_k,
            same_template_k=args.same_template_k,
            low_score_k=args.low_score_k,
            max_message_chars=args.max_message_chars,
            include_nearest=not args.drop_nearest_examples,
        )
        assert_no_label_leakage(updated)
        augmented.append(updated)

    output_jsonl = Path(args.output_jsonl)
    output_report = Path(args.output_report)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(augmented, output_jsonl)
    report = {
        "retrieval_context_schema_version": RETRIEVAL_CONTEXT_SCHEMA_VERSION,
        "input_jsonl": args.input_jsonl,
        "index_meta": args.index_meta,
        "index_embeddings": args.index_embeddings,
        "model_path": args.model_path,
        "num_packets": len(augmented),
        "nearest_k": args.nearest_k,
        "same_template_k": args.same_template_k,
        "low_score_k": args.low_score_k,
        "max_query_events": args.max_query_events,
        "max_representative_events": args.max_representative_events,
        "max_member_ids": args.max_member_ids,
        "max_message_chars": args.max_message_chars,
        "drop_nearest_examples": args.drop_nearest_examples,
        "mean_packet_json_chars": _mean_json_chars(augmented),
        "max_packet_json_chars": _max_json_chars(augmented),
        "output_jsonl": str(output_jsonl),
        "packet_sha256": _sha256_json(augmented),
        "label_policy": "retrieved context excludes labels and evaluation metrics",
    }
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def retrieve_context_for_packet(
    packet: dict[str, Any],
    meta: pd.DataFrame,
    embeddings: np.ndarray,
    query_embedding: np.ndarray,
    *,
    nearest_k: int,
    same_template_k: int,
    low_score_k: int,
    max_message_chars: int = 240,
    include_nearest: bool = True,
) -> dict[str, Any]:
    member_ids = set(packet.get("cluster", {}).get("member_event_ids", []))
    template = packet.get("cluster", {}).get("template")
    similarities = embeddings @ query_embedding.astype(np.float32)
    candidates = meta.copy()
    candidates["similarity"] = similarities
    if "event_id" in candidates:
        candidates = candidates[~candidates["event_id"].isin(member_ids)].copy()

    same_template = candidates[candidates["template"] == template].copy() if "template" in candidates else candidates.iloc[0:0].copy()
    cluster_burst_mean = _cluster_score_mean(packet, "template_burst_score")
    lower_score = candidates.copy()
    if cluster_burst_mean is not None and "template_burst_score" in lower_score:
        lower_score = lower_score[lower_score["template_burst_score"] < cluster_burst_mean].copy()

    context = {
        "retrieval_context_schema_version": RETRIEVAL_CONTEXT_SCHEMA_VERSION,
        "nearest_examples": (
            examples_from_frame(
                candidates.sort_values("similarity", ascending=False).head(nearest_k),
                max_message_chars=max_message_chars,
            )
            if include_nearest
            else []
        ),
        "same_template_examples": examples_from_frame(
            same_template.sort_values("similarity", ascending=False).head(same_template_k),
            max_message_chars=max_message_chars,
        ),
        "lower_score_contrast_examples": examples_from_frame(
            lower_score.sort_values("similarity", ascending=False).head(low_score_k),
            max_message_chars=max_message_chars,
        ),
        "notes": [
            "Examples are retrieved without labels.",
            "Lower-score examples are contrastive context, not guaranteed normal examples.",
        ],
    }
    assert_no_label_leakage(context)
    return context


def compact_cluster_packet(
    packet: dict[str, Any],
    *,
    max_representative_events: int,
    max_member_ids: int,
    max_message_chars: int,
) -> dict[str, Any]:
    compacted = dict(packet)
    compacted["representative_events"] = [
        compact_event(event, max_message_chars=max_message_chars)
        for event in packet.get("representative_events", [])[:max_representative_events]
    ]
    cluster = dict(packet.get("cluster", {}))
    if max_member_ids > 0:
        cluster["member_event_ids"] = list(cluster.get("member_event_ids", []))[:max_member_ids]
        cluster["member_selection_ranks"] = list(cluster.get("member_selection_ranks", []))[:max_member_ids]
        cluster["member_ids_truncated"] = int(cluster.get("event_count") or 0) > max_member_ids
    else:
        cluster["member_event_ids"] = list(cluster.get("member_event_ids", []))
        cluster["member_selection_ranks"] = list(cluster.get("member_selection_ranks", []))
        cluster["member_ids_truncated"] = False
    compacted["cluster"] = cluster
    return compacted


def compact_event(event: dict[str, Any], *, max_message_chars: int) -> dict[str, Any]:
    compacted = dict(event)
    if "message" in compacted:
        compacted["message"] = truncate_text(str(compacted["message"]), max_message_chars)
    return compacted


def build_cluster_query_text(packet: dict[str, Any], *, max_query_events: int) -> str:
    cluster = packet.get("cluster", {})
    parts = [
        f"Template: {cluster.get('template', '')}",
        "Representative messages:",
    ]
    for event in packet.get("representative_events", [])[:max_query_events]:
        parts.append(str(event.get("message", "")))
    return "\n".join(parts)


def encode_queries(texts: list[str], *, model_path: str) -> np.ndarray:
    from scripts.build_log_retrieval_index import _load_sentence_transformer

    model = _load_sentence_transformer(model_path)
    embeddings = model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)


def examples_from_frame(df: pd.DataFrame, *, max_message_chars: int = 240) -> list[dict[str, Any]]:
    examples = []
    for _, row in df.iterrows():
        example = {
            "event_id": _as_int(row.get("event_id")),
            "event_order": _as_int(row.get("event_order")),
            "block_id": _clean_value(row.get("block_id")),
            "template": _clean_text(row.get("template")),
            "message": truncate_text(_clean_text(row.get("message")), max_message_chars),
            "similarity": _as_float(row.get("similarity")),
            "scores": {},
        }
        for column in SCORE_COLUMNS:
            if column in row.index:
                value = _clean_value(row.get(column))
                if value is not None:
                    example["scores"][column] = value
        examples.append(example)
    return examples


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    suffix = "...[truncated]"
    return text[: max(0, max_chars - len(suffix))].rstrip() + suffix


def write_jsonl(packets: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _cluster_score_mean(packet: dict[str, Any], score_name: str) -> float | None:
    summary = packet.get("score_summary", {}).get(score_name, {})
    value = summary.get("mean")
    if value is None:
        return None
    return float(value)


def _clean_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


def _clean_text(value: Any) -> str:
    cleaned = _clean_value(value)
    if cleaned is None:
        return ""
    return str(cleaned)


def _as_int(value: Any) -> int | None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    return int(cleaned)


def _as_float(value: Any) -> float | None:
    cleaned = _clean_value(value)
    if cleaned is None:
        return None
    return float(cleaned)


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _mean_json_chars(values: list[dict[str, Any]]) -> float:
    if not values:
        return 0.0
    sizes = [_json_chars(value) for value in values]
    return sum(sizes) / len(sizes)


def _max_json_chars(values: list[dict[str, Any]]) -> int:
    if not values:
        return 0
    return max(_json_chars(value) for value in values)


def _json_chars(value: dict[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
