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

from scripts.analyze_anomaly_clusters import _deduplicate_events, _infer_totals, _safe_div, _safe_ratio  # noqa: E402
from scripts.analyze_label_free_incident_clusters import merge_intervals  # noqa: E402
from scripts.build_llm_review_packets import LABEL_LIKE_KEYS, assert_no_label_leakage  # noqa: E402
from scripts.run_classical_triage_baseline import rank_events  # noqa: E402


PACKET_SCHEMA_VERSION = "streaming_incident_packet_v1"
DEFAULT_SCORE_COLUMNS = [
    "template_burst_score",
    "novelty_score",
    "rarity_score",
    "template_count_deviation_score",
    "local_sequence_context_score",
    "markov_bigram_surprise",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate production-like streaming incident buffers before asynchronous LLM review."
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/event_level_candidates_bgl_multiblock.csv",
    )
    parser.add_argument(
        "--event-cv-json",
        default="outputs/reports/event_level_cv_bgl_multiblock.json",
    )
    parser.add_argument(
        "--true-clusters-csv",
        default="outputs/reports/anomaly_cluster_density_clusters.csv",
    )
    parser.add_argument("--dataset", default="bgl_multiblock")
    parser.add_argument("--total-stream-events", type=int, default=None)
    parser.add_argument("--total-anomalies", type=int, default=None)
    parser.add_argument(
        "--calibration-fraction",
        type=float,
        default=0.20,
        help="Initial fraction of ordered candidate events used to set label-free score thresholds.",
    )
    parser.add_argument("--dense-quantile", type=float, default=0.98)
    parser.add_argument("--diversity-quantile", type=float, default=0.98)
    parser.add_argument("--enable-dense-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-diversity-channel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-gap", type=int, default=50)
    parser.add_argument("--context-before", type=int, default=10)
    parser.add_argument("--context-after", type=int, default=10)
    parser.add_argument("--close-after", type=int, default=50)
    parser.add_argument("--max-representative-events", type=int, default=8)
    parser.add_argument("--max-packets", type=int, default=200)
    parser.add_argument(
        "--output-regions-csv",
        default="outputs/reports/streaming_incident_regions.csv",
    )
    parser.add_argument(
        "--output-packets-jsonl",
        default="outputs/reports/streaming_incident_packets.jsonl",
    )
    parser.add_argument(
        "--output-report-json",
        default="outputs/reports/streaming_incident_triage_report.json",
    )
    parser.add_argument(
        "--output-report-md",
        default="outputs/reports/streaming_incident_triage_report.md",
    )
    args = parser.parse_args()

    df = _deduplicate_events(pd.read_csv(args.input_csv))
    _validate_input(df)
    df = _enrich_markov_scores(df)
    df = df.sort_values(["block_id", "event_order", "event_id"] if "block_id" in df else ["event_order", "event_id"])
    totals = _infer_totals(
        df,
        event_cv_json=Path(args.event_cv_json),
        total_stream_events=args.total_stream_events,
        total_anomalies=args.total_anomalies,
    )
    true_clusters = _read_true_clusters(Path(args.true_clusters_csv))
    thresholds = _calibrate_thresholds(
        df,
        calibration_fraction=args.calibration_fraction,
        dense_quantile=args.dense_quantile,
        diversity_quantile=args.diversity_quantile,
    )
    regions = simulate_stream(
        df,
        thresholds=thresholds,
        enable_dense_channel=args.enable_dense_channel,
        enable_diversity_channel=args.enable_diversity_channel,
        max_gap=args.max_gap,
        context_before=args.context_before,
        context_after=args.context_after,
        close_after=args.close_after,
    )
    metrics = evaluate_streaming_regions(regions, df, true_clusters, totals=totals)
    packets = build_incident_packets(
        regions,
        df,
        dataset=args.dataset,
        max_packets=args.max_packets,
        max_representative_events=args.max_representative_events,
        thresholds=thresholds,
    )

    report = {
        "dataset": args.dataset,
        "input_csv": args.input_csv,
        "population": "streaming simulation over deduplicated GEV-suspicious candidate events",
        "totals": totals,
        "parameters": {
            "calibration_fraction": args.calibration_fraction,
            "dense_quantile": args.dense_quantile,
            "diversity_quantile": args.diversity_quantile,
            "enable_dense_channel": args.enable_dense_channel,
            "enable_diversity_channel": args.enable_diversity_channel,
            "max_gap": args.max_gap,
            "context_before": args.context_before,
            "context_after": args.context_after,
            "close_after": args.close_after,
            "max_representative_events": args.max_representative_events,
            "max_packets": args.max_packets,
        },
        "thresholds": thresholds,
        "metrics": metrics,
        "llm_packet_plan": {
            "packets_written": len(packets),
            "mode": "asynchronous incident-level review",
            "labels_in_packets": False,
            "estimated_llm_calls": len(packets),
        },
        "outputs": {
            "regions_csv": args.output_regions_csv,
            "packets_jsonl": args.output_packets_jsonl,
            "report_json": args.output_report_json,
            "report_md": args.output_report_md,
        },
        "notes": [
            "This simulates the post-GEV streaming incident buffer, not full raw-stream online GEV fitting.",
            "Thresholds are calibrated from an initial ordered prefix without using labels.",
            "LLM calls are asynchronous and incident-level; no LLM is called per event or per window.",
            "Labels are used only after region emission for offline metrics.",
        ],
    }

    paths = [
        Path(args.output_regions_csv),
        Path(args.output_packets_jsonl),
        Path(args.output_report_json),
        Path(args.output_report_md),
    ]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    regions.to_csv(args.output_regions_csv, index=False)
    write_jsonl(packets, Path(args.output_packets_jsonl))
    Path(args.output_report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, Path(args.output_report_md))
    print(json.dumps(report, indent=2))


def simulate_stream(
    df: pd.DataFrame,
    *,
    thresholds: dict[str, float],
    enable_dense_channel: bool,
    enable_diversity_channel: bool,
    max_gap: int,
    context_before: int,
    context_after: int,
    close_after: int,
) -> pd.DataFrame:
    active: dict[Any, dict[str, Any]] = {}
    emitted: list[dict[str, Any]] = []
    region_id = 0
    for _, row in df.iterrows():
        block_id = _clean(row.get("block_id", "__global__"))
        event_order = int(row["event_order"])
        stale = [
            key
            for key, incident in active.items()
            if key == block_id and event_order - int(incident["last_seed_order"]) > close_after
        ]
        for key in stale:
            region_id += 1
            emitted.append(_finalize_incident(active.pop(key), region_id, context_before, context_after))

        fired = _trigger_reasons(
            row,
            thresholds,
            enable_dense_channel=enable_dense_channel,
            enable_diversity_channel=enable_diversity_channel,
        )
        if not fired:
            continue
        incident = active.get(block_id)
        if incident is None or event_order - int(incident["last_seed_order"]) > max_gap:
            if incident is not None:
                region_id += 1
                emitted.append(_finalize_incident(incident, region_id, context_before, context_after))
            active[block_id] = _new_incident(row, fired)
        else:
            _update_incident(incident, row, fired)

    for key in sorted(active):
        region_id += 1
        emitted.append(_finalize_incident(active[key], region_id, context_before, context_after))
    if not emitted:
        return pd.DataFrame()
    return pd.DataFrame(emitted).sort_values(["block_id", "start_event_order", "end_event_order"])


def evaluate_streaming_regions(
    regions: pd.DataFrame,
    df: pd.DataFrame,
    true_clusters: pd.DataFrame,
    *,
    totals: dict[str, int | str | None],
) -> dict[str, Any]:
    total_stream_events = int(totals["total_stream_events"])
    total_anomalies = int(totals["total_anomalies"])
    merged = merge_intervals(regions) if not regions.empty else pd.DataFrame(columns=["block_id", "interval_start", "interval_end"])
    mask = _rows_in_intervals(df, merged)
    covered = df[mask]
    interval_events = int(
        sum(int(row["interval_end"]) - int(row["interval_start"]) + 1 for _, row in merged.iterrows())
    )
    covered_anomalies = int(covered["label"].sum()) if "label" in covered else 0
    true_hits, delays = _true_cluster_hits_and_delays(true_clusters, regions)
    anomaly_recall = _safe_ratio(covered_anomalies, total_anomalies)
    stream_load = _safe_ratio(interval_events, total_stream_events)
    return {
        "regions_emitted": int(len(regions)),
        "merged_regions": int(len(merged)),
        "llm_call_count": int(len(regions)),
        "llm_calls_per_10000_events": _safe_ratio(len(regions) * 10000, total_stream_events),
        "interval_events": interval_events,
        "stream_load": stream_load,
        "covered_candidate_events": int(len(covered)),
        "covered_anomaly_events": covered_anomalies,
        "anomaly_recall_against_all": anomaly_recall,
        "region_weighted_density": _safe_ratio(covered_anomalies, interval_events),
        "compression_ratio": _safe_div(anomaly_recall, stream_load),
        "true_clusters": int(len(true_clusters)),
        "true_clusters_hit": int(true_hits),
        "true_cluster_recall": _safe_ratio(true_hits, len(true_clusters)),
        "mean_detection_delay_events": sum(delays) / len(delays) if delays else None,
        "median_detection_delay_events": float(pd.Series(delays).median()) if delays else None,
        "max_detection_delay_events": max(delays) if delays else None,
    }


def build_incident_packets(
    regions: pd.DataFrame,
    df: pd.DataFrame,
    *,
    dataset: str,
    max_packets: int,
    max_representative_events: int,
    thresholds: dict[str, float],
) -> list[dict[str, Any]]:
    packets = []
    if regions.empty:
        return packets
    selected_regions = regions.head(max_packets)
    for _, region in selected_regions.iterrows():
        region_events = _events_in_region(df, region)
        representative = _representative_events(region_events, max_events=max_representative_events)
        packet = {
            "packet_schema_version": PACKET_SCHEMA_VERSION,
            "packet_id": f"{dataset}:streaming_incident:{int(region['region_id']):04d}",
            "dataset": dataset,
            "prompt_strategy": "streaming_incident_review_v1",
            "runtime_mode": "asynchronous_incident_review",
            "thresholds": {key: float(value) for key, value in thresholds.items()},
            "region": {
                "region_id": int(region["region_id"]),
                "block_id": _clean(region["block_id"]),
                "start_event_order": int(region["start_event_order"]),
                "end_event_order": int(region["end_event_order"]),
                "interval_start": int(region["interval_start"]),
                "interval_end": int(region["interval_end"]),
                "emit_event_order": int(region["emit_event_order"]),
                "seed_events": int(region["seed_events"]),
                "trigger_reasons": _split_reasons(region["trigger_reasons"]),
                "dominant_template": _clean(region.get("dominant_template")),
            },
            "score_summary": _score_summary(region_events),
            "representative_events": representative,
            "instructions": [
                "Review this label-free incident packet.",
                "Ground the decision only in provided events, templates, scores, and context.",
                "Return strict JSON with incident decision, priority, evidence_event_ids, reason_codes, and concise summary.",
            ],
        }
        assert_no_label_leakage(packet)
        packets.append(packet)
    return packets


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    m = report["metrics"]
    lines = [
        "# Streaming Incident Triage Simulation",
        "",
        "This report simulates a production-like post-GEV incident buffer. Events are processed in event order, suspicious seeds open or update incident buffers, and closed buffers become asynchronous LLM incident packets.",
        "",
        "## Protocol",
        "",
        f"- Input: `{report['input_csv']}`",
        f"- Calibration fraction: `{report['parameters']['calibration_fraction']}`",
        f"- Dense quantile: `{report['parameters']['dense_quantile']}`",
        f"- Diversity quantile: `{report['parameters']['diversity_quantile']}`",
        f"- Max seed gap: `{report['parameters']['max_gap']}` events",
        f"- Close after: `{report['parameters']['close_after']}` events without a new seed",
        f"- Context before/after: `{report['parameters']['context_before']}` / `{report['parameters']['context_after']}` events",
        "",
        "## Calibrated Thresholds",
        "",
        "| Score | Threshold |",
        "| --- | ---: |",
    ]
    for key, value in report["thresholds"].items():
        lines.append(f"| {key} | {_fmt(value)} |")
    lines.extend(
        [
            "",
            "## Main Metrics",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
            f"| Regions emitted / LLM calls | {m['regions_emitted']} |",
            f"| LLM calls per 10k stream events | {_fmt(m['llm_calls_per_10000_events'])} |",
            f"| Stream load | {_fmt(m['stream_load'])} |",
            f"| Covered anomaly events | {m['covered_anomaly_events']} |",
            f"| Anomaly recall | {_fmt(m['anomaly_recall_against_all'])} |",
            f"| True cluster recall | {_fmt(m['true_cluster_recall'])} |",
            f"| Region weighted density | {_fmt(m['region_weighted_density'])} |",
            f"| Compression ratio | {_fmt(m['compression_ratio'])} |",
            f"| Median detection delay | {_fmt(m['median_detection_delay_events'])} events |",
            "",
            "## Interpretation",
            "",
            "- The synchronous part is cheap: score triggers and incident-buffer updates.",
            "- The LLM is asynchronous and called once per emitted incident region, not once per log or window.",
            "- Labels are used only after incident emission for metrics.",
            "- This is a post-GEV streaming simulation; full raw-stream deployment would integrate the same buffer with online Drain and rolling-window GEV scoring.",
            "",
            "## Outputs",
            "",
        ]
    )
    for name, value in report["outputs"].items():
        lines.append(f"- `{name}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_jsonl(packets: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def _calibrate_thresholds(
    df: pd.DataFrame,
    *,
    calibration_fraction: float,
    dense_quantile: float,
    diversity_quantile: float,
) -> dict[str, float]:
    if not 0 < calibration_fraction <= 1:
        raise ValueError("--calibration-fraction must be in (0, 1].")
    cutoff = max(1, int(len(df) * calibration_fraction))
    calibration = df.sort_values(["event_order", "event_id"]).head(cutoff)
    return {
        "template_burst_score": float(calibration["template_burst_score"].fillna(0).quantile(dense_quantile)),
        "novelty_score": float(calibration["novelty_score"].fillna(0).quantile(dense_quantile)),
        "markov_bigram_surprise": float(calibration["markov_bigram_surprise"].fillna(0).quantile(diversity_quantile)),
    }


def _trigger_reasons(
    row: pd.Series,
    thresholds: dict[str, float],
    *,
    enable_dense_channel: bool,
    enable_diversity_channel: bool,
) -> list[str]:
    reasons = []
    if enable_dense_channel:
        if float(row.get("template_burst_score", 0.0) or 0.0) >= thresholds["template_burst_score"]:
            reasons.append("dense_template_burst")
        if float(row.get("novelty_score", 0.0) or 0.0) >= thresholds["novelty_score"]:
            reasons.append("dense_novelty")
    if enable_diversity_channel and float(row.get("markov_bigram_surprise", 0.0) or 0.0) >= thresholds["markov_bigram_surprise"]:
        reasons.append("diversity_markov_bigram")
    return reasons


def _new_incident(row: pd.Series, reasons: list[str]) -> dict[str, Any]:
    event_order = int(row["event_order"])
    return {
        "block_id": _clean(row.get("block_id", "__global__")),
        "first_seed_order": event_order,
        "last_seed_order": event_order,
        "emit_event_order": event_order,
        "seed_events": [row.to_dict()],
        "trigger_reasons": set(reasons),
    }


def _update_incident(incident: dict[str, Any], row: pd.Series, reasons: list[str]) -> None:
    event_order = int(row["event_order"])
    incident["last_seed_order"] = event_order
    incident["emit_event_order"] = event_order
    incident["seed_events"].append(row.to_dict())
    incident["trigger_reasons"].update(reasons)


def _finalize_incident(
    incident: dict[str, Any],
    region_id: int,
    context_before: int,
    context_after: int,
) -> dict[str, Any]:
    events = incident["seed_events"]
    start = int(incident["first_seed_order"])
    end = int(incident["last_seed_order"])
    templates = [str(event.get("template")) for event in events if event.get("template") is not None]
    return {
        "region_id": region_id,
        "block_id": incident["block_id"],
        "start_event_order": start,
        "end_event_order": end,
        "interval_start": max(0, start - context_before),
        "interval_end": end + context_after,
        "emit_event_order": int(incident["emit_event_order"]) + context_after,
        "seed_events": len(events),
        "trigger_reasons": ",".join(sorted(incident["trigger_reasons"])),
        "unique_seed_templates": len(set(templates)),
        "dominant_template": _mode(templates),
        "max_template_burst_score": max(float(event.get("template_burst_score", 0.0) or 0.0) for event in events),
        "max_novelty_score": max(float(event.get("novelty_score", 0.0) or 0.0) for event in events),
        "max_markov_bigram_surprise": max(float(event.get("markov_bigram_surprise", 0.0) or 0.0) for event in events),
    }


def _true_cluster_hits_and_delays(true_clusters: pd.DataFrame, regions: pd.DataFrame) -> tuple[int, list[int]]:
    if true_clusters.empty or regions.empty:
        return 0, []
    hits = 0
    delays = []
    for _, truth in true_clusters.iterrows():
        overlaps = regions[
            (regions["block_id"] == truth["block_id"])
            & (regions["interval_start"] <= int(truth["interval_end"]))
            & (regions["interval_end"] >= int(truth["interval_start"]))
        ]
        if overlaps.empty:
            continue
        hits += 1
        first_region = overlaps.sort_values("start_event_order").iloc[0]
        delays.append(int(first_region["start_event_order"]) - int(truth["start_event_order"]))
    return hits, delays


def _rows_in_intervals(df: pd.DataFrame, intervals: pd.DataFrame) -> pd.Series:
    mask = pd.Series([False] * len(df), index=df.index)
    if intervals.empty:
        return mask
    for _, interval in intervals.iterrows():
        block_mask = df["block_id"] == interval["block_id"] if "block_id" in df.columns else True
        mask = mask | (
            block_mask
            & (df["event_order"] >= int(interval["interval_start"]))
            & (df["event_order"] <= int(interval["interval_end"]))
        )
    return mask


def _events_in_region(df: pd.DataFrame, region: pd.Series) -> pd.DataFrame:
    block_mask = df["block_id"] == region["block_id"] if "block_id" in df.columns else True
    return df[
        block_mask
        & (df["event_order"] >= int(region["interval_start"]))
        & (df["event_order"] <= int(region["interval_end"]))
    ].copy()


def _representative_events(events: pd.DataFrame, *, max_events: int) -> list[dict[str, Any]]:
    if events.empty or max_events <= 0:
        return []
    sort_columns = [column for column in ["template_burst_score", "novelty_score", "markov_bigram_surprise"] if column in events]
    chosen = events.sort_values(sort_columns, ascending=[False] * len(sort_columns)).head(max_events)
    rows = []
    for _, row in chosen.sort_values("event_order").iterrows():
        item = {
            "event_id": _as_int(row.get("event_id")),
            "event_order": _as_int(row.get("event_order")),
            "block_id": _clean(row.get("block_id")),
            "template": _clean(row.get("template")),
            "message": _clean(row.get("message")),
            "scores": {
                column: float(row.get(column, 0.0) or 0.0)
                for column in DEFAULT_SCORE_COLUMNS
                if column in row.index
            },
        }
        _assert_no_label_like_keys(item)
        rows.append(item)
    return rows


def _score_summary(events: pd.DataFrame) -> dict[str, dict[str, float]]:
    summary = {}
    for column in DEFAULT_SCORE_COLUMNS:
        if column not in events:
            continue
        values = pd.to_numeric(events[column], errors="coerce").dropna()
        if values.empty:
            continue
        summary[column] = {
            "min": float(values.min()),
            "mean": float(values.mean()),
            "max": float(values.max()),
        }
    return summary


def _enrich_markov_scores(df: pd.DataFrame) -> pd.DataFrame:
    if "markov_bigram_surprise" in df:
        return df
    enriched = df.copy()
    scored = rank_events(enriched, "markov_bigram_surprise")
    enriched["markov_bigram_surprise"] = scored["classical_score"].reindex(enriched.index).fillna(0.0)
    scored_trigram = rank_events(enriched, "markov_trigram_surprise")
    enriched["markov_trigram_surprise"] = scored_trigram["classical_score"].reindex(enriched.index).fillna(0.0)
    return enriched


def _read_true_clusters(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    clusters = pd.read_csv(path)
    if "is_non_singleton_cluster" in clusters:
        clusters = clusters[clusters["is_non_singleton_cluster"].astype(bool)].copy()
    return clusters


def _split_reasons(raw: Any) -> list[str]:
    return [item for item in str(raw).split(",") if item]


def _validate_input(df: pd.DataFrame) -> None:
    required = {"event_order", "event_id", "label", "template", "message", "template_burst_score", "novelty_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")


def _mode(values: list[str]) -> str | None:
    if not values:
        return None
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _clean(value: Any) -> Any:
    if value is None:
        return None
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    return value


def _as_int(value: Any) -> int | None:
    cleaned = _clean(value)
    if cleaned is None:
        return None
    return int(cleaned)


def _assert_no_label_like_keys(value: dict[str, Any]) -> None:
    for key, item in value.items():
        normalized = str(key).lower()
        if normalized in LABEL_LIKE_KEYS or normalized.startswith("label_"):
            raise ValueError(f"Label-like key found in packet: {key}")
        if isinstance(item, dict):
            _assert_no_label_like_keys(item)
        if isinstance(item, list):
            for subitem in item:
                if isinstance(subitem, dict):
                    _assert_no_label_like_keys(subitem)


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
