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

from scripts.analyze_anomaly_clusters import (  # noqa: E402
    add_cluster_intervals,
    build_gap_clusters,
    _deduplicate_events,
    _infer_totals,
    _parse_int_list,
    _safe_div,
    _safe_ratio,
)
from scripts.run_classical_triage_baseline import DEFAULT_METHODS, rank_events  # noqa: E402


DEFAULT_METHODS_FOR_REGIONS = [
    "template_burst",
    "novelty",
    "markov_bigram_surprise",
    "burst_plus_markov",
]
DEFAULT_TOP_K = [100, 250, 500, 1000, 2500, 5000]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build label-free suspicious event-order regions and evaluate incident-cluster coverage offline."
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/event_level_candidates_bgl_multiblock.csv",
    )
    parser.add_argument(
        "--event-cv-json",
        default="outputs/reports/event_level_cv_bgl_multiblock.json",
    )
    parser.add_argument("--dataset", default="bgl_multiblock")
    parser.add_argument("--total-stream-events", type=int, default=None)
    parser.add_argument("--total-anomalies", type=int, default=None)
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS_FOR_REGIONS),
        help="Comma-separated score methods. Available: " + ", ".join(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--top-k",
        default=",".join(str(item) for item in DEFAULT_TOP_K),
        help="Comma-separated seed budgets used before label-free clustering.",
    )
    parser.add_argument("--seed-max-gap", type=int, default=50)
    parser.add_argument("--region-context-margin", type=int, default=10)
    parser.add_argument("--true-max-gap", type=int, default=50)
    parser.add_argument("--true-context-margin", type=int, default=10)
    parser.add_argument("--min-true-anomalies-per-cluster", type=int, default=2)
    parser.add_argument(
        "--split-by-block",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--output-summary-csv",
        default="outputs/reports/label_free_incident_cluster_summary.csv",
    )
    parser.add_argument(
        "--output-regions-csv",
        default="outputs/reports/label_free_incident_cluster_regions.csv",
    )
    parser.add_argument(
        "--output-report-json",
        default="outputs/reports/label_free_incident_cluster_report.json",
    )
    parser.add_argument(
        "--output-report-md",
        default="outputs/reports/label_free_incident_cluster_report.md",
    )
    args = parser.parse_args()

    df = _deduplicate_events(pd.read_csv(args.input_csv))
    _validate_input(df)
    totals = _infer_totals(
        df,
        event_cv_json=Path(args.event_cv_json),
        total_stream_events=args.total_stream_events,
        total_anomalies=args.total_anomalies,
    )
    methods = _parse_methods(args.methods)
    top_k = _parse_int_list(args.top_k)
    true_clusters = _build_true_clusters(
        df,
        max_gap=args.true_max_gap,
        context_margin=args.true_context_margin,
        min_anomalies=args.min_true_anomalies_per_cluster,
        split_by_block=args.split_by_block,
    )

    summary_rows: list[dict[str, Any]] = []
    region_frames: list[pd.DataFrame] = []
    for method in methods:
        ranked = rank_events(df, method)
        for cutoff in top_k:
            seeds = ranked.head(cutoff).copy()
            regions = build_label_free_regions(
                seeds,
                full_df=df,
                method=method,
                top_k=cutoff,
                max_gap=args.seed_max_gap,
                context_margin=args.region_context_margin,
                split_by_block=args.split_by_block,
            )
            metrics = evaluate_regions(
                regions,
                df,
                true_clusters,
                totals=totals,
                method=method,
                top_k=cutoff,
                seed_events=len(seeds),
            )
            summary_rows.append(metrics)
            if not regions.empty:
                region_frames.append(regions)

    summary = pd.DataFrame(summary_rows)
    regions_all = pd.concat(region_frames, ignore_index=True) if region_frames else pd.DataFrame()
    best_incident = summary.sort_values(
        ["true_cluster_recall", "anomaly_recall_against_all", "stream_load"],
        ascending=[False, False, True],
    ).head(1)
    best_compression = summary.sort_values(
        ["compression_ratio", "anomaly_recall_against_all"],
        ascending=[False, False],
        na_position="last",
    ).head(1)
    report = {
        "dataset": args.dataset,
        "input_csv": args.input_csv,
        "population": "label-free regions constructed from score-ranked suspicious events; labels used only for offline evaluation",
        "totals": totals,
        "parameters": {
            "methods": methods,
            "top_k": top_k,
            "seed_max_gap": args.seed_max_gap,
            "region_context_margin": args.region_context_margin,
            "true_max_gap": args.true_max_gap,
            "true_context_margin": args.true_context_margin,
            "min_true_anomalies_per_cluster": args.min_true_anomalies_per_cluster,
            "split_by_block": args.split_by_block,
        },
        "true_cluster_count": int(len(true_clusters)),
        "best_incident_recall": best_incident.iloc[0].to_dict() if not best_incident.empty else None,
        "best_compression": best_compression.iloc[0].to_dict() if not best_compression.empty else None,
        "outputs": {
            "summary_csv": args.output_summary_csv,
            "regions_csv": args.output_regions_csv,
            "report_json": args.output_report_json,
            "report_md": args.output_report_md,
        },
        "notes": [
            "Region construction uses score-ranked events only; labels are not used to select or cluster seeds.",
            "Evaluation compares predicted regions with label-derived incident clusters from the offline density analysis.",
            "Stream load is computed from merged event-order intervals, not from the number of seed events.",
        ],
    }

    output_paths = [
        Path(args.output_summary_csv),
        Path(args.output_regions_csv),
        Path(args.output_report_json),
        Path(args.output_report_md),
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_summary_csv, index=False)
    regions_all.to_csv(args.output_regions_csv, index=False)
    Path(args.output_report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, summary, Path(args.output_report_md))
    print(json.dumps(report, indent=2))


def build_label_free_regions(
    seeds: pd.DataFrame,
    *,
    full_df: pd.DataFrame,
    method: str,
    top_k: int,
    max_gap: int,
    context_margin: int,
    split_by_block: bool,
) -> pd.DataFrame:
    if seeds.empty:
        return pd.DataFrame()
    seed_df = seeds.copy()
    seed_df["_seed_marker"] = 1
    clusters = _build_seed_clusters(seed_df, max_gap=max_gap, split_by_block=split_by_block)
    if clusters.empty:
        return clusters
    regions = _add_region_intervals(clusters, full_df, context_margin=context_margin)
    regions["method"] = method
    regions["top_k"] = top_k
    regions["seed_max_gap"] = max_gap
    regions["region_context_margin"] = context_margin
    return regions


def evaluate_regions(
    regions: pd.DataFrame,
    df: pd.DataFrame,
    true_clusters: pd.DataFrame,
    *,
    totals: dict[str, int | str | None],
    method: str,
    top_k: int,
    seed_events: int,
) -> dict[str, Any]:
    total_stream_events = int(totals["total_stream_events"])
    total_anomalies = int(totals["total_anomalies"])
    merged = merge_intervals(regions)
    covered_mask = _rows_in_intervals(df, merged)
    covered = df[covered_mask]
    covered_anomalies = int(covered["label"].sum()) if "label" in covered else 0
    interval_events = int(sum(int(row["interval_end"]) - int(row["interval_start"]) + 1 for _, row in merged.iterrows()))
    hit_true = _hit_true_clusters(true_clusters, merged)
    anomaly_recall = _safe_ratio(covered_anomalies, total_anomalies)
    stream_load = _safe_ratio(interval_events, total_stream_events)
    seed_positive_events = int(regions["seed_positive_events"].sum()) if "seed_positive_events" in regions else 0
    return {
        "method": method,
        "top_k": top_k,
        "seed_events": int(seed_events),
        "seed_positive_events": seed_positive_events,
        "seed_precision": _safe_ratio(seed_positive_events, seed_events),
        "regions": int(len(regions)),
        "merged_regions": int(len(merged)),
        "interval_events": interval_events,
        "stream_load": stream_load,
        "covered_candidate_events": int(len(covered)),
        "covered_anomaly_events": covered_anomalies,
        "anomaly_recall_against_all": anomaly_recall,
        "region_weighted_density": _safe_ratio(covered_anomalies, interval_events),
        "compression_ratio": _safe_div(anomaly_recall, stream_load),
        "true_clusters": int(len(true_clusters)),
        "true_clusters_hit": int(hit_true),
        "true_cluster_recall": _safe_ratio(hit_true, len(true_clusters)),
    }


def merge_intervals(regions: pd.DataFrame) -> pd.DataFrame:
    if regions.empty:
        return pd.DataFrame(columns=["block_id", "interval_start", "interval_end"])
    rows = []
    for block_id, group in regions.sort_values(["block_id", "interval_start", "interval_end"]).groupby("block_id", sort=True):
        current_start: int | None = None
        current_end: int | None = None
        for _, row in group.iterrows():
            start = int(row["interval_start"])
            end = int(row["interval_end"])
            if current_start is None:
                current_start = start
                current_end = end
            elif start <= int(current_end) + 1:
                current_end = max(int(current_end), end)
            else:
                rows.append({"block_id": block_id, "interval_start": current_start, "interval_end": current_end})
                current_start = start
                current_end = end
        if current_start is not None:
            rows.append({"block_id": block_id, "interval_start": current_start, "interval_end": current_end})
    return pd.DataFrame(rows)


def write_markdown_report(report: dict[str, Any], summary: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Label-Free Incident Cluster Coverage",
        "",
        "This report builds suspicious regions without labels by clustering score-ranked event seeds in event order.",
        "Labels are used only afterwards to evaluate anomaly and incident-cluster coverage.",
        "",
        "## Protocol",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Input: `{report['input_csv']}`",
        f"- Seed max gap: `{report['parameters']['seed_max_gap']}` events",
        f"- Region context margin: `{report['parameters']['region_context_margin']}` events",
        f"- Label-derived incident clusters for evaluation: `{report['true_cluster_count']}`",
        "",
        "## Best Rows",
        "",
    ]
    for title, row in [
        ("Best Incident-Cluster Recall", report["best_incident_recall"]),
        ("Best Compression", report["best_compression"]),
    ]:
        lines.extend([f"### {title}", ""])
        if row is None:
            lines.append("No rows available.")
        else:
            lines.extend(
                [
                    f"- Method: `{row['method']}`",
                    f"- Top K seeds: `{int(row['top_k'])}`",
                    f"- Regions: `{int(row['regions'])}`",
                    f"- Stream load: `{_fmt(row['stream_load'])}`",
                    f"- Anomaly recall: `{_fmt(row['anomaly_recall_against_all'])}`",
                    f"- True cluster recall: `{_fmt(row['true_cluster_recall'])}`",
                    f"- Region weighted density: `{_fmt(row['region_weighted_density'])}`",
                    f"- Compression ratio: `{_fmt(row['compression_ratio'])}`",
                ]
            )
        lines.append("")

    preferred = summary[summary["top_k"].isin([500, 1000, 5000])].copy()
    lines.extend(
        [
            "## Selected Top-K Comparison",
            "",
            "| Method | Top K | Regions | Stream load | Anomaly recall | True cluster recall | Density | Compression |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in preferred.sort_values(["top_k", "method"]).to_dict("records"):
        lines.append(
            "| {method} | {top_k} | {regions} | {stream_load} | {anomaly_recall} | {cluster_recall} | {density} | {compression} |".format(
                method=row["method"],
                top_k=int(row["top_k"]),
                regions=int(row["regions"]),
                stream_load=_fmt(row["stream_load"]),
                anomaly_recall=_fmt(row["anomaly_recall_against_all"]),
                cluster_recall=_fmt(row["true_cluster_recall"]),
                density=_fmt(row["region_weighted_density"]),
                compression=_fmt(row["compression_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Thesis Use",
            "",
            "- This is the detector-side counterpart to the label-derived anomaly-density analysis.",
            "- A useful incident pipeline should hit many label-derived clusters at low stream load.",
            "- These regions can become label-free LLM incident packets with representative lines, templates, scores, local context, and retrieval-lite examples.",
            "",
            "## Outputs",
            "",
        ]
    )
    for name, value in report["outputs"].items():
        lines.append(f"- `{name}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_true_clusters(
    df: pd.DataFrame,
    *,
    max_gap: int,
    context_margin: int,
    min_anomalies: int,
    split_by_block: bool,
) -> pd.DataFrame:
    base = build_gap_clusters(df, max_gap=max_gap, split_by_block=split_by_block)
    enriched = add_cluster_intervals(
        base,
        df,
        context_margin=context_margin,
        min_anomalies_per_cluster=min_anomalies,
    )
    if enriched.empty:
        return enriched
    return enriched[enriched["anomaly_events"] >= min_anomalies].copy()


def _build_seed_clusters(seeds: pd.DataFrame, *, max_gap: int, split_by_block: bool) -> pd.DataFrame:
    group_columns = ["block_id"] if split_by_block and "block_id" in seeds.columns else [None]
    rows = []
    cluster_id = 0
    for group_key, group in _iter_groups(seeds, group_columns):
        ordered = group.sort_values(["event_order", "event_id"] if "event_id" in group else ["event_order"])
        current = []
        last_order: int | None = None
        for _, row in ordered.iterrows():
            event_order = int(row["event_order"])
            if last_order is not None and event_order - last_order > max_gap:
                cluster_id += 1
                rows.append(_seed_cluster_row(cluster_id, group_key, current))
                current = []
            current.append(row.to_dict())
            last_order = event_order
        if current:
            cluster_id += 1
            rows.append(_seed_cluster_row(cluster_id, group_key, current))
    return pd.DataFrame(rows)


def _add_region_intervals(clusters: pd.DataFrame, full_df: pd.DataFrame, *, context_margin: int) -> pd.DataFrame:
    if clusters.empty:
        return clusters
    bounds = _block_bounds(full_df)
    rows = []
    for _, row in clusters.iterrows():
        block_id = row["block_id"]
        lower, upper = bounds.get(block_id, bounds["__global__"])
        start = max(int(row["start_event_order"]) - context_margin, lower)
        end = min(int(row["end_event_order"]) + context_margin, upper)
        item = row.to_dict()
        item["interval_start"] = start
        item["interval_end"] = end
        item["interval_length"] = end - start + 1
        rows.append(item)
    return pd.DataFrame(rows)


def _seed_cluster_row(cluster_id: int, group_key: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
    orders = [int(event["event_order"]) for event in events]
    scores = [float(event.get("classical_score", 0.0) or 0.0) for event in events]
    labels = [int(event.get("label", 0) or 0) for event in events]
    templates = [str(event["template"]) for event in events if "template" in event and not pd.isna(event["template"])]
    return {
        "region_id": cluster_id,
        "block_id": _clean_group_key(group_key),
        "seed_events": len(events),
        "seed_positive_events": sum(labels),
        "start_event_order": min(orders),
        "end_event_order": max(orders),
        "seed_span": max(orders) - min(orders) + 1,
        "max_seed_score": max(scores) if scores else 0.0,
        "mean_seed_score": sum(scores) / len(scores) if scores else 0.0,
        "unique_seed_templates": len(set(templates)),
        "dominant_seed_template": _mode(templates),
    }


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


def _hit_true_clusters(true_clusters: pd.DataFrame, predicted: pd.DataFrame) -> int:
    if true_clusters.empty or predicted.empty:
        return 0
    hits = 0
    for _, truth in true_clusters.iterrows():
        truth_block = truth["block_id"]
        truth_start = int(truth["interval_start"])
        truth_end = int(truth["interval_end"])
        overlap = predicted[
            (predicted["block_id"] == truth_block)
            & (predicted["interval_start"] <= truth_end)
            & (predicted["interval_end"] >= truth_start)
        ]
        if not overlap.empty:
            hits += 1
    return hits


def _block_bounds(df: pd.DataFrame) -> dict[Any, tuple[int, int]]:
    bounds = {"__global__": (int(df["event_order"].min()), int(df["event_order"].max()))}
    if "block_id" in df.columns:
        for block_id, group in df.groupby("block_id"):
            bounds[_clean_group_key(block_id)] = (int(group["event_order"].min()), int(group["event_order"].max()))
    return bounds


def _iter_groups(df: pd.DataFrame, group_columns: list[str | None]):
    if group_columns == [None]:
        yield "__global__", df
    else:
        for group_key, group in df.groupby(group_columns[0], sort=True):
            yield group_key, group


def _parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [method for method in methods if method not in DEFAULT_METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}")
    return methods


def _validate_input(df: pd.DataFrame) -> None:
    required = {"event_order", "label"}
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


def _clean_group_key(value: Any) -> Any:
    if value is None:
        return "__global__"
    if pd.isna(value):
        return "__global__"
    if hasattr(value, "item"):
        value = value.item()
    return value


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
