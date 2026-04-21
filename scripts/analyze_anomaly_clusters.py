from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_MAX_GAPS = [5, 10, 20, 50, 100, 200, 500]
DEFAULT_CONTEXT_MARGINS = [0, 10, 50]
DEFAULT_TOP_N = [1, 5, 10, 20, 50]
DEFAULT_GAP_THRESHOLDS = [1, 5, 10, 20, 50, 100, 200, 500]
DEFAULT_DISPERSION_BINS = [100, 500, 1000]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure anomaly concentration and incident-like cluster density in an ordered event stream."
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/event_level_candidates_bgl_multiblock.csv",
        help="Event-level table with event_order/event_id and label columns.",
    )
    parser.add_argument(
        "--event-cv-json",
        default="outputs/reports/event_level_cv_bgl_multiblock.json",
        help="Optional sidecar JSON used to infer full stream/anomaly totals.",
    )
    parser.add_argument("--dataset", default="bgl_multiblock")
    parser.add_argument("--total-stream-events", type=int, default=None)
    parser.add_argument("--total-anomalies", type=int, default=None)
    parser.add_argument(
        "--max-gaps",
        default=",".join(str(item) for item in DEFAULT_MAX_GAPS),
        help="Comma-separated maximum event-position gaps for linking anomalies into clusters.",
    )
    parser.add_argument(
        "--context-margins",
        default=",".join(str(item) for item in DEFAULT_CONTEXT_MARGINS),
        help="Comma-separated margins added to each cluster interval for density/context analysis.",
    )
    parser.add_argument(
        "--min-anomalies-per-cluster",
        type=int,
        default=2,
        help="Minimum anomaly count for a region to count as a non-isolated cluster.",
    )
    parser.add_argument(
        "--split-by-block",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Do not link anomalies across block_id boundaries when block_id is available.",
    )
    parser.add_argument(
        "--top-n",
        default=",".join(str(item) for item in DEFAULT_TOP_N),
        help="Comma-separated top-N cluster cutoffs for concentration curves.",
    )
    parser.add_argument(
        "--gap-thresholds",
        default=",".join(str(item) for item in DEFAULT_GAP_THRESHOLDS),
        help="Comma-separated anomaly-gap thresholds summarized in the markdown report.",
    )
    parser.add_argument(
        "--dispersion-bin-sizes",
        default=",".join(str(item) for item in DEFAULT_DISPERSION_BINS),
        help="Comma-separated event-bin sizes for anomaly-count dispersion index.",
    )
    parser.add_argument(
        "--primary-max-gap",
        type=int,
        default=50,
        help="Cluster setting used for detailed cluster/top-N exports.",
    )
    parser.add_argument(
        "--primary-context-margin",
        type=int,
        default=10,
        help="Context margin used for detailed cluster/top-N exports.",
    )
    parser.add_argument(
        "--output-summary-csv",
        default="outputs/reports/anomaly_cluster_density_summary.csv",
    )
    parser.add_argument(
        "--output-clusters-csv",
        default="outputs/reports/anomaly_cluster_density_clusters.csv",
    )
    parser.add_argument(
        "--output-topn-csv",
        default="outputs/reports/anomaly_cluster_density_topn.csv",
    )
    parser.add_argument(
        "--output-report-json",
        default="outputs/reports/anomaly_cluster_density.json",
    )
    parser.add_argument(
        "--output-report-md",
        default="outputs/reports/anomaly_cluster_density.md",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    _validate_input(df)
    df = _deduplicate_events(df)

    totals = _infer_totals(
        df,
        event_cv_json=Path(args.event_cv_json),
        total_stream_events=args.total_stream_events,
        total_anomalies=args.total_anomalies,
    )
    max_gaps = _parse_int_list(args.max_gaps)
    context_margins = _parse_int_list(args.context_margins, allow_zero=True)
    top_n = _parse_int_list(args.top_n)
    gap_thresholds = _parse_int_list(args.gap_thresholds)
    dispersion_bin_sizes = _parse_int_list(args.dispersion_bin_sizes)

    summary, cluster_sets, topn = analyze_cluster_grid(
        df,
        totals=totals,
        max_gaps=max_gaps,
        context_margins=context_margins,
        min_anomalies_per_cluster=args.min_anomalies_per_cluster,
        split_by_block=args.split_by_block,
        top_n=top_n,
    )

    primary_key = (args.primary_max_gap, args.primary_context_margin)
    if primary_key not in cluster_sets:
        raise ValueError(
            "Primary max gap/context margin must be included in --max-gaps and --context-margins."
        )
    primary_clusters = cluster_sets[primary_key]

    gap_summary = build_gap_summary(df, thresholds=gap_thresholds, split_by_block=args.split_by_block)
    dispersion = build_dispersion_summary(df, totals=totals, bin_sizes=dispersion_bin_sizes)
    report = build_report(
        dataset=args.dataset,
        input_csv=args.input_csv,
        totals=totals,
        max_gaps=max_gaps,
        context_margins=context_margins,
        primary_max_gap=args.primary_max_gap,
        primary_context_margin=args.primary_context_margin,
        min_anomalies_per_cluster=args.min_anomalies_per_cluster,
        split_by_block=args.split_by_block,
        summary=summary,
        topn=topn,
        gap_summary=gap_summary,
        dispersion=dispersion,
        outputs={
            "summary_csv": args.output_summary_csv,
            "clusters_csv": args.output_clusters_csv,
            "topn_csv": args.output_topn_csv,
            "report_json": args.output_report_json,
            "report_md": args.output_report_md,
        },
    )

    output_paths = [
        Path(args.output_summary_csv),
        Path(args.output_clusters_csv),
        Path(args.output_topn_csv),
        Path(args.output_report_json),
        Path(args.output_report_md),
    ]
    for path in output_paths:
        path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(args.output_summary_csv, index=False)
    primary_clusters.to_csv(args.output_clusters_csv, index=False)
    topn.to_csv(args.output_topn_csv, index=False)
    Path(args.output_report_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, Path(args.output_report_md))
    print(json.dumps(report, indent=2))


def analyze_cluster_grid(
    df: pd.DataFrame,
    *,
    totals: dict[str, int | str | None],
    max_gaps: list[int],
    context_margins: list[int],
    min_anomalies_per_cluster: int,
    split_by_block: bool,
    top_n: list[int],
) -> tuple[pd.DataFrame, dict[tuple[int, int], pd.DataFrame], pd.DataFrame]:
    rows = []
    cluster_sets = {}
    topn_rows = []
    for max_gap in max_gaps:
        base_clusters = build_gap_clusters(df, max_gap=max_gap, split_by_block=split_by_block)
        for context_margin in context_margins:
            clusters = add_cluster_intervals(
                base_clusters,
                df,
                context_margin=context_margin,
                min_anomalies_per_cluster=min_anomalies_per_cluster,
            )
            cluster_sets[(max_gap, context_margin)] = clusters
            rows.append(
                {
                    "max_gap": max_gap,
                    "context_margin": context_margin,
                    **summarize_clusters(
                        clusters,
                        totals=totals,
                        min_anomalies_per_cluster=min_anomalies_per_cluster,
                    ),
                }
            )
            topn_rows.extend(
                build_topn_rows(
                    clusters,
                    totals=totals,
                    max_gap=max_gap,
                    context_margin=context_margin,
                    top_n=top_n,
                    min_anomalies_per_cluster=min_anomalies_per_cluster,
                )
            )
    return pd.DataFrame(rows), cluster_sets, pd.DataFrame(topn_rows)


def build_gap_clusters(df: pd.DataFrame, *, max_gap: int, split_by_block: bool) -> pd.DataFrame:
    anomalies = df[df["label"].astype(int) == 1].copy()
    if anomalies.empty:
        return pd.DataFrame(columns=_cluster_columns())
    group_columns = ["block_id"] if split_by_block and "block_id" in anomalies.columns else [None]
    rows = []
    cluster_id = 0
    for group_key, group in _iter_groups(anomalies, group_columns):
        ordered = group.sort_values(["event_order", "event_id"] if "event_id" in group else ["event_order"])
        current_events: list[dict[str, Any]] = []
        last_order: int | None = None
        for _, row in ordered.iterrows():
            event_order = int(row["event_order"])
            if last_order is not None and event_order - last_order > max_gap:
                cluster_id += 1
                rows.append(_cluster_row(cluster_id, group_key, current_events))
                current_events = []
            current_events.append(row.to_dict())
            last_order = event_order
        if current_events:
            cluster_id += 1
            rows.append(_cluster_row(cluster_id, group_key, current_events))
    return pd.DataFrame(rows, columns=_cluster_columns())


def add_cluster_intervals(
    clusters: pd.DataFrame,
    df: pd.DataFrame,
    *,
    context_margin: int,
    min_anomalies_per_cluster: int,
) -> pd.DataFrame:
    if clusters.empty:
        return clusters.copy()
    enriched = clusters.copy()
    bounds = _block_bounds(df)
    interval_starts = []
    interval_ends = []
    interval_lengths = []
    densities = []
    qualifies = []
    for _, row in enriched.iterrows():
        block_key = _clean_group_key(row.get("block_id"))
        lower, upper = bounds.get(block_key, bounds["__global__"])
        start = max(int(row["start_event_order"]) - context_margin, lower)
        end = min(int(row["end_event_order"]) + context_margin, upper)
        length = max(0, end - start + 1)
        anomaly_count = int(row["anomaly_events"])
        interval_starts.append(start)
        interval_ends.append(end)
        interval_lengths.append(length)
        densities.append(anomaly_count / length if length else 0.0)
        qualifies.append(anomaly_count >= min_anomalies_per_cluster)
    enriched["context_margin"] = context_margin
    enriched["interval_start"] = interval_starts
    enriched["interval_end"] = interval_ends
    enriched["interval_length"] = interval_lengths
    enriched["density"] = densities
    enriched["is_non_singleton_cluster"] = qualifies
    enriched = enriched.sort_values(["anomaly_events", "density", "interval_length"], ascending=[False, False, True])
    enriched["anomaly_mass_rank"] = range(1, len(enriched) + 1)
    return enriched


def summarize_clusters(
    clusters: pd.DataFrame,
    *,
    totals: dict[str, int | str | None],
    min_anomalies_per_cluster: int,
) -> dict[str, int | float | None]:
    total_stream_events = int(totals["total_stream_events"])
    total_anomalies = int(totals["total_anomalies"])
    qualifying = clusters[clusters["anomaly_events"] >= min_anomalies_per_cluster].copy()
    clustered_anomalies = int(qualifying["anomaly_events"].sum()) if not qualifying.empty else 0
    interval_length = int(qualifying["interval_length"].sum()) if not qualifying.empty else 0
    density_values = qualifying["density"].tolist() if not qualifying.empty else []
    stream_coverage = _safe_ratio(interval_length, total_stream_events)
    anomaly_coverage = _safe_ratio(clustered_anomalies, total_anomalies)
    return {
        "clusters_total": int(len(clusters)),
        "non_singleton_clusters": int(len(qualifying)),
        "singleton_clusters": int(len(clusters) - len(qualifying)),
        "clustered_anomalies": clustered_anomalies,
        "isolated_anomalies": total_anomalies - clustered_anomalies,
        "cluster_anomaly_coverage": anomaly_coverage,
        "cluster_stream_coverage": stream_coverage,
        "compression_ratio": _safe_div(anomaly_coverage, stream_coverage),
        "weighted_cluster_density": _safe_ratio(clustered_anomalies, interval_length),
        "mean_cluster_density": mean(density_values) if density_values else None,
        "median_cluster_density": median(density_values) if density_values else None,
        "mean_cluster_length": mean(qualifying["interval_length"].tolist()) if not qualifying.empty else None,
        "median_cluster_length": median(qualifying["interval_length"].tolist()) if not qualifying.empty else None,
        "max_cluster_anomalies": int(qualifying["anomaly_events"].max()) if not qualifying.empty else 0,
    }


def build_topn_rows(
    clusters: pd.DataFrame,
    *,
    totals: dict[str, int | str | None],
    max_gap: int,
    context_margin: int,
    top_n: list[int],
    min_anomalies_per_cluster: int,
) -> list[dict[str, int | float | None]]:
    total_stream_events = int(totals["total_stream_events"])
    total_anomalies = int(totals["total_anomalies"])
    qualifying = clusters[clusters["anomaly_events"] >= min_anomalies_per_cluster].copy()
    rows = []
    for cutoff in top_n:
        selected = qualifying.head(cutoff)
        anomalies = int(selected["anomaly_events"].sum()) if not selected.empty else 0
        interval_length = int(selected["interval_length"].sum()) if not selected.empty else 0
        anomaly_coverage = _safe_ratio(anomalies, total_anomalies)
        stream_coverage = _safe_ratio(interval_length, total_stream_events)
        rows.append(
            {
                "max_gap": max_gap,
                "context_margin": context_margin,
                "top_n_clusters": cutoff,
                "clusters_available": int(len(qualifying)),
                "selected_clusters": int(min(cutoff, len(qualifying))),
                "anomalies_covered": anomalies,
                "anomaly_coverage": anomaly_coverage,
                "interval_events": interval_length,
                "stream_coverage": stream_coverage,
                "weighted_density": _safe_ratio(anomalies, interval_length),
                "compression_ratio": _safe_div(anomaly_coverage, stream_coverage),
            }
        )
    return rows


def build_gap_summary(df: pd.DataFrame, *, thresholds: list[int], split_by_block: bool) -> dict[str, Any]:
    gaps = anomaly_gaps(df, split_by_block=split_by_block)
    if not gaps:
        return {"num_gaps": 0, "thresholds": {}}
    return {
        "num_gaps": len(gaps),
        "mean_gap": float(mean(gaps)),
        "median_gap": float(median(gaps)),
        "max_gap": int(max(gaps)),
        "thresholds": {
            str(threshold): {
                "gaps_at_or_below": int(sum(1 for gap in gaps if gap <= threshold)),
                "share": sum(1 for gap in gaps if gap <= threshold) / len(gaps),
            }
            for threshold in thresholds
        },
    }


def anomaly_gaps(df: pd.DataFrame, *, split_by_block: bool) -> list[int]:
    anomalies = df[df["label"].astype(int) == 1].copy()
    group_columns = ["block_id"] if split_by_block and "block_id" in anomalies.columns else [None]
    gaps = []
    for _, group in _iter_groups(anomalies, group_columns):
        positions = sorted(int(value) for value in group["event_order"].tolist())
        gaps.extend([right - left for left, right in zip(positions, positions[1:])])
    return gaps


def build_dispersion_summary(
    df: pd.DataFrame,
    *,
    totals: dict[str, int | str | None],
    bin_sizes: list[int],
) -> list[dict[str, int | float | None]]:
    anomalies = df[df["label"].astype(int) == 1]
    positions = sorted(int(value) for value in anomalies["event_order"].tolist())
    if not positions:
        return []
    start = min(int(value) for value in df["event_order"].tolist())
    total_stream_events = int(totals["total_stream_events"])
    rows = []
    for bin_size in bin_sizes:
        num_bins = max(1, int(np.ceil(total_stream_events / bin_size)))
        counts = np.zeros(num_bins, dtype=int)
        for position in positions:
            index = min(num_bins - 1, max(0, (position - start) // bin_size))
            counts[index] += 1
        mean_count = float(counts.mean())
        variance = float(counts.var(ddof=0))
        rows.append(
            {
                "bin_size": bin_size,
                "num_bins": num_bins,
                "mean_anomalies_per_bin": mean_count,
                "variance_anomalies_per_bin": variance,
                "dispersion_index": variance / mean_count if mean_count else None,
                "nonempty_bins": int(np.count_nonzero(counts)),
                "nonempty_bin_share": float(np.count_nonzero(counts) / num_bins),
                "max_bin_anomalies": int(counts.max()),
            }
        )
    return rows


def build_report(
    *,
    dataset: str,
    input_csv: str,
    totals: dict[str, int | str | None],
    max_gaps: list[int],
    context_margins: list[int],
    primary_max_gap: int,
    primary_context_margin: int,
    min_anomalies_per_cluster: int,
    split_by_block: bool,
    summary: pd.DataFrame,
    topn: pd.DataFrame,
    gap_summary: dict[str, Any],
    dispersion: list[dict[str, int | float | None]],
    outputs: dict[str, str],
) -> dict[str, Any]:
    primary = summary[
        (summary["max_gap"] == primary_max_gap)
        & (summary["context_margin"] == primary_context_margin)
    ].iloc[0].to_dict()
    primary_topn = topn[
        (topn["max_gap"] == primary_max_gap)
        & (topn["context_margin"] == primary_context_margin)
    ].to_dict("records")
    best_compression = (
        summary.sort_values("compression_ratio", ascending=False, na_position="last").head(1).iloc[0].to_dict()
    )
    best_coverage_at_low_load = (
        summary[summary["cluster_stream_coverage"] <= 0.05]
        .sort_values(["cluster_anomaly_coverage", "compression_ratio"], ascending=[False, False], na_position="last")
        .head(1)
    )
    return {
        "dataset": dataset,
        "input_csv": input_csv,
        "population": "labeled event orders; labels used only for offline anomaly-concentration analysis",
        "totals": totals,
        "parameters": {
            "max_gaps": max_gaps,
            "context_margins": context_margins,
            "primary_max_gap": primary_max_gap,
            "primary_context_margin": primary_context_margin,
            "min_anomalies_per_cluster": min_anomalies_per_cluster,
            "split_by_block": split_by_block,
        },
        "primary_setting": primary,
        "primary_topn": primary_topn,
        "best_compression_setting": best_compression,
        "best_under_5pct_stream_load": (
            best_coverage_at_low_load.iloc[0].to_dict() if not best_coverage_at_low_load.empty else None
        ),
        "gap_summary": gap_summary,
        "dispersion": dispersion,
        "outputs": outputs,
        "interpretation": [
            "Gap-based clusters measure whether labeled anomalies form incident-like bursts in event order.",
            "Cluster anomaly coverage is the fraction of all anomalies inside non-singleton clusters.",
            "Cluster stream coverage is the fraction of the original stream spanned by those cluster intervals.",
            "Compression ratio is anomaly coverage divided by stream coverage; values above 1 mean the cluster regions are anomaly-enriched.",
            "This analysis is not a detector by itself because it uses labels. It motivates LLM incident review and provides an upper-bound view of how concentrated anomaly episodes are.",
        ],
    }


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    primary = report["primary_setting"]
    lines = [
        "# Anomaly Cluster Density Analysis",
        "",
        "This report measures whether labeled anomalies appear as compact incident-like groups rather than isolated events.",
        "Labels are used only for offline analysis and thesis interpretation, not for model scoring or LLM packet construction.",
        "",
        "## Primary Setting",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Input: `{report['input_csv']}`",
        f"- Max anomaly gap: `{report['parameters']['primary_max_gap']}` events",
        f"- Context margin: `{report['parameters']['primary_context_margin']}` events",
        f"- Minimum anomalies per cluster: `{report['parameters']['min_anomalies_per_cluster']}`",
        f"- Split by block: `{report['parameters']['split_by_block']}`",
        "",
        "## Main Numbers",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total stream events | {report['totals']['total_stream_events']} |",
        f"| Total anomalies | {report['totals']['total_anomalies']} |",
        f"| Non-singleton clusters | {int(primary['non_singleton_clusters'])} |",
        f"| Clustered anomalies | {int(primary['clustered_anomalies'])} |",
        f"| Isolated anomalies | {int(primary['isolated_anomalies'])} |",
        f"| Cluster anomaly coverage | {_fmt(primary['cluster_anomaly_coverage'])} |",
        f"| Cluster stream coverage | {_fmt(primary['cluster_stream_coverage'])} |",
        f"| Compression ratio | {_fmt(primary['compression_ratio'])} |",
        f"| Weighted cluster density | {_fmt(primary['weighted_cluster_density'])} |",
        f"| Median cluster density | {_fmt(primary['median_cluster_density'])} |",
        "",
        "Interpretation: cluster coverage tells how much anomaly mass belongs to incident-like bursts; stream coverage tells how much of the log those regions occupy.",
        "",
        "## Top-N Cluster Concentration",
        "",
        "| Top clusters | Anomaly coverage | Stream coverage | Weighted density | Compression |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["primary_topn"]:
        lines.append(
            "| {top_n_clusters} | {anomaly_coverage} | {stream_coverage} | {weighted_density} | {compression_ratio} |".format(
                top_n_clusters=int(row["top_n_clusters"]),
                anomaly_coverage=_fmt(row["anomaly_coverage"]),
                stream_coverage=_fmt(row["stream_coverage"]),
                weighted_density=_fmt(row["weighted_density"]),
                compression_ratio=_fmt(row["compression_ratio"]),
            )
        )
    lines.extend(
        [
            "",
            "## Anomaly Gap Summary",
            "",
            f"- Number of within-block anomaly gaps: `{report['gap_summary'].get('num_gaps', 0)}`",
            f"- Median anomaly gap: `{_fmt(report['gap_summary'].get('median_gap'))}` events",
            f"- Mean anomaly gap: `{_fmt(report['gap_summary'].get('mean_gap'))}` events",
            "",
            "| Gap threshold | Share of anomaly gaps at or below threshold |",
            "| ---: | ---: |",
        ]
    )
    for threshold, values in report["gap_summary"].get("thresholds", {}).items():
        lines.append(f"| {threshold} | {_fmt(values['share'])} |")
    lines.extend(
        [
            "",
            "## Dispersion By Fixed Event Bins",
            "",
            "| Bin size | Dispersion index | Nonempty bin share | Max anomalies in bin |",
            "| ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["dispersion"]:
        lines.append(
            "| {bin_size} | {dispersion_index} | {nonempty_bin_share} | {max_bin_anomalies} |".format(
                bin_size=row["bin_size"],
                dispersion_index=_fmt(row["dispersion_index"]),
                nonempty_bin_share=_fmt(row["nonempty_bin_share"]),
                max_bin_anomalies=row["max_bin_anomalies"],
            )
        )
    lines.extend(
        [
            "",
            "## Thesis Use",
            "",
            "- Use this as anomaly-concentration evidence, not as a deployable detector.",
            "- A high compression ratio supports incident-level LLM review: the LLM can summarize dense regions instead of judging every event independently.",
            "- The detailed cluster CSV can seed future LLM packets that include representative lines, local context, dominant templates, and retrieval-lite examples.",
            "",
            "## Outputs",
            "",
        ]
    )
    for name, value in report["outputs"].items():
        lines.append(f"- `{name}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _cluster_row(cluster_id: int, group_key: Any, events: list[dict[str, Any]]) -> dict[str, Any]:
    orders = [int(event["event_order"]) for event in events]
    event_ids = [int(event["event_id"]) for event in events if "event_id" in event and not pd.isna(event["event_id"])]
    templates = [str(event["template"]) for event in events if "template" in event and not pd.isna(event["template"])]
    messages = [str(event["message"]) for event in events if "message" in event and not pd.isna(event["message"])]
    return {
        "cluster_id": cluster_id,
        "block_id": _clean_group_key(group_key),
        "anomaly_events": len(events),
        "start_event_order": min(orders),
        "end_event_order": max(orders),
        "anomaly_span": max(orders) - min(orders) + 1,
        "first_event_id": min(event_ids) if event_ids else None,
        "last_event_id": max(event_ids) if event_ids else None,
        "unique_templates": len(set(templates)),
        "dominant_template": _mode(templates),
        "example_message": messages[0] if messages else None,
    }


def _cluster_columns() -> list[str]:
    return [
        "cluster_id",
        "block_id",
        "anomaly_events",
        "start_event_order",
        "end_event_order",
        "anomaly_span",
        "first_event_id",
        "last_event_id",
        "unique_templates",
        "dominant_template",
        "example_message",
    ]


def _iter_groups(df: pd.DataFrame, group_columns: list[str | None]):
    if group_columns == [None]:
        yield "__global__", df
    else:
        for group_key, group in df.groupby(group_columns[0], sort=True):
            yield group_key, group


def _block_bounds(df: pd.DataFrame) -> dict[Any, tuple[int, int]]:
    global_bounds = (int(df["event_order"].min()), int(df["event_order"].max()))
    bounds = {"__global__": global_bounds}
    if "block_id" in df.columns:
        for block_id, group in df.groupby("block_id"):
            bounds[_clean_group_key(block_id)] = (int(group["event_order"].min()), int(group["event_order"].max()))
    return bounds


def _deduplicate_events(df: pd.DataFrame) -> pd.DataFrame:
    dedupe_column = "event_id" if "event_id" in df.columns else "event_order"
    return df.sort_values(["event_order", dedupe_column]).drop_duplicates(subset=[dedupe_column]).copy()


def _validate_input(df: pd.DataFrame) -> None:
    required = {"event_order", "label"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")


def _infer_totals(
    df: pd.DataFrame,
    *,
    event_cv_json: Path,
    total_stream_events: int | None,
    total_anomalies: int | None,
) -> dict[str, int | str | None]:
    sidecar = _read_sidecar_totals(event_cv_json)
    stream_events = total_stream_events or sidecar.get("total_stream_events") or _event_order_total(df)
    anomalies = total_anomalies or sidecar.get("total_anomalies") or int(df["label"].sum())
    return {
        "total_stream_events": int(stream_events),
        "total_anomalies": int(anomalies),
        "sidecar_json": str(event_cv_json) if event_cv_json.exists() else None,
    }


def _read_sidecar_totals(path: Path) -> dict[str, int]:
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


def _event_order_total(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    return int(df["event_order"].max()) - int(df["event_order"].min()) + 1


def _parse_int_list(raw: str, *, allow_zero: bool = False) -> list[int]:
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    minimum = 0 if allow_zero else 1
    invalid = [value for value in values if value < minimum]
    if invalid:
        raise ValueError(f"Values must be >= {minimum}: {invalid}")
    return values


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


def _safe_ratio(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator) / float(denominator)


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def _fmt(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


if __name__ == "__main__":
    main()
