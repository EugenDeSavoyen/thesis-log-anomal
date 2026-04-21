from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RANK_COLUMNS = [
    "template_burst_score",
    "template_count_deviation_score",
    "novelty_score",
    "event_order",
]
DEFAULT_THRESHOLDS = [0, 10, 25, 50, 75, 100, 150, 200, 300, 500]
DEFAULT_TOP_N = [100, 250, 500, 1000, 2500, 5000]
COMPARISON_METHODS = [
    "template_burst_score",
    "template_count_deviation_score",
    "burst_plus_sequence_context",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a review sample from event-level candidates using template-burst ranking."
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/event_level_candidates_bgl_multiblock.csv",
    )
    parser.add_argument("--threshold", type=float, default=100.0)
    parser.add_argument("--max-review-events", type=int, default=500)
    parser.add_argument(
        "--total-stream-events",
        type=int,
        default=None,
        help="Total stream event count. If omitted, inferred from max event_order/event_id when possible.",
    )
    parser.add_argument(
        "--total-anomalies",
        type=int,
        default=None,
        help="Total stream anomaly count. If omitted, inferred from labels in the candidate table.",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(str(item) for item in DEFAULT_THRESHOLDS),
        help="Comma-separated template-burst thresholds for the high-recall pool sweep.",
    )
    parser.add_argument(
        "--top-n",
        default=",".join(str(item) for item in DEFAULT_TOP_N),
        help="Comma-separated review sample sizes for the ranked top-N sweep.",
    )
    parser.add_argument(
        "--event-cv-csv",
        default="outputs/reports/event_level_cv_bgl_multiblock.csv",
        help="Optional event-level CV summary CSV for the method-comparison report.",
    )
    parser.add_argument(
        "--event-cv-json",
        default="outputs/reports/event_level_cv_bgl_multiblock.json",
        help="Optional event-level CV JSON used to infer stream/anomaly totals.",
    )
    parser.add_argument(
        "--output-sample",
        default="outputs/reports/template_burst_review_sample.csv",
    )
    parser.add_argument(
        "--output-template-summary",
        default="outputs/reports/template_burst_review_template_summary.csv",
    )
    parser.add_argument(
        "--output-report",
        default="outputs/reports/template_burst_review_sample.json",
    )
    parser.add_argument(
        "--output-threshold-sweep",
        default="outputs/reports/template_burst_threshold_sweep.csv",
    )
    parser.add_argument(
        "--output-topn-sweep",
        default="outputs/reports/template_burst_topn_sweep.csv",
    )
    parser.add_argument(
        "--output-method-comparison",
        default="outputs/reports/event_level_triage_comparison.csv",
    )
    parser.add_argument(
        "--output-method-report",
        default="outputs/reports/event_level_triage_comparison.md",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    sidecar_totals = _read_sidecar_totals(Path(args.event_cv_json))
    totals = _infer_totals(
        df,
        total_stream_events=args.total_stream_events or sidecar_totals.get("total_stream_events"),
        total_anomalies=args.total_anomalies or sidecar_totals.get("total_anomalies"),
    )
    selected_pool = df[df["template_burst_score"] >= args.threshold].copy()
    ranked = selected_pool.sort_values(
        RANK_COLUMNS,
        ascending=[False, False, False, True],
    )
    review_sample = ranked.head(args.max_review_events).copy()

    sample_path = Path(args.output_sample)
    template_summary_path = Path(args.output_template_summary)
    report_path = Path(args.output_report)
    threshold_sweep_path = Path(args.output_threshold_sweep)
    topn_sweep_path = Path(args.output_topn_sweep)
    method_comparison_path = Path(args.output_method_comparison)
    method_report_path = Path(args.output_method_report)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    template_summary_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    threshold_sweep_path.parent.mkdir(parents=True, exist_ok=True)
    topn_sweep_path.parent.mkdir(parents=True, exist_ok=True)
    method_comparison_path.parent.mkdir(parents=True, exist_ok=True)
    method_report_path.parent.mkdir(parents=True, exist_ok=True)

    review_columns = [
        "event_id",
        "event_order",
        "block_id",
        "label",
        "template",
        "template_burst_score",
        "template_count_deviation_score",
        "template_count_past_z",
        "novelty_score",
        "rarity_score",
        "local_sequence_context_score",
        "max_template_count_in_window",
        "max_template_ratio_in_window",
        "message",
    ]
    review_sample[[column for column in review_columns if column in review_sample.columns]].to_csv(
        sample_path,
        index=False,
    )

    template_summary = _template_summary(review_sample)
    template_summary.to_csv(template_summary_path, index=False)
    thresholds = _parse_float_list(args.thresholds)
    top_n_values = _parse_int_list(args.top_n)
    threshold_sweep = _threshold_sweep(df, thresholds, totals)
    topn_sweep = _topn_sweep(ranked, top_n_values, totals)
    threshold_sweep.to_csv(threshold_sweep_path, index=False)
    topn_sweep.to_csv(topn_sweep_path, index=False)
    method_comparison = _method_comparison(Path(args.event_cv_csv))
    method_comparison.to_csv(method_comparison_path, index=False)
    _write_method_report(method_comparison, method_report_path)

    report = {
        "input_csv": args.input_csv,
        "totals": totals,
        "selection": {
            "score": "template_burst_score",
            "threshold": args.threshold,
            "rank_order": RANK_COLUMNS,
            "max_review_events": args.max_review_events,
        },
        "total_candidate_table": _metrics(df, totals),
        "threshold_pool": _metrics(selected_pool, totals),
        "review_sample": _metrics(review_sample, totals),
        "threshold_sweep_best_f1": _best_row(threshold_sweep, "f1"),
        "topn_sweep_best_precision": _best_row(topn_sweep, "precision"),
        "outputs": {
            "sample_csv": str(sample_path),
            "template_summary_csv": str(template_summary_path),
            "report_json": str(report_path),
            "threshold_sweep_csv": str(threshold_sweep_path),
            "topn_sweep_csv": str(topn_sweep_path),
            "method_comparison_csv": str(method_comparison_path),
            "method_report_md": str(method_report_path),
        },
        "notes": [
            "The threshold pool is the high-recall template-burst candidate set.",
            "The review sample is a ranked top-N subset intended for human inspection, not final recall reporting.",
            "Labels are included only for offline evaluation and should not be used by an analyst-facing process.",
            "Stream totals can be overridden with --total-stream-events and --total-anomalies for other datasets.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def _parse_float_list(raw: str) -> list[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return sorted(set(values))


def _parse_int_list(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return sorted({value for value in values if value > 0})


def _infer_totals(
    df: pd.DataFrame,
    *,
    total_stream_events: int | None = None,
    total_anomalies: int | None = None,
) -> dict:
    total_anomalies_was_inferred = total_anomalies is None
    inferred_stream_events = None
    for column in ["event_order", "event_id"]:
        if column in df.columns and not df.empty:
            inferred_stream_events = int(df[column].max()) + 1
            break
    if total_stream_events is None:
        total_stream_events = inferred_stream_events or int(len(df))
    if total_anomalies is None:
        total_anomalies = int(df["label"].sum()) if "label" in df else 0
    return {
        "total_stream_events": int(total_stream_events),
        "total_anomalies": int(total_anomalies),
        "stream_events_inferred_from_event_ids": total_stream_events == inferred_stream_events,
        "anomalies_inferred_from_candidates": total_anomalies_was_inferred,
    }


def _read_sidecar_totals(path: Path) -> dict:
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


def _metrics(df: pd.DataFrame, totals: dict) -> dict:
    total = int(len(df))
    positives = int(df["label"].sum()) if "label" in df else 0
    total_anomalies = int(totals["total_anomalies"])
    total_stream_events = int(totals["total_stream_events"])
    precision = None if total == 0 else positives / total
    recall = None if total_anomalies == 0 else positives / total_anomalies
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    return {
        "num_events": total,
        "positive_events": positives,
        "negative_events": int(total - positives),
        "precision": precision,
        "recall_against_candidate_anomalies": recall,
        "f1": f1,
        "event_load_ratio_against_stream": None if total_stream_events == 0 else total / total_stream_events,
    }


def _threshold_sweep(df: pd.DataFrame, thresholds: list[float], totals: dict) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        metrics = _metrics(df[df["template_burst_score"] >= threshold], totals)
        rows.append({"threshold": threshold, **metrics})
    return pd.DataFrame(rows)


def _topn_sweep(ranked: pd.DataFrame, top_n_values: list[int], totals: dict) -> pd.DataFrame:
    rows = []
    for top_n in top_n_values:
        metrics = _metrics(ranked.head(top_n), totals)
        rows.append({"top_n": top_n, **metrics})
    return pd.DataFrame(rows)


def _method_comparison(path: Path) -> pd.DataFrame:
    columns = [
        "method",
        "micro_precision",
        "micro_recall_within_suspicious",
        "micro_all_anomaly_recall",
        "micro_f1",
        "micro_candidate_ratio_within_suspicious",
    ]
    if not path.exists():
        return pd.DataFrame(columns=columns)
    df = pd.read_csv(path)
    rows = df[df["model_name"].isin(COMPARISON_METHODS)].copy()
    rows = rows.rename(columns={"model_name": "method"})
    rows["method"] = rows["method"].map(
        {
            "template_burst_score": "template burst",
            "template_count_deviation_score": "template-count deviation",
            "burst_plus_sequence_context": "burst plus sequence context",
        }
    ).fillna(rows["method"])
    return rows[[column for column in columns if column in rows.columns]].sort_values(
        "micro_f1",
        ascending=False,
    )


def _write_method_report(df: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Event-Level Triage Comparison",
        "",
        "This table compares the current zero-LLM event triage methods inside GEV-suspicious windows.",
        "",
    ]
    if df.empty:
        lines.extend(
            [
                "No comparison rows were written because the event-level CV CSV was not found.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "| Method | Micro precision | Micro recall inside suspicious | Micro recall among all anomalies | Micro F1 | Candidate ratio inside suspicious |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in df.to_dict("records"):
            lines.append(
                "| {method} | {precision:.3f} | {recall:.3f} | {all_recall:.3f} | {f1:.3f} | {ratio:.3f} |".format(
                    method=row["method"],
                    precision=row["micro_precision"],
                    recall=row["micro_recall_within_suspicious"],
                    all_recall=row["micro_all_anomaly_recall"],
                    f1=row["micro_f1"],
                    ratio=row["micro_candidate_ratio_within_suspicious"],
                )
            )
        lines.extend(
            [
                "",
                "Interpretation: template burst remains the strongest simple event-level score. "
                "Template-count deviation is nearly as strong and slightly more selective, while the first naive sequence-context combination does not improve the tradeoff.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _best_row(df: pd.DataFrame, metric: str) -> dict | None:
    if df.empty or metric not in df.columns:
        return None
    ranked = df.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if ranked.empty:
        return None
    return ranked.iloc[0].to_dict()


def _template_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=[
                "template",
                "num_events",
                "positive_events",
                "precision",
                "first_event_id",
                "example_message",
            ]
        )
    grouped = (
        df.groupby("template")
        .agg(
            num_events=("event_id", "count"),
            positive_events=("label", "sum"),
            first_event_id=("event_id", "min"),
            example_message=("message", "first"),
        )
        .reset_index()
    )
    grouped["precision"] = grouped["positive_events"] / grouped["num_events"]
    return grouped.sort_values(
        ["positive_events", "num_events", "precision"],
        ascending=[False, False, False],
    )


if __name__ == "__main__":
    main()
