from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TOP_K = [100, 250, 500, 1000, 2500, 5000]
DEFAULT_METHODS = [
    "template_burst",
    "template_count_deviation",
    "novelty",
    "rarity",
    "template_count_z",
    "markov_bigram_surprise",
    "markov_trigram_surprise",
    "burst_plus_deviation",
    "burst_plus_sequence_context",
    "burst_plus_markov",
]
REVIEW_COLUMNS = [
    "event_id",
    "event_order",
    "block_id",
    "label",
    "template",
    "classical_score",
    "classical_method",
    "template_burst_score",
    "template_count_deviation_score",
    "template_count_past_z",
    "novelty_score",
    "rarity_score",
    "local_sequence_context_score",
    "markov_bigram_surprise",
    "markov_trigram_surprise",
    "message",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Establish a zero-LLM classical event triage baseline inside suspicious windows."
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/event_level_candidates_bgl_multiblock.csv",
    )
    parser.add_argument(
        "--event-cv-json",
        default="outputs/reports/event_level_cv_bgl_multiblock.json",
        help="Optional event-level CV JSON used to infer stream/anomaly totals.",
    )
    parser.add_argument("--total-stream-events", type=int, default=None)
    parser.add_argument("--total-anomalies", type=int, default=None)
    parser.add_argument(
        "--top-k",
        default=",".join(str(item) for item in DEFAULT_TOP_K),
        help="Comma-separated review cutoffs for precision@K/recall@K.",
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method names. Available: " + ", ".join(DEFAULT_METHODS),
    )
    parser.add_argument(
        "--primary-method",
        default="template_burst",
        choices=DEFAULT_METHODS,
    )
    parser.add_argument("--primary-top-k", type=int, default=500)
    parser.add_argument(
        "--output-summary-csv",
        default="outputs/reports/classical_triage_baseline_summary.csv",
    )
    parser.add_argument(
        "--output-template-csv",
        default="outputs/reports/classical_triage_baseline_template_summary.csv",
    )
    parser.add_argument(
        "--output-review-csv",
        default="outputs/reports/classical_triage_baseline_review_sample.csv",
    )
    parser.add_argument(
        "--output-report-json",
        default="outputs/reports/classical_triage_baseline.json",
    )
    parser.add_argument(
        "--output-report-md",
        default="outputs/reports/classical_triage_baseline.md",
    )
    args = parser.parse_args()

    df = _with_markov_columns(pd.read_csv(args.input_csv))
    totals = _infer_totals(
        df,
        event_cv_json=Path(args.event_cv_json),
        total_stream_events=args.total_stream_events,
        total_anomalies=args.total_anomalies,
    )
    methods = _parse_methods(args.methods)
    top_k = _parse_int_list(args.top_k)

    summary = build_classical_triage_summary(df, methods=methods, top_k=top_k, totals=totals)
    template_summary = build_template_summary(
        df,
        method=args.primary_method,
        top_k=args.primary_top_k,
        totals=totals,
    )
    review_sample = rank_events(df, args.primary_method).head(args.primary_top_k).copy()
    review_sample["classical_method"] = args.primary_method

    summary_path = Path(args.output_summary_csv)
    template_path = Path(args.output_template_csv)
    review_path = Path(args.output_review_csv)
    report_json_path = Path(args.output_report_json)
    report_md_path = Path(args.output_report_md)
    for path in [summary_path, template_path, review_path, report_json_path, report_md_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_path, index=False)
    template_summary.to_csv(template_path, index=False)
    review_sample[[column for column in REVIEW_COLUMNS if column in review_sample.columns]].to_csv(
        review_path,
        index=False,
    )

    best_precision = _best_at(summary, "precision")
    best_f1 = _best_at(summary, "f1")
    report = {
        "input_csv": args.input_csv,
        "population": "deduplicated_events_inside_gev_suspicious_windows",
        "totals": totals,
        "methods": methods,
        "top_k": top_k,
        "primary_review_sample": {
            "method": args.primary_method,
            "top_k": args.primary_top_k,
            "output_csv": str(review_path),
        },
        "best_precision_at_k": best_precision,
        "best_f1_at_k": best_f1,
        "outputs": {
            "summary_csv": str(summary_path),
            "template_summary_csv": str(template_path),
            "review_sample_csv": str(review_path),
            "report_json": str(report_json_path),
            "report_md": str(report_md_path),
        },
        "notes": [
            "This is the zero-LLM classical baseline for later LLM triage comparison.",
            "Metrics are rank-based because the later LLM stage should receive a bounded review budget.",
            "Labels are only used for offline evaluation, never for score construction.",
            "Template metrics measure how many unique templates/templates-with-anomalies remain for downstream processing.",
        ],
    }
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(summary, template_summary, report, report_md_path)
    print(json.dumps(report, indent=2))


def build_classical_triage_summary(
    df: pd.DataFrame,
    *,
    methods: list[str],
    top_k: list[int],
    totals: dict,
) -> pd.DataFrame:
    rows = []
    for method in methods:
        ranked = rank_events(df, method)
        for cutoff in top_k:
            selected = ranked.head(cutoff)
            rows.append(
                {
                    "method": method,
                    "top_k": cutoff,
                    **_selection_metrics(selected, df, totals),
                }
            )
    return pd.DataFrame(rows)


def build_template_summary(
    df: pd.DataFrame,
    *,
    method: str,
    top_k: int,
    totals: dict,
) -> pd.DataFrame:
    ranked = rank_events(df, method).head(top_k)
    if ranked.empty:
        return pd.DataFrame(
            columns=[
                "method",
                "top_k",
                "template",
                "num_events",
                "positive_events",
                "precision",
                "first_event_id",
                "example_message",
            ]
        )
    grouped = (
        ranked.groupby("template")
        .agg(
            num_events=("event_id", "count"),
            positive_events=("label", "sum"),
            first_event_id=("event_id", "min"),
            example_message=("message", "first"),
        )
        .reset_index()
    )
    grouped["method"] = method
    grouped["top_k"] = top_k
    grouped["precision"] = grouped["positive_events"] / grouped["num_events"]
    return grouped[
        [
            "method",
            "top_k",
            "template",
            "num_events",
            "positive_events",
            "precision",
            "first_event_id",
            "example_message",
        ]
    ].sort_values(["positive_events", "num_events", "precision"], ascending=[False, False, False])


def rank_events(df: pd.DataFrame, method: str) -> pd.DataFrame:
    scored = df.copy()
    scored["classical_score"] = _score(scored, method)
    scored["classical_method"] = method
    tie_breakers = [
        column
        for column in [
            "classical_score",
            "template_burst_score",
            "template_count_deviation_score",
            "novelty_score",
            "rarity_score",
            "event_order",
        ]
        if column in scored.columns
    ]
    ascending = [False for _ in tie_breakers]
    if tie_breakers and tie_breakers[-1] == "event_order":
        ascending[-1] = True
    return scored.sort_values(tie_breakers, ascending=ascending)


def _score(df: pd.DataFrame, method: str) -> pd.Series:
    if method == "template_burst":
        return _column(df, "template_burst_score")
    if method == "template_count_deviation":
        return _column(df, "template_count_deviation_score")
    if method == "novelty":
        return _column(df, "novelty_score")
    if method == "rarity":
        return _column(df, "rarity_score")
    if method == "template_count_z":
        return _column(df, "template_count_past_z")
    if method == "markov_bigram_surprise":
        return _markov_surprise(df, order=2)
    if method == "markov_trigram_surprise":
        return _markov_surprise(df, order=3)
    if method == "burst_plus_deviation":
        return _normalized(df, "template_burst_score") + _normalized(df, "template_count_deviation_score")
    if method == "burst_plus_sequence_context":
        return _normalized(df, "template_burst_score") + _normalized(df, "local_sequence_context_score")
    if method == "burst_plus_markov":
        return _normalized(df, "template_burst_score") + _normalized(_with_markov_columns(df), "markov_bigram_surprise")
    raise ValueError(f"Unknown classical triage method: {method}")


def _with_markov_columns(df: pd.DataFrame) -> pd.DataFrame:
    if {"markov_bigram_surprise", "markov_trigram_surprise"}.issubset(df.columns):
        return df
    enriched = df.copy()
    enriched["markov_bigram_surprise"] = _markov_surprise(enriched, order=2)
    enriched["markov_trigram_surprise"] = _markov_surprise(enriched, order=3)
    return enriched


def _markov_surprise(df: pd.DataFrame, *, order: int, alpha: float = 0.1) -> pd.Series:
    """Online template-transition surprise, scored before updating counts.

    The model is deliberately lightweight: within each block, it uses previous
    template transitions observed earlier in the stream to estimate the
    probability of the current template given the previous order-1 templates.
    Higher values mean more surprising transitions.
    """
    if order < 2:
        raise ValueError("Markov order must be at least 2.")
    if "template" not in df:
        return pd.Series([0.0] * len(df), index=df.index)

    sort_columns = [column for column in ["block_id", "event_order", "event_id"] if column in df.columns]
    ordered = df.sort_values(sort_columns) if sort_columns else df.copy()
    vocab: set[str] = set()
    context_counts: dict[tuple[str, ...], int] = {}
    transition_counts: dict[tuple[tuple[str, ...], str], int] = {}
    history_by_block: dict[str, list[str]] = {}
    scores = pd.Series([0.0] * len(df), index=df.index, dtype=float)
    context_size = order - 1

    for index, row in ordered.iterrows():
        template = str(row.get("template") or "<unknown>")
        block_id = str(row.get("block_id") or "__global__")
        history = history_by_block.setdefault(block_id, [])
        vocab_size = max(len(vocab), 1)

        if len(history) >= context_size:
            context = tuple(history[-context_size:])
            context_count = context_counts.get(context, 0)
            transition_count = transition_counts.get((context, template), 0)
            probability = (transition_count + alpha) / (context_count + alpha * vocab_size)
            scores.loc[index] = float(-np.log(max(probability, 1e-12)))
        else:
            scores.loc[index] = 0.0

        if len(history) >= context_size:
            context = tuple(history[-context_size:])
            context_counts[context] = context_counts.get(context, 0) + 1
            transition_counts[(context, template)] = transition_counts.get((context, template), 0) + 1
        history.append(template)
        vocab.add(template)

    return scores


def _selection_metrics(selected: pd.DataFrame, all_candidates: pd.DataFrame, totals: dict) -> dict:
    selected_events = int(len(selected))
    positives = int(selected["label"].sum()) if "label" in selected else 0
    candidate_events = int(len(all_candidates))
    candidate_anomalies = int(all_candidates["label"].sum()) if "label" in all_candidates else 0
    total_stream_events = int(totals["total_stream_events"])
    total_anomalies = int(totals["total_anomalies"])

    selected_templates = set(selected["template"].dropna()) if "template" in selected else set()
    all_templates = set(all_candidates["template"].dropna()) if "template" in all_candidates else set()
    all_anomaly_templates = (
        set(all_candidates.loc[all_candidates["label"] == 1, "template"].dropna())
        if {"label", "template"}.issubset(all_candidates.columns)
        else set()
    )
    selected_anomaly_templates = (
        set(selected.loc[selected["label"] == 1, "template"].dropna())
        if {"label", "template"}.issubset(selected.columns)
        else set()
    )
    precision = _safe_ratio(positives, selected_events)
    recall = _safe_ratio(positives, total_anomalies)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "events": selected_events,
        "positive_events": positives,
        "negative_events": selected_events - positives,
        "precision": precision,
        "recall_against_all_anomalies": recall,
        "recall_against_candidate_anomalies": _safe_ratio(positives, candidate_anomalies),
        "f1": f1,
        "event_load_ratio_against_stream": _safe_ratio(selected_events, total_stream_events),
        "event_load_ratio_against_candidates": _safe_ratio(selected_events, candidate_events),
        "unique_templates": len(selected_templates),
        "template_load_ratio_against_candidates": _safe_ratio(len(selected_templates), len(all_templates)),
        "positive_templates": len(selected_anomaly_templates),
        "positive_template_recall": _safe_ratio(len(selected_anomaly_templates), len(all_anomaly_templates)),
    }


def _infer_totals(
    df: pd.DataFrame,
    *,
    event_cv_json: Path,
    total_stream_events: int | None,
    total_anomalies: int | None,
) -> dict:
    sidecar = _read_sidecar_totals(event_cv_json)
    stream_events = total_stream_events or sidecar.get("total_stream_events") or _event_id_total(df)
    anomalies = total_anomalies or sidecar.get("total_anomalies") or int(df["label"].sum())
    return {
        "total_stream_events": int(stream_events),
        "total_anomalies": int(anomalies),
        "sidecar_json": str(event_cv_json) if event_cv_json.exists() else None,
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


def _event_id_total(df: pd.DataFrame) -> int:
    for column in ["event_order", "event_id"]:
        if column in df and not df.empty:
            return int(df[column].max()) + 1
    return int(len(df))


def write_markdown_report(summary: pd.DataFrame, template_summary: pd.DataFrame, report: dict, path: Path) -> None:
    top_k_values = sorted(summary["top_k"].unique().tolist())
    preferred_k = 500 if 500 in top_k_values else top_k_values[0]
    at_preferred = summary[summary["top_k"] == preferred_k].sort_values("precision", ascending=False)

    lines = [
        "# Classical Triage Baseline",
        "",
        "This is the zero-LLM event-level baseline for later comparison with LLM triage.",
        "",
        "## Protocol",
        "",
        "- Input population: deduplicated events inside GEV-suspicious windows.",
        "- Scores use only classical/template features, not labels.",
        "- Evaluation is rank-based with fixed top-K review budgets.",
        "- Template metrics report how many distinct templates remain for downstream processing.",
        "",
        "## Top-K Comparison",
        "",
        f"Shown at top-{preferred_k}.",
        "",
        "| Method | Precision | Recall all anomalies | Recall candidate anomalies | F1 | Stream load | Candidate load | Unique templates | Positive template recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in at_preferred.to_dict("records"):
        lines.append(
            "| {method} | {precision:.3f} | {recall_all:.3f} | {recall_candidate:.3f} | {f1:.3f} | {stream_load:.4f} | {candidate_load:.3f} | {templates} | {template_recall:.3f} |".format(
                method=row["method"],
                precision=row["precision"],
                recall_all=row["recall_against_all_anomalies"],
                recall_candidate=row["recall_against_candidate_anomalies"],
                f1=row["f1"],
                stream_load=row["event_load_ratio_against_stream"],
                candidate_load=row["event_load_ratio_against_candidates"],
                templates=int(row["unique_templates"]),
                template_recall=row["positive_template_recall"],
            )
        )

    lines.extend(
        [
            "",
            "## Best Rows",
            "",
            f"- Best precision@K: `{report['best_precision_at_k']['method']}` at top `{int(report['best_precision_at_k']['top_k'])}` with precision `{report['best_precision_at_k']['precision']:.3f}`.",
            f"- Best F1@K: `{report['best_f1_at_k']['method']}` at top `{int(report['best_f1_at_k']['top_k'])}` with F1 `{report['best_f1_at_k']['f1']:.3f}`.",
            "",
            "## Primary Review Sample Templates",
            "",
            "| Template | Events | Positive events | Precision | First event id |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in template_summary.head(20).to_dict("records"):
        lines.append(
            "| {template} | {events} | {positives} | {precision:.3f} | {first_event_id} |".format(
                template=str(row["template"]).replace("|", "\\|"),
                events=int(row["num_events"]),
                positives=int(row["positive_events"]),
                precision=row["precision"],
                first_event_id=int(row["first_event_id"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This baseline is intentionally simple and reproducible. It should be the direct comparator for an LLM triage stage using the same input population and the same top-K review budgets.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _best_at(summary: pd.DataFrame, metric: str) -> dict | None:
    ranked = summary.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if ranked.empty:
        return None
    return ranked.iloc[0].to_dict()


def _parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [method for method in methods if method not in DEFAULT_METHODS]
    if unknown:
        raise ValueError(f"Unknown method(s): {', '.join(unknown)}")
    return methods


def _parse_int_list(raw: str) -> list[int]:
    return sorted({int(item.strip()) for item in raw.split(",") if item.strip() and int(item.strip()) > 0})


def _column(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df:
        return pd.Series([0.0] * len(df), index=df.index)
    return df[column].fillna(0.0).astype(float)


def _normalized(df: pd.DataFrame, column: str) -> pd.Series:
    values = _column(df, column)
    max_value = values.max()
    if max_value <= 0:
        return values
    return values / max_value


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


if __name__ == "__main__":
    main()
