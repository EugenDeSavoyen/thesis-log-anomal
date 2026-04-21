from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


SAFE_FEATURES_V1 = [
    "gev_score",
    "gev_excess",
    "log_count",
    "distinct_templates",
    "template_entropy",
    "rare_template_ratio",
    "unseen_template_ratio",
    "new_template_ratio",
    "mean_event_novelty_score",
    "count_z_baseline",
]


FEATURE_AUDIT = {
    "gev_score": "safe: derived from GEV fit already used by the pre-LLM stage",
    "gev_excess": "safe: derived from GEV fit already used by the pre-LLM stage",
    "log_count": "safe: computed inside the window",
    "distinct_templates": "safe: computed inside the window after Drain parsing",
    "template_entropy": "safe: computed inside the window after Drain parsing",
    "rare_template_ratio": "past-safe: historical template counts come from the history segment",
    "unseen_template_ratio": "past-safe: historical template counts come from the history segment",
    "new_template_ratio": "safe: Drain online parsing flag available at the event time",
    "mean_event_novelty_score": "past-safe: novelty uses the history template reference",
    "count_z_baseline": "past-safe: expanding baseline over previous stream windows only",
    "error_ratio": "unavailable: current BGL schema has anomaly labels but no independent severity field",
    "error_z_baseline": "unavailable: current BGL schema has anomaly labels but no independent severity field",
}


def prepare_pre_model_artifacts(payload: dict, config: dict) -> dict:
    """Materialize the window-level modeling table before fitting ML baselines."""
    pre_model_config = config.get("pre_model", {})
    if not bool(pre_model_config.get("enabled", False)):
        return {}

    windows = payload["windows_after_gev"]
    table = _build_modeling_rows(windows, pre_model_config)
    output_csv = Path(pre_model_config.get("output_csv", "data/processed/pre_model_windows.csv"))
    output_report = Path(pre_model_config.get("output_report", "outputs/reports/pre_model_readiness.json"))
    output_note = Path(pre_model_config.get("output_note", "outputs/reports/pre_model_methodology.md"))

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_note.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(output_csv, table)
    report = _build_report(table, output_csv, config)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output_note.write_text(_build_methodology_note(report), encoding="utf-8")
    return report


def _build_modeling_rows(windows: list[dict], pre_model_config: dict) -> list[dict]:
    split = pre_model_config.get("split", {})
    split_strategy = split.get("strategy", "time_ordered")
    train_fraction = float(split.get("train_fraction", 0.50))
    validation_fraction = float(split.get("validation_fraction", 0.25))
    event_count_config = pre_model_config.get("event_count_features", {})

    scored_windows = [window for window in windows if window.get("evt")]
    block_split_map = (
        _build_block_split_map(scored_windows, train_fraction, validation_fraction)
        if split_strategy == "block_aware"
        else {}
    )
    log_counts_seen: list[float] = []
    rows: list[dict] = []

    for index, window in enumerate(scored_windows):
        feature_summary = window.get("feature_summary", {})
        ground_truth = window.get("ground_truth", {})
        evt_features = _extract_evt_features(window)
        log_count = float(feature_summary.get("num_records", len(window.get("records", []))))
        count_z_baseline = _past_z_score(log_count, log_counts_seen)
        log_counts_seen.append(log_count)

        row = {
            "window_id": window.get("window_id"),
            "window_order": index,
            "start_index": window.get("start_index"),
            "end_index": window.get("end_index"),
            "start_time": window.get("start_time"),
            "end_time": window.get("end_time"),
            "block_id": _window_block_id(window),
            "split": _assign_split(
                index,
                len(scored_windows),
                train_fraction,
                validation_fraction,
                window=window,
                block_split_map=block_split_map,
            ),
            "population_all_windows": 1,
            "population_post_gev": int(bool(window.get("evt", {}).get("is_suspicious"))),
            "label": ground_truth.get("window_label"),
            "anomaly_event_ratio": _safe_ratio(
                float(ground_truth.get("num_anomalous_records", 0)),
                float(ground_truth.get("num_labeled_records", 0)),
            ),
            "gev_score": evt_features["gev_score"],
            "gev_excess": evt_features["gev_excess"],
            "log_count": log_count,
            "distinct_templates": float(feature_summary.get("unique_templates", 0.0)),
            "template_entropy": float(feature_summary.get("template_entropy", 0.0)),
            "rare_template_ratio": float(feature_summary.get("historically_rare_template_ratio", 0.0)),
            "unseen_template_ratio": float(feature_summary.get("unseen_in_history_ratio", 0.0)),
            "new_template_ratio": float(feature_summary.get("new_template_ratio", 0.0)),
            "mean_event_novelty_score": float(feature_summary.get("mean_event_novelty_score", 0.0)),
            "count_z_baseline": count_z_baseline,
            "gev_is_suspicious": int(bool(window.get("evt", {}).get("is_suspicious"))),
            "_template_counts": dict(window.get("template_summary", {}).get("template_counts", {})),
        }
        rows.append(row)

    if bool(event_count_config.get("enabled", False)):
        _add_event_count_features(
            rows,
            top_k=int(event_count_config.get("top_k", 100)),
        )

    return rows


def _extract_evt_features(window: dict) -> dict[str, float]:
    evt = window.get("evt", {})
    if evt.get("method") != "gev_ensemble":
        value = _as_float(evt.get("value"))
        threshold = _as_float(evt.get("threshold"))
        return {
            "gev_score": value,
            "gev_excess": value - threshold,
        }

    member_scores: list[float] = []
    member_excesses: list[float] = []
    for member in window.get("evt_ensemble", {}).get("members", {}).values():
        value = _as_float(member.get("value"))
        threshold = _as_float(member.get("threshold"))
        if math.isfinite(value):
            member_scores.append(value)
        if math.isfinite(value) and math.isfinite(threshold):
            member_excesses.append(value - threshold)

    return {
        "gev_score": max(member_scores) if member_scores else 0.0,
        "gev_excess": max(member_excesses) if member_excesses else 0.0,
    }


def _past_z_score(value: float, previous_values: list[float]) -> float:
    if len(previous_values) < 3:
        return 0.0
    baseline_mean = mean(previous_values)
    baseline_std = pstdev(previous_values)
    if baseline_std == 0:
        return 0.0
    return (value - baseline_mean) / baseline_std


def _assign_split(
    index: int,
    total: int,
    train_fraction: float,
    validation_fraction: float,
    window: dict | None = None,
    block_split_map: dict[str, str] | None = None,
) -> str:
    if block_split_map and window is not None:
        block_id = _window_block_id(window)
        if block_id in block_split_map:
            return block_split_map[block_id]

    if total <= 0:
        return "test"
    train_end = max(1, int(total * train_fraction))
    validation_end = max(train_end + 1, int(total * (train_fraction + validation_fraction)))
    if index < train_end:
        return "train"
    if index < validation_end:
        return "validation"
    return "test"


def _write_csv(path: Path, rows: list[dict]) -> None:
    event_count_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("event_count__")
        }
    )
    fieldnames = [
        "window_id",
        "window_order",
        "start_index",
        "end_index",
        "start_time",
        "end_time",
        "block_id",
        "split",
        "population_all_windows",
        "population_post_gev",
        "label",
        "anomaly_event_ratio",
        *SAFE_FEATURES_V1,
        *event_count_fields,
        "gev_is_suspicious",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if not key.startswith("_")} for row in rows])


def _build_report(rows: list[dict], output_csv: Path, config: dict) -> dict:
    by_population = {
        "all_windows": rows,
        "post_gev": [row for row in rows if row["population_post_gev"] == 1],
    }
    populations = {
        name: _summarize_population(items)
        for name, items in by_population.items()
    }
    split_summary = {
        split: _summarize_population([row for row in rows if row["split"] == split])
        for split in ["train", "validation", "test"]
    }

    return {
        "dataset": config.get("data", {}).get("dataset"),
        "modeling_population_primary": "post_gev",
        "modeling_population_secondary": "all_windows",
        "label_rule": config.get("windowing", {}).get("label_rule", "any_anomaly"),
        "split_policy": "time_ordered_window_split",
        "split_strategy": config.get("pre_model", {}).get("split", {}).get("strategy", "time_ordered"),
        "safe_features_v1": SAFE_FEATURES_V1,
        "num_safe_features_v1": len(SAFE_FEATURES_V1),
        "event_count_features": _event_count_feature_report(rows),
        "feature_audit": FEATURE_AUDIT,
        "artifacts": {
            "modeling_table_csv": str(output_csv),
        },
        "populations": populations,
        "splits": split_summary,
        "block_splits": _summarize_block_splits(rows),
        "gev_only_baseline": _summarize_gev_baseline(rows),
        "overfitting_risk": _overfitting_risk(populations),
        "training_recommendation": {
            "first_models": ["logistic_regression", "shallow_xgboost", "shallow_lightgbm"],
            "primary_training_set": "post_gev if it has enough positives; otherwise all_windows with post_gev evaluation",
            "threshold_objective": "minimize forwarded workload subject to high anomaly recall",
            "notes": [
                "Do not use random splits because overlapping windows leak neighboring events.",
                "Do not use anomaly labels to approximate severity/error features.",
                "Treat current local BGL subset as a small-sample experiment, not final evidence.",
            ],
        },
    }


def _summarize_population(rows: list[dict]) -> dict:
    positives = sum(1 for row in rows if row.get("label") == 1)
    negatives = sum(1 for row in rows if row.get("label") == 0)
    return {
        "num_windows": len(rows),
        "positive_windows": positives,
        "negative_windows": negatives,
        "positive_rate": _safe_ratio(float(positives), float(len(rows))),
        "features_to_windows_ratio": _safe_ratio(float(len(SAFE_FEATURES_V1)), float(len(rows))),
        "features_to_positive_windows_ratio": _safe_ratio(float(len(SAFE_FEATURES_V1)), float(positives)),
    }


def _summarize_gev_baseline(rows: list[dict]) -> dict:
    def summarize(items: list[dict]) -> dict:
        tp = sum(1 for row in items if row.get("label") == 1 and row.get("gev_is_suspicious") == 1)
        fp = sum(1 for row in items if row.get("label") == 0 and row.get("gev_is_suspicious") == 1)
        tn = sum(1 for row in items if row.get("label") == 0 and row.get("gev_is_suspicious") == 0)
        fn = sum(1 for row in items if row.get("label") == 1 and row.get("gev_is_suspicious") == 0)
        precision = _safe_ratio(float(tp), float(tp + fp))
        recall = _safe_ratio(float(tp), float(tp + fn))
        f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return {
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "forwarded_window_ratio": _safe_ratio(float(tp + fp), float(len(items))),
        }

    return {
        "overall": summarize(rows),
        "by_split": {
            split: summarize([row for row in rows if row["split"] == split])
            for split in ["train", "validation", "test"]
        },
    }


def _overfitting_risk(populations: dict[str, dict]) -> dict:
    primary = populations["post_gev"]
    positives = primary["positive_windows"]
    windows = primary["num_windows"]
    if positives < 20 or windows < 100:
        level = "high"
    elif positives < 50 or windows < 300:
        level = "medium"
    else:
        level = "low"
    return {
        "level": level,
        "reason": "risk is based on primary post-GEV sample size and positive-window count",
        "recommended_response": "start with the 10-feature v1 subset, shallow models, strong regularization, and time-based validation",
    }


def _build_methodology_note(report: dict) -> str:
    populations = report["populations"]
    baseline = report["gev_only_baseline"]["overall"]
    return f"""# Pre-Model Setup

## Modeling Population

- Primary population: post-GEV suspicious windows.
- Secondary population: all scored windows, retained as a diagnostic comparison.
- Label rule: `{report["label_rule"]}`.
- Split policy: `{report["split_strategy"]}`.

## Safe Feature Set v1

The first supervised baselines should use these {report["num_safe_features_v1"]} features:

{chr(10).join(f"- `{feature}`" for feature in report["safe_features_v1"])}

`error_ratio` and `error_z_baseline` are intentionally excluded because the current BGL table exposes anomaly labels, not an independent severity field. Using labels as error features would leak the target.

## Event Count Feature Family

- Enabled event-count features: {report["event_count_features"]["num_features"]}
- Selection policy: top templates by frequency in the training split only.
- These columns provide the classical event-count-vector representation used by PCA, Isolation Forest, Logistic Regression, and SVM baselines.

## Current Sample Size

- All windows: {populations["all_windows"]["num_windows"]}
- Post-GEV windows: {populations["post_gev"]["num_windows"]}
- Positive post-GEV windows: {populations["post_gev"]["positive_windows"]}
- Features / post-GEV windows: {populations["post_gev"]["features_to_windows_ratio"]}
- Features / positive post-GEV windows: {populations["post_gev"]["features_to_positive_windows_ratio"]}

## Block Split Summary

{chr(10).join(f"- `{split}`: {summary}" for split, summary in report["block_splits"].items())}

## GEV-Only Baseline

- Precision: {baseline["precision"]}
- Recall: {baseline["recall"]}
- F1: {baseline["f1"]}
- Forwarded window ratio: {baseline["forwarded_window_ratio"]}

## Decision Before Training

The next step is to train `LogisticRegression`, shallow `XGBoost`, and shallow `LightGBM` only after confirming the modeling table looks sensible. Because the current local dataset is small, early experiments should be treated as methodology checks rather than final performance evidence.
"""


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _add_event_count_features(rows: list[dict], top_k: int) -> None:
    train_rows = [row for row in rows if row.get("split") == "train"]
    template_totals: dict[str, int] = {}
    for row in train_rows:
        for template, count in row.get("_template_counts", {}).items():
            template_totals[template] = template_totals.get(template, 0) + int(count)

    selected_templates = [
        template
        for template, _ in sorted(template_totals.items(), key=lambda item: (-item[1], item[0]))[:top_k]
    ]
    feature_names = {
        template: f"event_count__{index:03d}__{_template_hash(template)}"
        for index, template in enumerate(selected_templates, start=1)
    }
    for row in rows:
        counts = row.get("_template_counts", {})
        for template, feature_name in feature_names.items():
            row[feature_name] = int(counts.get(template, 0))
        row["_event_count_feature_map"] = feature_names


def _template_hash(template: str) -> str:
    return hashlib.sha1(template.encode("utf-8", errors="ignore")).hexdigest()[:10]


def _event_count_feature_report(rows: list[dict]) -> dict:
    event_count_fields = sorted(
        {
            key
            for row in rows
            for key in row
            if key.startswith("event_count__")
        }
    )
    feature_map = {}
    for row in rows:
        feature_map = row.get("_event_count_feature_map", {})
        if feature_map:
            break
    reverse_map = {feature: template for template, feature in feature_map.items()}
    return {
        "num_features": len(event_count_fields),
        "feature_columns": event_count_fields,
        "template_map": {
            feature: reverse_map.get(feature)
            for feature in event_count_fields
        },
    }


def _window_block_id(window: dict) -> str | None:
    block_id = window.get("metadata", {}).get("block_id")
    if block_id not in (None, ""):
        return str(block_id)
    block_ids: list[str] = []
    for record in window.get("records", []):
        record_block = record.get("metadata", {}).get("block_id")
        if record_block not in (None, ""):
            block_ids.append(str(record_block))
    if not block_ids:
        return None
    counts: dict[str, int] = {}
    for item in block_ids:
        counts[item] = counts.get(item, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _build_block_split_map(
    windows: list[dict],
    train_fraction: float,
    validation_fraction: float,
) -> dict[str, str]:
    block_ids = []
    for window in windows:
        block_id = _window_block_id(window)
        if block_id is not None and block_id not in block_ids:
            block_ids.append(block_id)

    if len(block_ids) < 3:
        return {}

    groups = _split_block_ids_by_modulo(block_ids)
    return {
        block_id: split
        for split, ids in groups.items()
        for block_id in ids
    }


def _split_block_ids_by_modulo(block_ids: list[str]) -> dict[str, list[str]]:
    groups = {"train": [], "validation": [], "test": []}
    cycle = ["train", "validation", "test"]
    for index, block_id in enumerate(block_ids):
        groups[cycle[index % len(cycle)]].append(block_id)
    return groups


def _summarize_block_splits(rows: list[dict]) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for split in ["train", "validation", "test"]:
        split_rows = [row for row in rows if row["split"] == split]
        block_ids = sorted({row.get("block_id") for row in split_rows if row.get("block_id") not in (None, "")})
        summary[split] = {
            "block_ids": block_ids,
            "num_blocks": len(block_ids),
            "num_windows": len(split_rows),
            "positive_windows": sum(1 for row in split_rows if row.get("label") == 1),
            "negative_windows": sum(1 for row in split_rows if row.get("label") == 0),
        }
    return summary
