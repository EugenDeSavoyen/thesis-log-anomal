from __future__ import annotations

import csv
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EVENT_FEATURES = [
    "novelty_score",
    "rarity_score",
    "historical_count_log",
    "unseen_in_history",
    "historically_rare",
    "is_new_template",
    "template_burst_score",
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
    "template_count_past_mean",
    "template_count_past_std",
    "template_count_past_z",
    "template_count_past_ratio",
    "template_count_past_excess",
    "template_count_seen_windows",
    "template_count_deviation_score",
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
    "local_sequence_context_score",
]


def run_event_level_baselines(
    windows_after_gev: list[dict],
    stream_records: list[dict],
    *,
    min_validation_recall: float = 0.90,
    output_report: str | Path | None = None,
    output_table: str | Path | None = None,
    output_models: str | Path | None = None,
) -> dict:
    """Run zero-LLM event-level baselines inside GEV-suspicious windows.

    The candidate population is restricted to events that appear in suspicious
    windows. Overlapping windows can repeat the same event, so the modeling
    table is deduplicated by event_id and keeps max/aggregate evidence.
    """
    event_rows = build_event_candidate_rows(windows_after_gev)
    split_rows = assign_event_splits(event_rows)
    split_summary = summarize_splits(split_rows)
    total_event_summary = summarize_total_events(stream_records, split_rows)

    if not _all_splits_have_both_classes(split_rows):
        report = {
            "population": "events_inside_suspicious_windows",
            "min_validation_recall": min_validation_recall,
            "split_summary": split_summary,
            "total_event_summary": total_event_summary,
            "feature_columns": EVENT_FEATURES,
            "results": [],
            "best": None,
            "notes": [
                "At least one split does not contain both classes; event-level baselines were not fitted.",
            ],
        }
        _write_outputs(report, [], {}, output_report, output_table, output_models)
        return report

    df = pd.DataFrame(split_rows)
    results: list[dict] = []
    fitted_models: dict[str, object] = {}

    for model_name, scorer in _fit_scorers(df):
        result = _evaluate_scorer(
            df,
            model_name=model_name,
            scorer=scorer,
            min_validation_recall=min_validation_recall,
            total_event_summary=total_event_summary,
        )
        results.append(result)
        if scorer.get("model") is not None:
            fitted_models[model_name] = scorer["model"]

    best = _select_best(results)
    report = {
        "population": "events_inside_suspicious_windows",
        "min_validation_recall": min_validation_recall,
        "selection_rule": f"highest validation precision among thresholds with recall >= {min_validation_recall}",
        "split_summary": split_summary,
        "total_event_summary": total_event_summary,
        "feature_columns": EVENT_FEATURES,
        "results": results,
        "best": {
            "model_name": best["model_name"],
            "threshold": best["threshold"],
            "test_precision": best["metrics"]["test"]["precision"],
            "test_recall_within_suspicious": best["metrics"]["test"]["recall"],
            "test_recall_all_anomalies": best["metrics"]["test"]["all_anomaly_recall"],
            "test_f1": best["metrics"]["test"]["f1"],
        },
        "notes": [
            "This is a zero-LLM event-level baseline; it only ranks events already forwarded by GEV.",
            "all_anomaly_recall includes anomalies missed by GEV in the denominator.",
            "within-suspicious recall measures how well the event baseline finds anomalies that GEV already forwarded.",
        ],
    }
    _write_outputs(report, results, fitted_models, output_report, output_table, output_models)
    return report


def run_event_level_cross_validation(
    windows_after_gev: list[dict],
    stream_records: list[dict],
    *,
    min_validation_recall: float = 0.90,
    output_report: str | Path | None = None,
    output_table: str | Path | None = None,
    output_models: str | Path | None = None,
) -> dict:
    """Run block-aware CV for event-level baselines inside suspicious windows."""
    event_rows = build_event_candidate_rows(windows_after_gev)
    total_event_summary = summarize_total_events(stream_records, event_rows)
    folds = make_event_block_folds(event_rows)
    all_results: list[dict] = []
    fitted_models: dict[str, object] = {}

    for fold in folds:
        fold_rows = assign_event_fold(event_rows, fold)
        if not _all_splits_have_both_classes(fold_rows):
            all_results.append(
                {
                    "fold": fold["name"],
                    "model_name": "skipped",
                    "threshold": None,
                    "train_blocks": fold["train_blocks"],
                    "validation_blocks": fold["validation_blocks"],
                    "test_blocks": fold["test_blocks"],
                    "metrics": {},
                    "notes": ["At least one split does not contain both classes."],
                }
            )
            continue

        df = pd.DataFrame(fold_rows)
        for model_name, scorer in _fit_scorers(df):
            result = _evaluate_scorer(
                df,
                model_name=model_name,
                scorer=scorer,
                min_validation_recall=min_validation_recall,
                total_event_summary=total_event_summary,
            )
            result["fold"] = fold["name"]
            result["train_blocks"] = fold["train_blocks"]
            result["validation_blocks"] = fold["validation_blocks"]
            result["test_blocks"] = fold["test_blocks"]
            all_results.append(result)
            if scorer.get("model") is not None:
                fitted_models[f"{fold['name']}::{model_name}"] = scorer["model"]

    summary = _summarize_cv_results(all_results, total_event_summary["total_anomalous_events"])
    report = {
        "population": "events_inside_suspicious_windows",
        "min_validation_recall": min_validation_recall,
        "selection_rule": f"per-fold highest validation precision among thresholds with recall >= {min_validation_recall}",
        "folds": folds,
        "total_event_summary": total_event_summary,
        "feature_columns": EVENT_FEATURES,
        "summary": summary,
        "fold_results": all_results,
        "notes": [
            "This is block-aware event-level CV after the GEV candidate generator.",
            "Rows are deduplicated events inside suspicious windows.",
            "Micro all_anomaly_recall divides summed test true positives by total anomalous stream events.",
        ],
    }
    _write_cv_outputs(report, summary, fitted_models, output_report, output_table, output_models)
    return report


def write_event_candidate_rows(rows: list[dict], output_csv: str | Path) -> None:
    """Write deduplicated event-level candidate rows for inspection/sampling."""
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "event_id",
        "event_order",
        "block_id",
        "label",
        "template",
        *EVENT_FEATURES,
        "message",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def build_event_candidate_rows(windows_after_gev: list[dict]) -> list[dict]:
    by_event: dict[int, dict] = {}
    template_count_history: dict[str, list[float]] = {}

    for window in sorted(windows_after_gev, key=lambda item: item.get("window_id", 0)):
        feature_summary = window.get("feature_summary", {})
        evt_features = _window_evt_features(window)
        template_counts = window.get("template_summary", {}).get("template_counts", {})
        num_records = max(1, int(feature_summary.get("num_records", len(window.get("records", []))) or 1))
        burst_scores = _compute_burst_scores(window.get("records", []))
        sequence_contexts = _compute_sequence_contexts(window.get("records", []))
        template_deviation = {
            template: _template_count_deviation(float(count), template_count_history.get(template, []))
            for template, count in template_counts.items()
        }

        if not window.get("evt", {}).get("is_suspicious"):
            _update_template_count_history(template_count_history, template_counts)
            continue

        for record, burst_score, sequence_context in zip(window.get("records", []), burst_scores, sequence_contexts):
            event_id = record.get("event_id")
            if event_id is None:
                continue

            template = record.get("template") or "<unknown>"
            novelty = record.get("novelty", {})
            template_count = int(template_counts.get(template, 0))
            deviation = template_deviation.get(template, _template_count_deviation(float(template_count), []))
            row = by_event.setdefault(
                int(event_id),
                {
                    "event_id": int(event_id),
                    "event_order": int(event_id),
                    "split": None,
                    "block_id": _record_block_id(record),
                    "label": record.get("label"),
                    "template": template,
                    "message": record.get("message") or record.get("log", ""),
                    "novelty_score": 0.0,
                    "rarity_score": 0.0,
                    "historical_count_log": 0.0,
                    "unseen_in_history": 0.0,
                    "historically_rare": 0.0,
                    "is_new_template": 0.0,
                    "template_burst_score": 0.0,
                    "suspicious_window_count": 0.0,
                    "max_window_gev_score": 0.0,
                    "max_window_gev_excess": 0.0,
                    "max_window_template_entropy": 0.0,
                    "max_window_rare_template_ratio": 0.0,
                    "max_window_unseen_template_ratio": 0.0,
                    "max_window_new_template_ratio": 0.0,
                    "max_window_mean_event_novelty_score": 0.0,
                    "max_template_count_in_window": 0.0,
                    "max_template_ratio_in_window": 0.0,
                    "template_count_past_mean": 0.0,
                    "template_count_past_std": 0.0,
                    "template_count_past_z": 0.0,
                    "template_count_past_ratio": 0.0,
                    "template_count_past_excess": 0.0,
                    "template_count_seen_windows": 0.0,
                    "template_count_deviation_score": 0.0,
                    "prev_event_novelty_score": 0.0,
                    "next_event_novelty_score": 0.0,
                    "neighbor_max_novelty_score": 0.0,
                    "neighbor_mean_novelty_score": 0.0,
                    "local_unseen_count_radius2": 0.0,
                    "local_new_template_count_radius2": 0.0,
                    "local_template_switch_count_radius2": 0.0,
                    "prev_template_same": 0.0,
                    "next_template_same": 0.0,
                    "relative_position_in_window": 0.0,
                    "distance_to_window_edge_ratio": 0.0,
                    "local_sequence_context_score": 0.0,
                },
            )

            row["label"] = record.get("label")
            if row.get("block_id") in (None, ""):
                row["block_id"] = _record_block_id(record)
            row["novelty_score"] = max(row["novelty_score"], float(novelty.get("novelty_score", 0.0)))
            row["rarity_score"] = max(row["rarity_score"], float(novelty.get("rarity_score", 0.0)))
            row["historical_count_log"] = max(
                row["historical_count_log"],
                float(np.log1p(float(novelty.get("historical_count", 0.0)))),
            )
            row["unseen_in_history"] = max(row["unseen_in_history"], float(bool(novelty.get("unseen_in_history"))))
            row["historically_rare"] = max(row["historically_rare"], float(bool(novelty.get("historically_rare"))))
            row["is_new_template"] = max(
                row["is_new_template"],
                float(bool(novelty.get("is_new_template")) or bool(record.get("drain", {}).get("is_new_template"))),
            )
            row["template_burst_score"] = max(row["template_burst_score"], float(burst_score))
            row["suspicious_window_count"] += 1.0
            row["max_window_gev_score"] = max(row["max_window_gev_score"], evt_features["gev_score"])
            row["max_window_gev_excess"] = max(row["max_window_gev_excess"], evt_features["gev_excess"])
            row["max_window_template_entropy"] = max(
                row["max_window_template_entropy"],
                float(feature_summary.get("template_entropy", 0.0)),
            )
            row["max_window_rare_template_ratio"] = max(
                row["max_window_rare_template_ratio"],
                float(feature_summary.get("historically_rare_template_ratio", 0.0)),
            )
            row["max_window_unseen_template_ratio"] = max(
                row["max_window_unseen_template_ratio"],
                float(feature_summary.get("unseen_in_history_ratio", 0.0)),
            )
            row["max_window_new_template_ratio"] = max(
                row["max_window_new_template_ratio"],
                float(feature_summary.get("new_template_ratio", 0.0)),
            )
            row["max_window_mean_event_novelty_score"] = max(
                row["max_window_mean_event_novelty_score"],
                float(feature_summary.get("mean_event_novelty_score", 0.0)),
            )
            row["max_template_count_in_window"] = max(row["max_template_count_in_window"], float(template_count))
            row["max_template_ratio_in_window"] = max(
                row["max_template_ratio_in_window"],
                float(template_count) / float(num_records),
            )
            row["template_count_past_mean"] = max(
                row["template_count_past_mean"],
                deviation["past_mean"],
            )
            row["template_count_past_std"] = max(
                row["template_count_past_std"],
                deviation["past_std"],
            )
            row["template_count_past_z"] = max(
                row["template_count_past_z"],
                deviation["z_score"],
            )
            row["template_count_past_ratio"] = max(
                row["template_count_past_ratio"],
                deviation["ratio"],
            )
            row["template_count_past_excess"] = max(
                row["template_count_past_excess"],
                deviation["excess"],
            )
            row["template_count_seen_windows"] = max(
                row["template_count_seen_windows"],
                deviation["seen_windows"],
            )
            row["template_count_deviation_score"] = max(
                row["template_count_deviation_score"],
                deviation["deviation_score"],
            )
            for feature_name in [
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
                "local_sequence_context_score",
            ]:
                row[feature_name] = max(row[feature_name], sequence_context[feature_name])

        _update_template_count_history(template_count_history, template_counts)

    return sorted(by_event.values(), key=lambda row: row["event_order"])


def assign_event_splits(rows: list[dict]) -> list[dict]:
    block_ids = []
    for row in rows:
        block_id = row.get("block_id")
        if block_id not in (None, "") and block_id not in block_ids:
            block_ids.append(block_id)

    split_by_block = {}
    if len(block_ids) >= 3:
        cycle = ["train", "validation", "test"]
        split_by_block = {
            block_id: cycle[index % len(cycle)]
            for index, block_id in enumerate(block_ids)
        }

    assigned = []
    total = len(rows)
    train_end = max(1, int(total * 0.50)) if total else 0
    validation_end = max(train_end + 1, int(total * 0.75)) if total else 0
    for index, row in enumerate(rows):
        updated = dict(row)
        block_id = row.get("block_id")
        if block_id in split_by_block:
            updated["split"] = split_by_block[block_id]
        elif index < train_end:
            updated["split"] = "train"
        elif index < validation_end:
            updated["split"] = "validation"
        else:
            updated["split"] = "test"
        assigned.append(updated)
    return assigned


def make_event_block_folds(rows: list[dict]) -> list[dict]:
    block_stats: dict[str, dict[str, int]] = {}
    for row in rows:
        block_id = row.get("block_id")
        if block_id in (None, ""):
            continue
        stats = block_stats.setdefault(str(block_id), {"count": 0, "positives": 0})
        stats["count"] += 1
        if row.get("label") == 1:
            stats["positives"] += 1

    positive_blocks = [block_id for block_id, stats in sorted(block_stats.items()) if stats["positives"] > 0]
    normal_blocks = [block_id for block_id, stats in sorted(block_stats.items()) if stats["positives"] == 0]
    if len(positive_blocks) < 3:
        raise ValueError("Need at least three positive blocks for event-level block-aware CV.")

    folds = []
    all_blocks = sorted(block_stats)
    for index, test_positive in enumerate(positive_blocks):
        validation_positive = positive_blocks[(index + 1) % len(positive_blocks)]
        test_blocks = [test_positive]
        if normal_blocks:
            test_blocks.append(normal_blocks[index % len(normal_blocks)])
        validation_blocks = [validation_positive]
        train_blocks = [
            block for block in all_blocks if block not in set(test_blocks + validation_blocks)
        ]
        folds.append(
            {
                "name": f"fold_{index + 1}",
                "train_blocks": train_blocks,
                "validation_blocks": validation_blocks,
                "test_blocks": test_blocks,
            }
        )
    return folds


def assign_event_fold(rows: list[dict], fold: dict) -> list[dict]:
    train_blocks = set(fold["train_blocks"])
    validation_blocks = set(fold["validation_blocks"])
    test_blocks = set(fold["test_blocks"])
    assigned = []
    for row in rows:
        block_id = str(row.get("block_id"))
        updated = dict(row)
        if block_id in train_blocks:
            updated["split"] = "train"
        elif block_id in validation_blocks:
            updated["split"] = "validation"
        elif block_id in test_blocks:
            updated["split"] = "test"
        else:
            continue
        assigned.append(updated)
    return assigned


def summarize_splits(rows: list[dict]) -> dict:
    return {
        split: _summarize_rows([row for row in rows if row.get("split") == split])
        for split in ["train", "validation", "test"]
    }


def summarize_total_events(stream_records: list[dict], candidate_rows: list[dict]) -> dict:
    labeled_records = [record for record in stream_records if record.get("label") in (0, 1)]
    anomalous_event_ids = {
        int(record["event_id"])
        for record in labeled_records
        if record.get("label") == 1 and record.get("event_id") is not None
    }
    candidate_event_ids = {int(row["event_id"]) for row in candidate_rows}
    candidate_anomaly_ids = {
        int(row["event_id"])
        for row in candidate_rows
        if row.get("label") == 1
    }
    return {
        "total_labeled_events": len(labeled_records),
        "total_anomalous_events": len(anomalous_event_ids),
        "candidate_events_inside_suspicious_windows": len(candidate_event_ids),
        "candidate_anomalous_events_inside_suspicious_windows": len(candidate_anomaly_ids),
        "gev_event_load_ratio": _safe_ratio(float(len(candidate_event_ids)), float(len(labeled_records))),
        "gev_anomaly_event_recall": _safe_ratio(float(len(candidate_anomaly_ids)), float(len(anomalous_event_ids))),
    }


def _fit_scorers(df: pd.DataFrame) -> list[tuple[str, dict]]:
    splits = _split_frame(df)
    scorers: list[tuple[str, dict]] = [
        ("novelty_score", {"score_fn": lambda data: data["novelty_score"].to_numpy(dtype=float)}),
        ("rarity_score", {"score_fn": lambda data: data["rarity_score"].to_numpy(dtype=float)}),
        ("template_burst_score", {"score_fn": lambda data: data["template_burst_score"].to_numpy(dtype=float)}),
        ("template_count_past_z", {"score_fn": lambda data: data["template_count_past_z"].to_numpy(dtype=float)}),
        (
            "template_count_deviation_score",
            {"score_fn": lambda data: data["template_count_deviation_score"].to_numpy(dtype=float)},
        ),
        (
            "local_sequence_context_score",
            {"score_fn": lambda data: data["local_sequence_context_score"].to_numpy(dtype=float)},
        ),
        (
            "novelty_plus_burst",
            {"score_fn": lambda data: (data["novelty_score"] + data["template_burst_score"]).to_numpy(dtype=float)},
        ),
        (
            "burst_plus_sequence_context",
            {
                "score_fn": lambda data: (
                    data["template_burst_score"] + data["local_sequence_context_score"]
                ).to_numpy(dtype=float)
            },
        ),
    ]
    scorers.append(("logistic_regression", _fit_logistic(splits)))
    if len(splits["train"][splits["train"]["label"] == 0]) >= 2:
        scorers.append(("pca_spe", _fit_pca(splits)))
    scorers.append(("isolation_forest", _fit_isolation_forest(splits)))
    return scorers


def _fit_logistic(splits: dict[str, pd.DataFrame]) -> dict:
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    penalty="l2",
                    C=0.25,
                    solver="liblinear",
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(splits["train"][EVENT_FEATURES], splits["train"]["label"].astype(int))
    return {"model": model, "score_fn": lambda data: model.predict_proba(data[EVENT_FEATURES])[:, 1]}


def _fit_pca(splits: dict[str, pd.DataFrame]) -> dict:
    normal_train = splits["train"][splits["train"]["label"] == 0][EVENT_FEATURES]
    n_components = max(1, min(normal_train.shape[0] - 1, normal_train.shape[1], 10))
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )
    model.fit(normal_train)

    def score_fn(data: pd.DataFrame) -> np.ndarray:
        raw = data[EVENT_FEATURES]
        transformed = model.transform(raw)
        reconstructed = model.named_steps["pca"].inverse_transform(transformed)
        scaled = model.named_steps["scaler"].transform(model.named_steps["imputer"].transform(raw))
        return np.sum((scaled - reconstructed) ** 2, axis=1)

    return {"model": model, "score_fn": score_fn, "n_components": n_components}


def _fit_isolation_forest(splits: dict[str, pd.DataFrame]) -> dict:
    normal_train = splits["train"][splits["train"]["label"] == 0][EVENT_FEATURES]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("forest", IsolationForest(n_estimators=200, contamination="auto", random_state=42)),
        ]
    )
    model.fit(normal_train)
    return {"model": model, "score_fn": lambda data: -model.decision_function(data[EVENT_FEATURES])}


def _evaluate_scorer(
    df: pd.DataFrame,
    *,
    model_name: str,
    scorer: dict,
    min_validation_recall: float,
    total_event_summary: dict,
) -> dict:
    splits = _split_frame(df)
    scores = {
        split: np.asarray(scorer["score_fn"](data), dtype=float)
        for split, data in splits.items()
    }
    threshold = _choose_threshold(
        splits["validation"]["label"].astype(int).to_numpy(),
        scores["validation"],
        min_validation_recall,
    )
    return {
        "model_name": model_name,
        "threshold": threshold,
        "extra": {key: value for key, value in scorer.items() if key not in {"model", "score_fn"}},
        "metrics": {
            split: _metrics(
                splits[split]["label"].astype(int).to_numpy(),
                scores[split],
                threshold,
                total_anomalous_events=total_event_summary["total_anomalous_events"],
            )
            for split in ["train", "validation", "test"]
        },
    }


def _split_frame(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        split: df[(df["split"] == split) & (df["label"].isin([0, 1]))].copy()
        for split in ["train", "validation", "test"]
    }


def _choose_threshold(y_true: np.ndarray, scores: np.ndarray, min_recall: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    candidates = []
    for index, threshold in enumerate(thresholds):
        if recall[index] >= min_recall:
            candidates.append((precision[index], recall[index], threshold))
    if not candidates:
        return float(np.quantile(scores, 0.90))
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return float(candidates[0][2])


def _metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    *,
    total_anomalous_events: int,
) -> dict:
    pred = (scores >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    precision = _safe_ratio(float(tp), float(tp + fp))
    recall = _safe_ratio(float(tp), float(tp + fn))
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "num_events": int(len(y_true)),
        "positive_events": int(np.sum(y_true == 1)),
        "negative_events": int(np.sum(y_true == 0)),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": _safe_average_precision(y_true, scores),
        "roc_auc": _safe_roc_auc(y_true, scores),
        "candidate_event_ratio_within_suspicious": _safe_ratio(float(tp + fp), float(len(y_true))),
        "all_anomaly_recall": _safe_ratio(float(tp), float(total_anomalous_events)),
    }


def _all_splits_have_both_classes(rows: list[dict]) -> bool:
    for split in ["train", "validation", "test"]:
        labels = {row.get("label") for row in rows if row.get("split") == split and row.get("label") in (0, 1)}
        if labels != {0, 1}:
            return False
    return True


def _select_best(results: list[dict]) -> dict:
    return max(
        results,
        key=lambda result: (
            result["metrics"]["validation"]["precision"] or 0.0,
            result["metrics"]["validation"]["recall"] or 0.0,
            result["metrics"]["validation"]["f1"] or 0.0,
        ),
    )


def _write_outputs(
    report: dict,
    results: list[dict],
    fitted_models: dict[str, object],
    output_report: str | Path | None,
    output_table: str | Path | None,
    output_models: str | Path | None,
) -> None:
    if output_report is not None:
        path = Path(output_report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if output_table is not None:
        _write_summary_csv(results, Path(output_table))
    if output_models is not None:
        path = Path(output_models)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": fitted_models, "report": report}, path)


def _write_cv_outputs(
    report: dict,
    summary: list[dict],
    fitted_models: dict[str, object],
    output_report: str | Path | None,
    output_table: str | Path | None,
    output_models: str | Path | None,
) -> None:
    if output_report is not None:
        path = Path(output_report)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if output_table is not None:
        _write_cv_summary_csv(summary, Path(output_table))
    if output_models is not None:
        path = Path(output_models)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": fitted_models, "report": report}, path)


def _write_summary_csv(results: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "threshold",
        "test_precision",
        "test_recall_within_suspicious",
        "test_all_anomaly_recall",
        "test_f1",
        "test_candidate_event_ratio_within_suspicious",
        "test_pr_auc",
        "test_roc_auc",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            metrics = result["metrics"]["test"]
            writer.writerow(
                {
                    "model_name": result["model_name"],
                    "threshold": result["threshold"],
                    "test_precision": metrics["precision"],
                    "test_recall_within_suspicious": metrics["recall"],
                    "test_all_anomaly_recall": metrics["all_anomaly_recall"],
                    "test_f1": metrics["f1"],
                    "test_candidate_event_ratio_within_suspicious": metrics[
                        "candidate_event_ratio_within_suspicious"
                    ],
                    "test_pr_auc": metrics["pr_auc"],
                    "test_roc_auc": metrics["roc_auc"],
                }
            )


def _write_cv_summary_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model_name",
        "num_folds",
        "test_precision_mean",
        "test_recall_mean",
        "test_f1_mean",
        "test_candidate_ratio_mean",
        "micro_precision",
        "micro_recall_within_suspicious",
        "micro_f1",
        "micro_candidate_ratio_within_suspicious",
        "micro_all_anomaly_recall",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])


def _summarize_cv_results(results: list[dict], total_anomalous_events: int) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for result in results:
        if result.get("model_name") == "skipped" or "test" not in result.get("metrics", {}):
            continue
        grouped.setdefault(result["model_name"], []).append(result)

    rows = []
    for model_name, items in grouped.items():
        test_metrics = [item["metrics"]["test"] for item in items]
        micro = _micro_average(test_metrics, total_anomalous_events)
        row = {
            "model_name": model_name,
            "num_folds": len(items),
            "test_precision_mean": _mean_metric(test_metrics, "precision"),
            "test_recall_mean": _mean_metric(test_metrics, "recall"),
            "test_f1_mean": _mean_metric(test_metrics, "f1"),
            "test_candidate_ratio_mean": _mean_metric(test_metrics, "candidate_event_ratio_within_suspicious"),
            "test_pr_auc_mean": _mean_metric(test_metrics, "pr_auc"),
            "test_roc_auc_mean": _mean_metric(test_metrics, "roc_auc"),
            "micro_precision": micro["precision"],
            "micro_recall_within_suspicious": micro["recall"],
            "micro_f1": micro["f1"],
            "micro_candidate_ratio_within_suspicious": micro["candidate_event_ratio_within_suspicious"],
            "micro_all_anomaly_recall": micro["all_anomaly_recall"],
            "micro_confusion_matrix": micro["confusion_matrix"],
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["micro_f1"] or 0.0, row["micro_recall_within_suspicious"] or 0.0), reverse=True)
    return rows


def _micro_average(metrics: list[dict], total_anomalous_events: int) -> dict:
    tp = sum(item["confusion_matrix"]["tp"] for item in metrics)
    fp = sum(item["confusion_matrix"]["fp"] for item in metrics)
    tn = sum(item["confusion_matrix"]["tn"] for item in metrics)
    fn = sum(item["confusion_matrix"]["fn"] for item in metrics)
    precision = _safe_ratio(float(tp), float(tp + fp))
    recall = _safe_ratio(float(tp), float(tp + fn))
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "candidate_event_ratio_within_suspicious": _safe_ratio(float(tp + fp), float(tp + fp + tn + fn)),
        "all_anomaly_recall": _safe_ratio(float(tp), float(total_anomalous_events)),
    }


def _mean_metric(metrics: list[dict], key: str) -> float | None:
    values = [item.get(key) for item in metrics if item.get(key) is not None]
    if not values:
        return None
    return float(np.mean(values))


def _summarize_rows(rows: list[dict]) -> dict:
    positives = sum(1 for row in rows if row.get("label") == 1)
    negatives = sum(1 for row in rows if row.get("label") == 0)
    return {
        "num_events": len(rows),
        "positive_events": positives,
        "negative_events": negatives,
        "positive_rate": _safe_ratio(float(positives), float(len(rows))),
    }


def _window_evt_features(window: dict) -> dict[str, float]:
    evt = window.get("evt", {})
    if evt.get("method") != "gev_ensemble":
        value = float(evt.get("value", 0.0) or 0.0)
        threshold = float(evt.get("threshold", 0.0) or 0.0)
        return {"gev_score": value, "gev_excess": value - threshold}

    scores = []
    excesses = []
    for member in window.get("evt_ensemble", {}).get("members", {}).values():
        value = float(member.get("value", 0.0) or 0.0)
        threshold = float(member.get("threshold", 0.0) or 0.0)
        scores.append(value)
        excesses.append(value - threshold)
    return {
        "gev_score": max(scores) if scores else 0.0,
        "gev_excess": max(excesses) if excesses else 0.0,
    }


def _compute_burst_scores(records: list[dict]) -> list[int]:
    indicators = [
        bool(record.get("novelty", {}).get("unseen_in_history"))
        or bool(record.get("novelty", {}).get("is_new_template"))
        for record in records
    ]
    scores = [0 for _ in records]
    start = 0
    while start < len(records):
        if not indicators[start]:
            start += 1
            continue
        end = start
        while end < len(records) and indicators[end]:
            end += 1
        burst_len = end - start
        for index in range(start, end):
            scores[index] = burst_len
        start = end
    return scores


def _compute_sequence_contexts(records: list[dict], radius: int = 2) -> list[dict[str, float]]:
    contexts: list[dict[str, float]] = []
    num_records = len(records)
    templates = [record.get("template") or "<unknown>" for record in records]
    novelty_scores = [
        float(record.get("novelty", {}).get("novelty_score", 0.0))
        for record in records
    ]

    for index, record in enumerate(records):
        start = max(0, index - radius)
        end = min(num_records, index + radius + 1)
        neighbor_indices = [item for item in range(start, end) if item != index]
        local_indices = list(range(start, end))
        local_records = [records[item] for item in local_indices]
        neighbor_scores = [novelty_scores[item] for item in neighbor_indices]

        prev_template_same = float(index > 0 and templates[index - 1] == templates[index])
        next_template_same = float(index + 1 < num_records and templates[index + 1] == templates[index])
        switch_count = sum(
            1
            for left, right in zip(local_indices, local_indices[1:])
            if templates[left] != templates[right]
        )
        local_unseen_count = sum(
            1 for item in local_records if bool(item.get("novelty", {}).get("unseen_in_history"))
        )
        local_new_count = sum(
            1
            for item in local_records
            if bool(item.get("novelty", {}).get("is_new_template"))
            or bool(item.get("drain", {}).get("is_new_template"))
        )
        relative_position = 0.0 if num_records <= 1 else index / (num_records - 1)
        edge_distance = min(index, max(0, num_records - 1 - index))
        edge_distance_ratio = 0.0 if num_records <= 1 else edge_distance / ((num_records - 1) / 2)
        neighbor_max = max(neighbor_scores) if neighbor_scores else 0.0
        neighbor_mean = float(np.mean(neighbor_scores)) if neighbor_scores else 0.0
        local_sequence_context_score = (
            neighbor_max
            + local_unseen_count
            + local_new_count
            + switch_count
            + (1.0 - max(prev_template_same, next_template_same))
        )

        contexts.append(
            {
                "prev_event_novelty_score": novelty_scores[index - 1] if index > 0 else 0.0,
                "next_event_novelty_score": novelty_scores[index + 1] if index + 1 < num_records else 0.0,
                "neighbor_max_novelty_score": float(neighbor_max),
                "neighbor_mean_novelty_score": float(neighbor_mean),
                "local_unseen_count_radius2": float(local_unseen_count),
                "local_new_template_count_radius2": float(local_new_count),
                "local_template_switch_count_radius2": float(switch_count),
                "prev_template_same": prev_template_same,
                "next_template_same": next_template_same,
                "relative_position_in_window": float(relative_position),
                "distance_to_window_edge_ratio": float(edge_distance_ratio),
                "local_sequence_context_score": float(local_sequence_context_score),
            }
        )
    return contexts


def _template_count_deviation(current_count: float, history: list[float]) -> dict[str, float]:
    if not history:
        ratio = current_count
        return {
            "past_mean": 0.0,
            "past_std": 0.0,
            "z_score": 0.0,
            "ratio": ratio,
            "excess": current_count,
            "seen_windows": 0.0,
            "deviation_score": float(np.log1p(max(0.0, ratio))),
        }

    past_mean = float(np.mean(history))
    past_std = float(np.std(history, ddof=0))
    z_score = 0.0 if past_std == 0 else (current_count - past_mean) / past_std
    ratio = current_count / (past_mean + 1.0)
    excess = current_count - past_mean
    deviation_score = max(0.0, z_score) + float(np.log1p(max(0.0, ratio)))
    return {
        "past_mean": past_mean,
        "past_std": past_std,
        "z_score": float(z_score),
        "ratio": float(ratio),
        "excess": float(excess),
        "seen_windows": float(len(history)),
        "deviation_score": float(deviation_score),
    }


def _update_template_count_history(
    template_count_history: dict[str, list[float]],
    template_counts: dict[str, int],
) -> None:
    for template, count in template_counts.items():
        template_count_history.setdefault(template, []).append(float(count))


def _record_block_id(record: dict) -> str | None:
    block_id = record.get("metadata", {}).get("block_id", record.get("block_id"))
    if block_id in (None, ""):
        return None
    return str(block_id)


def _safe_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(average_precision_score(y_true, scores))


def _safe_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, scores))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
