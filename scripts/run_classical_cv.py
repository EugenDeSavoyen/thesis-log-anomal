from __future__ import annotations

import argparse
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


COMPACT_FEATURES = [
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Block-aware CV for GEV and classical log baselines.")
    parser.add_argument("--input-csv", default="data/processed/bgl_multiblock_sample_pre_model_windows.csv")
    parser.add_argument("--population", choices=["all_windows", "post_gev"], default="all_windows")
    parser.add_argument("--min-validation-recall", type=float, default=0.90)
    parser.add_argument("--output-report", default="outputs/reports/classical_cv_bgl_multiblock.json")
    parser.add_argument("--output-table", default="outputs/reports/classical_cv_bgl_multiblock.csv")
    parser.add_argument("--output-models", default="outputs/models/classical_cv_bgl_multiblock.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    df["block_id"] = df["block_id"].astype(str)
    if args.population == "post_gev":
        df = df[df["population_post_gev"] == 1].copy()

    folds = _make_folds(df)
    feature_families = _feature_families(df)
    all_results = []
    fitted_models = {}

    for fold in folds:
        split = _split_fold(df, fold)
        for model_name, family_name, feature_columns in _experiment_specs(feature_families):
            scorer = _fit_scorer(model_name, split, feature_columns)
            result = _evaluate_fold(
                fold=fold,
                model_name=model_name,
                feature_family=family_name,
                feature_columns=feature_columns,
                scorer=scorer,
                split=split,
                min_validation_recall=args.min_validation_recall,
            )
            all_results.append(result)
            if "model" in scorer:
                fitted_models[f"{fold['name']}::{family_name}::{model_name}"] = scorer["model"]

    summary = _summarize_results(all_results)
    report = {
        "input_csv": args.input_csv,
        "population": args.population,
        "min_validation_recall": args.min_validation_recall,
        "folds": folds,
        "feature_families": {name: columns for name, columns in feature_families.items()},
        "summary": summary,
        "fold_results": all_results,
        "notes": [
            "GEV is evaluated from the raw gev_score column with a validation-selected threshold.",
            "Event-count vocabulary is the pre-model table vocabulary, selected before this CV run.",
            "Folds test one anomaly-bearing block plus one normal/context block when available.",
        ],
    }

    output_report = Path(args.output_report)
    output_table = Path(args.output_table)
    output_models = Path(args.output_models)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_table.parent.mkdir(parents=True, exist_ok=True)
    output_models.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_summary_csv(summary, output_table)
    joblib.dump({"models": fitted_models, "report": report}, output_models)
    print(json.dumps({"report": str(output_report), "table": str(output_table), "models": str(output_models)}, indent=2))


def _make_folds(df: pd.DataFrame) -> list[dict]:
    block_stats = df.groupby("block_id")["label"].agg(["count", "sum"]).reset_index()
    positive_blocks = block_stats[block_stats["sum"] > 0]["block_id"].tolist()
    normal_blocks = block_stats[block_stats["sum"] == 0]["block_id"].tolist()
    if len(positive_blocks) < 3:
        raise ValueError("Need at least three positive blocks for this CV scheme.")

    folds = []
    for index, test_positive in enumerate(positive_blocks):
        validation_positive = positive_blocks[(index + 1) % len(positive_blocks)]
        test_blocks = [test_positive]
        if normal_blocks:
            test_blocks.append(normal_blocks[index % len(normal_blocks)])
        validation_blocks = [validation_positive]
        train_blocks = [
            block
            for block in sorted(df["block_id"].unique().tolist())
            if block not in set(test_blocks + validation_blocks)
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


def _feature_families(df: pd.DataFrame) -> dict[str, list[str]]:
    compact = [feature for feature in COMPACT_FEATURES if feature in df.columns]
    event_counts = [column for column in df.columns if column.startswith("event_count__")]
    if not event_counts:
        raise ValueError("No event_count__ columns found. Regenerate the pre-model table first.")
    return {
        "compact": compact,
        "event_counts": event_counts,
        "combined": compact + event_counts,
    }


def _experiment_specs(feature_families: dict[str, list[str]]) -> list[tuple[str, str, list[str]]]:
    specs = [
        ("gev_flag", "gev", ["gev_is_suspicious"]),
        ("gev_score", "gev_score", ["gev_score"]),
    ]
    for family_name, feature_columns in feature_families.items():
        for model_name in ["logistic_regression", "pca_spe", "isolation_forest"]:
            specs.append((model_name, family_name, feature_columns))
    return specs


def _split_fold(df: pd.DataFrame, fold: dict) -> dict[str, pd.DataFrame]:
    return {
        "train": df[df["block_id"].isin(fold["train_blocks"])].copy(),
        "validation": df[df["block_id"].isin(fold["validation_blocks"])].copy(),
        "test": df[df["block_id"].isin(fold["test_blocks"])].copy(),
    }


def _fit_scorer(model_name: str, split: dict[str, pd.DataFrame], feature_columns: list[str]) -> dict:
    if model_name == "gev_flag":
        return {"score_fn": lambda data: data["gev_is_suspicious"].to_numpy(dtype=float), "fixed_threshold": 0.5}
    if model_name == "gev_score":
        return {"score_fn": lambda data: data["gev_score"].to_numpy(dtype=float)}
    if model_name == "logistic_regression":
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
        model.fit(split["train"][feature_columns], split["train"]["label"])
        return {"model": model, "score_fn": lambda data: model.predict_proba(data[feature_columns])[:, 1]}
    if model_name == "pca_spe":
        normal_train = split["train"][split["train"]["label"] == 0][feature_columns]
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
            raw = data[feature_columns]
            transformed = model.transform(raw)
            reconstructed = model.named_steps["pca"].inverse_transform(transformed)
            scaled = model.named_steps["scaler"].transform(model.named_steps["imputer"].transform(raw))
            return np.sum((scaled - reconstructed) ** 2, axis=1)

        return {"model": model, "score_fn": score_fn, "n_components": n_components}
    if model_name == "isolation_forest":
        normal_train = split["train"][split["train"]["label"] == 0][feature_columns]
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("forest", IsolationForest(n_estimators=200, contamination="auto", random_state=42)),
            ]
        )
        model.fit(normal_train)
        return {"model": model, "score_fn": lambda data: -model.decision_function(data[feature_columns])}
    raise ValueError(f"Unsupported model: {model_name}")


def _evaluate_fold(
    fold: dict,
    model_name: str,
    feature_family: str,
    feature_columns: list[str],
    scorer: dict,
    split: dict[str, pd.DataFrame],
    min_validation_recall: float,
) -> dict:
    scores = {name: scorer["score_fn"](data) for name, data in split.items()}
    threshold = scorer.get(
        "fixed_threshold",
        _choose_threshold(split["validation"]["label"].to_numpy(), scores["validation"], min_validation_recall),
    )
    return {
        "fold": fold["name"],
        "model_name": model_name,
        "feature_family": feature_family,
        "num_features": len(feature_columns),
        "threshold": threshold,
        "train_blocks": fold["train_blocks"],
        "validation_blocks": fold["validation_blocks"],
        "test_blocks": fold["test_blocks"],
        "extra": {key: value for key, value in scorer.items() if key not in {"model", "score_fn"}},
        "metrics": {
            split_name: _metrics(split[split_name]["label"].to_numpy(), scores[split_name], threshold)
            for split_name in ["train", "validation", "test"]
        },
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


def _metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = (scores >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    precision = _safe_ratio(float(tp), float(tp + fp))
    recall = _safe_ratio(float(tp), float(tp + fn))
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "num_windows": int(len(y_true)),
        "positive_windows": int(np.sum(y_true == 1)),
        "negative_windows": int(np.sum(y_true == 0)),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": _safe_average_precision(y_true, scores),
        "roc_auc": _safe_roc_auc(y_true, scores),
        "forwarded_window_ratio": _safe_ratio(float(tp + fp), float(len(y_true))),
    }


def _summarize_results(results: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for result in results:
        groups.setdefault((result["feature_family"], result["model_name"]), []).append(result)

    rows = []
    for (feature_family, model_name), items in groups.items():
        test_metrics = [item["metrics"]["test"] for item in items]
        micro = _micro_average(test_metrics)
        row = {
            "feature_family": feature_family,
            "model_name": model_name,
            "num_folds": len(items),
            "num_features": items[0]["num_features"],
            "test_precision_mean": _mean_metric(test_metrics, "precision"),
            "test_precision_std": _std_metric(test_metrics, "precision"),
            "test_recall_mean": _mean_metric(test_metrics, "recall"),
            "test_recall_std": _std_metric(test_metrics, "recall"),
            "test_f1_mean": _mean_metric(test_metrics, "f1"),
            "test_f1_std": _std_metric(test_metrics, "f1"),
            "test_pr_auc_mean": _mean_metric(test_metrics, "pr_auc"),
            "test_roc_auc_mean": _mean_metric(test_metrics, "roc_auc"),
            "test_forwarded_ratio_mean": _mean_metric(test_metrics, "forwarded_window_ratio"),
            "micro_precision": micro["precision"],
            "micro_recall": micro["recall"],
            "micro_f1": micro["f1"],
            "micro_forwarded_ratio": micro["forwarded_window_ratio"],
            "micro_confusion_matrix": micro["confusion_matrix"],
        }
        rows.append(row)
    rows.sort(key=lambda row: (row["micro_f1"] or 0.0, row["micro_recall"] or 0.0), reverse=True)
    return rows


def _micro_average(metrics: list[dict]) -> dict:
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
        "forwarded_window_ratio": _safe_ratio(float(tp + fp), float(tp + fp + tn + fn)),
    }


def _write_summary_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "feature_family",
        "model_name",
        "num_folds",
        "num_features",
        "test_precision_mean",
        "test_precision_std",
        "test_recall_mean",
        "test_recall_std",
        "test_f1_mean",
        "test_f1_std",
        "test_pr_auc_mean",
        "test_roc_auc_mean",
        "test_forwarded_ratio_mean",
        "micro_precision",
        "micro_recall",
        "micro_f1",
        "micro_forwarded_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in fieldnames} for row in rows])


def _mean_metric(metrics: list[dict], key: str) -> float | None:
    values = [item[key] for item in metrics if item[key] is not None]
    if not values:
        return None
    return float(np.mean(values))


def _std_metric(metrics: list[dict], key: str) -> float | None:
    values = [item[key] for item in metrics if item[key] is not None]
    if not values:
        return None
    return float(np.std(values, ddof=0))


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


if __name__ == "__main__":
    main()
