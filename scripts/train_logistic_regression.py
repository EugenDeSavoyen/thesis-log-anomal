from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


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


MODEL_CONFIGS = [
    {
        "name": "lr_balanced_l2_c1",
        "class_weight": "balanced",
        "penalty": "l2",
        "C": 1.0,
        "solver": "liblinear",
    },
    {
        "name": "lr_balanced_l2_c025",
        "class_weight": "balanced",
        "penalty": "l2",
        "C": 0.25,
        "solver": "liblinear",
    },
    {
        "name": "lr_balanced_l1_c025",
        "class_weight": "balanced",
        "penalty": "l1",
        "C": 0.25,
        "solver": "liblinear",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train logistic regression on the pre-model window table.")
    parser.add_argument(
        "--input-csv",
        default="data/processed/bgl_multiblock_sample_pre_model_windows.csv",
    )
    parser.add_argument(
        "--population",
        choices=["all_windows", "post_gev"],
        default="all_windows",
    )
    parser.add_argument("--min-validation-recall", type=float, default=0.90)
    parser.add_argument("--output-report", default="outputs/reports/logistic_regression_bgl_multiblock.json")
    parser.add_argument("--output-model", default="outputs/models/logistic_regression_bgl_multiblock.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.population == "post_gev":
        df = df[df["population_post_gev"] == 1].copy()

    feature_columns = [feature for feature in SAFE_FEATURES_V1 if feature in df.columns]
    splits = _split_data(df, feature_columns)
    _validate_splits(splits)

    results = []
    fitted_models = {}
    for config in MODEL_CONFIGS:
        model = _build_model(config)
        model.fit(splits["train"]["x"], splits["train"]["y"])
        fitted_models[config["name"]] = model
        result = _evaluate_model(config["name"], model, splits, args.min_validation_recall)
        results.append(result)

    best = _select_best(results)
    best_model = fitted_models[best["name"]]

    report = {
        "input_csv": args.input_csv,
        "population": args.population,
        "feature_columns": feature_columns,
        "num_features": len(feature_columns),
        "split_summary": {
            split: {
                "num_windows": int(len(data["y"])),
                "positive_windows": int(np.sum(data["y"] == 1)),
                "negative_windows": int(np.sum(data["y"] == 0)),
                "positive_rate": _safe_ratio(float(np.sum(data["y"] == 1)), float(len(data["y"]))),
            }
            for split, data in splits.items()
        },
        "selection_rule": f"highest validation precision among thresholds with recall >= {args.min_validation_recall}",
        "best_model": best["name"],
        "best_threshold": best["threshold"],
        "results": results,
    }

    output_report = Path(args.output_report)
    output_model = Path(args.output_model)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump(
        {
            "model": best_model,
            "feature_columns": feature_columns,
            "threshold": best["threshold"],
            "population": args.population,
            "report": report,
        },
        output_model,
    )
    print(json.dumps({"best_model": best["name"], "report": str(output_report), "model": str(output_model)}, indent=2))


def _build_model(config: dict) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight=config["class_weight"],
                    penalty=config["penalty"],
                    C=config["C"],
                    solver=config["solver"],
                    max_iter=2000,
                    random_state=42,
                ),
            ),
        ]
    )


def _split_data(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict]:
    splits = {}
    for split in ["train", "validation", "test"]:
        part = df[df["split"] == split].copy()
        part = part[part["label"].isin([0, 1])]
        splits[split] = {
            "x": part[feature_columns],
            "y": part["label"].astype(int).to_numpy(),
            "population_post_gev": part["population_post_gev"].astype(int).to_numpy(),
        }
    return splits


def _validate_splits(splits: dict[str, dict]) -> None:
    for split, data in splits.items():
        labels = set(data["y"].tolist())
        if labels != {0, 1}:
            raise ValueError(f"{split} split must contain both classes, got {sorted(labels)}")


def _evaluate_model(
    name: str,
    model: Pipeline,
    splits: dict[str, dict],
    min_validation_recall: float,
) -> dict:
    probabilities = {
        split: model.predict_proba(data["x"])[:, 1]
        for split, data in splits.items()
    }
    threshold = _choose_threshold(
        y_true=splits["validation"]["y"],
        y_score=probabilities["validation"],
        min_recall=min_validation_recall,
    )
    return {
        "name": name,
        "threshold": threshold,
        "metrics": {
            split: _metrics(splits[split]["y"], probabilities[split], threshold)
            for split in ["train", "validation", "test"]
        },
        "coefficients": _coefficients(model),
    }


def _choose_threshold(y_true: np.ndarray, y_score: np.ndarray, min_recall: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    candidates = []
    for index, threshold in enumerate(thresholds):
        current_precision = precision[index]
        current_recall = recall[index]
        if current_recall >= min_recall:
            candidates.append((current_precision, current_recall, threshold))
    if not candidates:
        return 0.5
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    return float(candidates[0][2])


def _metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    precision = _safe_ratio(float(tp), float(tp + fp))
    recall = _safe_ratio(float(tp), float(tp + fn))
    f1 = None if precision is None or recall is None or precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": _safe_average_precision(y_true, y_score),
        "roc_auc": _safe_roc_auc(y_true, y_score),
        "forwarded_window_ratio": _safe_ratio(float(tp + fp), float(len(y_true))),
    }


def _coefficients(model: Pipeline) -> dict:
    classifier = model.named_steps["classifier"]
    feature_names = model.feature_names_in_
    coefficients = classifier.coef_[0]
    return {
        feature: float(coefficient)
        for feature, coefficient in sorted(
            zip(feature_names, coefficients),
            key=lambda item: abs(item[1]),
            reverse=True,
        )
    }


def _select_best(results: list[dict]) -> dict:
    def key(result: dict) -> tuple:
        metrics = result["metrics"]["validation"]
        return (
            metrics["precision"] or 0.0,
            metrics["recall"] or 0.0,
            metrics["f1"] or 0.0,
        )

    return max(results, key=key)


def _safe_average_precision(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(average_precision_score(y_true, y_score))


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(set(y_true.tolist())) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


if __name__ == "__main__":
    main()
