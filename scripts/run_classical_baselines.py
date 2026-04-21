from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description="Run classical log anomaly baselines on prepared window features.")
    parser.add_argument("--input-csv", default="data/processed/bgl_multiblock_sample_pre_model_windows.csv")
    parser.add_argument("--population", choices=["all_windows", "post_gev"], default="all_windows")
    parser.add_argument("--min-validation-recall", type=float, default=0.90)
    parser.add_argument("--output-report", default="outputs/reports/classical_baselines_bgl_multiblock.json")
    parser.add_argument("--output-models", default="outputs/models/classical_baselines_bgl_multiblock.joblib")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    if args.population == "post_gev":
        df = df[df["population_post_gev"] == 1].copy()

    feature_families = _feature_families(df)
    results = []
    fitted = {}
    for family_name, columns in feature_families.items():
        splits = _split_data(df, columns)
        _validate_splits(splits, family_name)

        experiments = [
            ("logistic_regression", _fit_logistic(splits)),
            ("pca_spe", _fit_pca(splits)),
            ("isolation_forest", _fit_isolation_forest(splits)),
        ]
        for model_name, scorer in experiments:
            result = _evaluate_scorer(
                model_name=model_name,
                feature_family=family_name,
                feature_columns=columns,
                scorer=scorer,
                splits=splits,
                min_validation_recall=args.min_validation_recall,
            )
            results.append(result)
            fitted[f"{family_name}::{model_name}"] = scorer["model"]

    best = _select_best(results)
    report = {
        "input_csv": args.input_csv,
        "population": args.population,
        "min_validation_recall": args.min_validation_recall,
        "feature_families": {name: columns for name, columns in feature_families.items()},
        "split_summary": _split_summary(df),
        "selection_rule": f"highest validation precision among thresholds with recall >= {args.min_validation_recall}",
        "best": {
            "model_name": best["model_name"],
            "feature_family": best["feature_family"],
            "threshold": best["threshold"],
        },
        "results": results,
    }

    output_report = Path(args.output_report)
    output_models = Path(args.output_models)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_models.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    joblib.dump({"models": fitted, "report": report}, output_models)
    print(
        json.dumps(
            {
                "best": report["best"],
                "report": str(output_report),
                "models": str(output_models),
            },
            indent=2,
        )
    )


def _feature_families(df: pd.DataFrame) -> dict[str, list[str]]:
    compact = [feature for feature in COMPACT_FEATURES if feature in df.columns]
    event_counts = [column for column in df.columns if column.startswith("event_count__")]
    if not event_counts:
        raise ValueError("No event_count__ columns found. Regenerate the pre-model table with event_count_features enabled.")
    return {
        "compact": compact,
        "event_counts": event_counts,
        "combined": compact + event_counts,
    }


def _split_data(df: pd.DataFrame, feature_columns: list[str]) -> dict[str, dict]:
    splits = {}
    for split in ["train", "validation", "test"]:
        part = df[df["split"] == split].copy()
        part = part[part["label"].isin([0, 1])]
        splits[split] = {
            "x": part[feature_columns],
            "y": part["label"].astype(int).to_numpy(),
        }
    return splits


def _validate_splits(splits: dict[str, dict], family_name: str) -> None:
    for split, data in splits.items():
        labels = set(data["y"].tolist())
        if labels != {0, 1}:
            raise ValueError(f"{family_name}: {split} split must contain both classes, got {sorted(labels)}")


def _fit_logistic(splits: dict[str, dict]) -> dict:
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
    model.fit(splits["train"]["x"], splits["train"]["y"])
    return {"model": model, "score_fn": lambda data: model.predict_proba(data)[:, 1]}


def _fit_pca(splits: dict[str, dict]) -> dict:
    train_x = splits["train"]["x"]
    train_y = splits["train"]["y"]
    normal_train_x = train_x[train_y == 0]
    n_components = max(1, min(normal_train_x.shape[0] - 1, normal_train_x.shape[1], 10))
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
        ]
    )
    model.fit(normal_train_x)

    def score_fn(data: pd.DataFrame) -> np.ndarray:
        transformed = model.transform(data)
        reconstructed = model.named_steps["pca"].inverse_transform(transformed)
        scaled = model.named_steps["scaler"].transform(model.named_steps["imputer"].transform(data))
        return np.sum((scaled - reconstructed) ** 2, axis=1)

    return {"model": model, "score_fn": score_fn, "n_components": n_components}


def _fit_isolation_forest(splits: dict[str, dict]) -> dict:
    train_x = splits["train"]["x"]
    train_y = splits["train"]["y"]
    normal_train_x = train_x[train_y == 0]
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "forest",
                IsolationForest(
                    n_estimators=200,
                    contamination="auto",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(normal_train_x)
    return {"model": model, "score_fn": lambda data: -model.decision_function(data)}


def _evaluate_scorer(
    model_name: str,
    feature_family: str,
    feature_columns: list[str],
    scorer: dict,
    splits: dict[str, dict],
    min_validation_recall: float,
) -> dict:
    scores = {split: scorer["score_fn"](data["x"]) for split, data in splits.items()}
    threshold = _choose_threshold(splits["validation"]["y"], scores["validation"], min_validation_recall)
    return {
        "model_name": model_name,
        "feature_family": feature_family,
        "num_features": len(feature_columns),
        "threshold": threshold,
        "extra": {key: value for key, value in scorer.items() if key not in {"model", "score_fn"}},
        "metrics": {
            split: _metrics(splits[split]["y"], scores[split], threshold)
            for split in ["train", "validation", "test"]
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
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": _safe_average_precision(y_true, scores),
        "roc_auc": _safe_roc_auc(y_true, scores),
        "forwarded_window_ratio": _safe_ratio(float(tp + fp), float(len(y_true))),
    }


def _split_summary(df: pd.DataFrame) -> dict:
    return {
        split: {
            "num_windows": int(len(part)),
            "positive_windows": int((part["label"] == 1).sum()),
            "negative_windows": int((part["label"] == 0).sum()),
        }
        for split, part in ((split, df[df["split"] == split]) for split in ["train", "validation", "test"])
    }


def _select_best(results: list[dict]) -> dict:
    return max(
        results,
        key=lambda result: (
            result["metrics"]["validation"]["precision"] or 0.0,
            result["metrics"]["validation"]["recall"] or 0.0,
            result["metrics"]["validation"]["f1"] or 0.0,
        ),
    )


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
