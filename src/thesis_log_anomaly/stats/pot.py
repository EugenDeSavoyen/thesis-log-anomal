from __future__ import annotations

from dataclasses import dataclass

from scipy.stats import genpareto

from thesis_log_anomaly.stats.thresholds import quantile_threshold


@dataclass
class PotFitResult:
    threshold_u: float
    threshold_z: float
    gpd_shape: float
    gpd_loc: float
    gpd_scale: float
    threshold_quantile: float
    tail_alpha: float
    num_exceedances: int
    method: str = "pot_gpd"


def fit_peak_over_threshold(
    values: list[float],
    threshold_quantile: float = 0.95,
    tail_alpha: float = 0.10,
) -> dict:
    """Fit POT/GPD and compute an absolute anomaly threshold."""
    if not values:
        raise ValueError("values must not be empty")

    threshold_u = quantile_threshold(values, threshold_quantile)
    if threshold_u is None:
        raise ValueError("could not compute POT threshold")

    exceedances = [value - threshold_u for value in values if value > threshold_u]
    if not exceedances:
        raise ValueError("no exceedances above POT threshold")

    gpd_shape, gpd_loc, gpd_scale = genpareto.fit(exceedances, floc=0)
    gpd_shape = float(gpd_shape)
    gpd_loc = float(gpd_loc)
    gpd_scale = float(gpd_scale)
    if gpd_scale <= 0:
        raise ValueError("Estimated GPD scale must be positive")

    exceedance_threshold = float(
        genpareto.ppf(1 - tail_alpha, c=gpd_shape, loc=0, scale=gpd_scale)
    )
    threshold_z = float(threshold_u + exceedance_threshold)

    return PotFitResult(
        threshold_u=float(threshold_u),
        threshold_z=threshold_z,
        gpd_shape=gpd_shape,
        gpd_loc=gpd_loc,
        gpd_scale=gpd_scale,
        threshold_quantile=threshold_quantile,
        tail_alpha=tail_alpha,
        num_exceedances=len(exceedances),
    ).__dict__


def extract_cluster_sizes(windows: list[dict]) -> list[float]:
    sizes: list[float] = []
    for window in windows:
        for cluster in window.get("clusters", []):
            size = cluster.get("size")
            if size is not None:
                sizes.append(float(size))
    return sizes


def extract_event_novelty_scores(windows: list[dict], score_field: str = "novelty_score") -> list[float]:
    scores: list[float] = []
    for window in windows:
        for record in window.get("records", []):
            score = record.get("novelty", {}).get(score_field)
            if score is not None:
                scores.append(float(score))
    return scores


def extract_template_burst_scores(windows: list[dict]) -> list[float]:
    scores: list[float] = []
    for window in windows:
        burst_scores = _compute_burst_scores(window.get("records", []))
        scores.extend(float(score) for score in burst_scores if score is not None)
    return scores


def count_exceedances(values: list[float], threshold_quantile: float) -> int:
    threshold_u = quantile_threshold(values, threshold_quantile)
    if threshold_u is None:
        return 0
    return sum(1 for value in values if value > threshold_u)


def refine_windows_with_pot(
    windows: list[dict],
    threshold_quantile: float = 0.95,
    tail_alpha: float = 0.10,
) -> dict:
    """Apply POT/GPD to cluster sizes and flag extreme clusters within windows."""
    sizes = extract_cluster_sizes(windows)
    fit = fit_peak_over_threshold(
        sizes,
        threshold_quantile=threshold_quantile,
        tail_alpha=tail_alpha,
    )
    threshold_z = fit["threshold_z"]
    threshold_u = fit["threshold_u"]

    refined_windows = []
    for window in windows:
        refined_clusters = []
        num_pot_anomalies = 0
        for cluster in window.get("clusters", []):
            refined_cluster = dict(cluster)
            size = refined_cluster.get("size")
            is_exceedance = size is not None and float(size) > threshold_u
            is_pot_anomaly = size is not None and float(size) > threshold_z
            refined_cluster["pot"] = {
                "threshold_u": threshold_u,
                "threshold_z": threshold_z,
                "is_exceedance": bool(is_exceedance),
                "is_anomaly": bool(is_pot_anomaly),
            }
            if is_pot_anomaly:
                num_pot_anomalies += 1
            refined_clusters.append(refined_cluster)

        refined_window = dict(window)
        refined_window["clusters"] = refined_clusters
        refined_window["pot"] = {
            "fit": fit,
            "num_pot_anomalies": num_pot_anomalies,
            "has_pot_anomaly": num_pot_anomalies > 0,
        }
        refined_windows.append(refined_window)

    return {
        "fit": fit,
        "windows": refined_windows,
    }


def refine_windows_with_event_pot(
    windows: list[dict],
    threshold_quantile: float = 0.95,
    tail_alpha: float = 0.10,
    score_field: str = "novelty_score",
    score_target: str = "event_novelty",
) -> dict:
    """Apply POT/GPD to per-event scores within suspicious windows."""
    scores = extract_event_novelty_scores(windows, score_field=score_field)
    fit = fit_peak_over_threshold(
        scores,
        threshold_quantile=threshold_quantile,
        tail_alpha=tail_alpha,
    )
    threshold_u = fit["threshold_u"]
    threshold_z = fit["threshold_z"]

    refined_windows = []
    for window in windows:
        refined_records = []
        candidate_event_ids: list[int] = []
        for record in window.get("records", []):
            refined_record = dict(record)
            score_value = float(record.get("novelty", {}).get(score_field, 0.0))
            refined_record["pot"] = {
                "threshold_u": threshold_u,
                "threshold_z": threshold_z,
                "score": score_value,
                "score_target": score_target,
                "is_exceedance": score_value > threshold_u,
                "is_anomaly": score_value > threshold_z,
            }
            if refined_record["pot"]["is_anomaly"] and record.get("event_id") is not None:
                candidate_event_ids.append(record["event_id"])
            refined_records.append(refined_record)

        refined_window = dict(window)
        refined_window["records"] = refined_records
        refined_window["pot"] = {
            "fit": fit,
            "candidate_event_ids": candidate_event_ids,
            "num_pot_anomalies": len(candidate_event_ids),
            "has_pot_anomaly": bool(candidate_event_ids),
            "score_target": score_target,
        }
        refined_windows.append(refined_window)

    return {
        "fit": fit,
        "windows": refined_windows,
    }


def refine_windows_with_event_pot_per_window(
    windows: list[dict],
    threshold_quantile: float = 0.95,
    tail_alpha: float = 0.10,
    score_field: str = "novelty_score",
    score_target: str = "event_novelty",
    min_exceedances: int = 1,
) -> dict:
    """Fit POT separately inside each suspicious window."""
    refined_windows = []
    fits: dict[int, dict] = {}

    for window in windows:
        scores = [
            float(record.get("novelty", {}).get(score_field, 0.0))
            for record in window.get("records", [])
            if record.get("novelty", {}).get(score_field) is not None
        ]
        if not scores or count_exceedances(scores, threshold_quantile) < min_exceedances:
            refined_window = dict(window)
            refined_window["pot"] = {
                "fit": None,
                "candidate_event_ids": [],
                "num_pot_anomalies": 0,
                "has_pot_anomaly": False,
                "score_target": score_target,
            }
            refined_windows.append(refined_window)
            continue

        fit = fit_peak_over_threshold(
            scores,
            threshold_quantile=threshold_quantile,
            tail_alpha=tail_alpha,
        )
        fits[window["window_id"]] = fit
        threshold_u = fit["threshold_u"]
        threshold_z = fit["threshold_z"]

        refined_records = []
        candidate_event_ids: list[int] = []
        for record in window.get("records", []):
            refined_record = dict(record)
            score_value = float(record.get("novelty", {}).get(score_field, 0.0))
            refined_record["pot"] = {
                "threshold_u": threshold_u,
                "threshold_z": threshold_z,
                "score": score_value,
                "score_target": score_target,
                "is_exceedance": score_value > threshold_u,
                "is_anomaly": score_value > threshold_z,
            }
            if refined_record["pot"]["is_anomaly"] and record.get("event_id") is not None:
                candidate_event_ids.append(record["event_id"])
            refined_records.append(refined_record)

        refined_window = dict(window)
        refined_window["records"] = refined_records
        refined_window["pot"] = {
            "fit": fit,
            "candidate_event_ids": candidate_event_ids,
            "num_pot_anomalies": len(candidate_event_ids),
            "has_pot_anomaly": bool(candidate_event_ids),
            "score_target": score_target,
        }
        refined_windows.append(refined_window)

    return {
        "fits": fits,
        "windows": refined_windows,
    }


def refine_windows_with_burst_pot(
    windows: list[dict],
    threshold_quantile: float = 0.95,
    tail_alpha: float = 0.10,
) -> dict:
    """Apply POT/GPD to unseen/new-template burst scores within suspicious windows."""
    scores = extract_template_burst_scores(windows)
    fit = fit_peak_over_threshold(
        scores,
        threshold_quantile=threshold_quantile,
        tail_alpha=tail_alpha,
    )
    threshold_u = fit["threshold_u"]
    threshold_z = fit["threshold_z"]

    refined_windows = []
    for window in windows:
        burst_scores = _compute_burst_scores(window.get("records", []))
        refined_records = []
        candidate_event_ids: list[int] = []
        for record, burst_score in zip(window.get("records", []), burst_scores):
            refined_record = dict(record)
            refined_record["pot"] = {
                "threshold_u": threshold_u,
                "threshold_z": threshold_z,
                "score": float(burst_score),
                "score_target": "template_burst",
                "is_exceedance": float(burst_score) > threshold_u,
                "is_anomaly": float(burst_score) > threshold_z,
            }
            if refined_record["pot"]["is_anomaly"] and record.get("event_id") is not None:
                candidate_event_ids.append(record["event_id"])
            refined_records.append(refined_record)

        refined_window = dict(window)
        refined_window["records"] = refined_records
        refined_window["pot"] = {
            "fit": fit,
            "candidate_event_ids": candidate_event_ids,
            "num_pot_anomalies": len(candidate_event_ids),
            "has_pot_anomaly": bool(candidate_event_ids),
            "score_target": "template_burst",
        }
        refined_windows.append(refined_window)

    return {
        "fit": fit,
        "windows": refined_windows,
    }


def _compute_burst_scores(records: list[dict]) -> list[int]:
    """Assign each event the length of its contiguous unseen/new-template burst."""
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
        for idx in range(start, end):
            scores[idx] = burst_len
        start = end
    return scores
