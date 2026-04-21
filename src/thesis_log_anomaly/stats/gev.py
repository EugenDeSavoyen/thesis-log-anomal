from __future__ import annotations

from dataclasses import dataclass

from lmoments3 import distr
from scipy.stats import genextreme


@dataclass
class GevFitResult:
    shape: float
    loc: float
    scale: float
    threshold: float
    alpha: float
    n: int
    method: str = "gev_lmoments"


def fit_gev_block_maxima(values: list[float], alpha: float = 0.90) -> dict:
    """Fit a GEV distribution to block maxima and compute an upper-tail threshold."""
    if len(values) < 3:
        raise ValueError("At least 3 values are required to fit GEV with L-moments")

    fit = distr.gev.lmom_fit(values)
    shape = float(fit["c"])
    loc = float(fit["loc"])
    scale = float(fit["scale"])
    if scale <= 0:
        raise ValueError("Estimated GEV scale must be positive")

    threshold = float(genextreme.ppf(alpha, c=shape, loc=loc, scale=scale))
    return GevFitResult(
        shape=shape,
        loc=loc,
        scale=scale,
        threshold=threshold,
        alpha=alpha,
        n=len(values),
    ).__dict__


def extract_window_statistics(parsed_windows: list[dict], statistic: str = "max_cluster_size") -> list[dict]:
    """Extract per-window statistics for EVT fitting from parsed window output."""
    window_stats = []
    for window in parsed_windows:
        clusters = window.get("clusters", [])
        sizes = [float(cluster["size"]) for cluster in clusters if cluster.get("size") is not None]
        if not sizes:
            continue

        feature_summary = window.get("feature_summary", {})
        max_cluster_size = max(sizes)
        mean_cluster_size = sum(sizes) / len(sizes)
        value = _select_statistic(
            statistic=statistic,
            feature_summary=feature_summary,
            max_cluster_size=max_cluster_size,
            mean_cluster_size=mean_cluster_size,
        )
        record = {
            "window_id": window.get("window_id"),
            "start_time": window.get("start_time"),
            "end_time": window.get("end_time"),
            "num_clusters": len(sizes),
            "max_cluster_size": max_cluster_size,
            "mean_cluster_size": mean_cluster_size,
            "value": value,
        }
        window_stats.append(record)
    return window_stats


def score_windows_with_gev(parsed_windows: list[dict], alpha: float = 0.90, statistic: str = "max_cluster_size") -> dict:
    """Fit GEV on window statistics and annotate windows that exceed the threshold."""
    window_stats = extract_window_statistics(parsed_windows, statistic=statistic)
    values = [row["value"] for row in window_stats]
    fit = fit_gev_block_maxima(values, alpha=alpha)
    threshold = fit["threshold"]

    suspicious_window_ids = {
        row["window_id"]
        for row in window_stats
        if row["value"] > threshold
    }

    scored_windows = []
    for window in parsed_windows:
        clusters = window.get("clusters", [])
        sizes = [float(cluster["size"]) for cluster in clusters if cluster.get("size") is not None]
        max_cluster_size = max(sizes) if sizes else None
        mean_cluster_size = (sum(sizes) / len(sizes)) if sizes else None
        value = _select_statistic(
            statistic=statistic,
            feature_summary=window.get("feature_summary", {}),
            max_cluster_size=max_cluster_size or 0.0,
            mean_cluster_size=mean_cluster_size or 0.0,
        )

        scored_window = dict(window)
        scored_window["evt"] = {
            "method": "gev",
            "statistic": statistic,
            "value": value,
            "threshold": threshold,
            "is_suspicious": window.get("window_id") in suspicious_window_ids,
            "fit": fit,
        }
        scored_windows.append(scored_window)

    return {
        "fit": fit,
        "window_stats": window_stats,
        "windows": scored_windows,
    }


def score_windows_with_gev_ensemble(
    parsed_windows: list[dict],
    statistics: list[str],
    alpha: float = 0.90,
) -> dict:
    """Run multiple GEV detectors in parallel and combine them with an OR rule."""
    per_stat_results: dict[str, dict] = {}
    by_window_id: dict[int, dict] = {window["window_id"]: dict(window) for window in parsed_windows}

    for statistic in statistics:
        result = score_windows_with_gev(parsed_windows, alpha=alpha, statistic=statistic)
        per_stat_results[statistic] = result
        for window in result["windows"]:
            aggregate = by_window_id[window["window_id"]]
            aggregate.setdefault("evt_ensemble", {"members": {}, "is_suspicious": False, "triggered_statistics": []})
            member_evt = dict(window["evt"])
            aggregate["evt_ensemble"]["members"][statistic] = member_evt
            if member_evt.get("is_suspicious"):
                aggregate["evt_ensemble"]["is_suspicious"] = True
                aggregate["evt_ensemble"]["triggered_statistics"].append(statistic)

    combined_windows = []
    for window_id in sorted(by_window_id):
        aggregate = by_window_id[window_id]
        aggregate["evt"] = {
            "method": "gev_ensemble",
            "statistic": "ensemble_or",
            "value": None,
            "threshold": None,
            "is_suspicious": aggregate.get("evt_ensemble", {}).get("is_suspicious", False),
            "fit": None,
        }
        combined_windows.append(aggregate)

    return {
        "statistics": statistics,
        "results": per_stat_results,
        "windows": combined_windows,
    }


def _select_statistic(
    statistic: str,
    feature_summary: dict,
    max_cluster_size: float,
    mean_cluster_size: float,
) -> float:
    if statistic == "max_cluster_size":
        return float(max_cluster_size)
    if statistic == "mean_cluster_size":
        return float(mean_cluster_size)
    if statistic in feature_summary:
        return float(feature_summary[statistic])
    raise ValueError(f"Unsupported GEV statistic: {statistic}")
