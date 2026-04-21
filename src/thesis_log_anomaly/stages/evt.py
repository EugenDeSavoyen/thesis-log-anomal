from thesis_log_anomaly.stats.gev import score_windows_with_gev, score_windows_with_gev_ensemble


def score_windows_with_evt(parsed_windows, config: dict):
    """Fit EVT on window statistics and return suspicious windows."""
    evt_config = config.get("evt", {})
    method = evt_config.get("method", "gev")
    statistic = evt_config.get("target_statistic", "max_cluster_size")
    alpha = float(evt_config.get("significance_level", 0.90))
    min_windows_for_fit = int(evt_config.get("min_windows_for_fit", 3))
    ensemble_enabled = bool(evt_config.get("ensemble_enabled", False))
    ensemble_statistics = evt_config.get("ensemble_statistics", [])
    if isinstance(ensemble_statistics, str):
        ensemble_statistics = [item.strip() for item in ensemble_statistics.split(",") if item.strip()]

    if method != "gev":
        raise ValueError(f"Unsupported EVT method: {method}")

    usable_windows = [window for window in parsed_windows if window.get("clusters")]
    if len(usable_windows) < min_windows_for_fit:
        return parsed_windows

    if ensemble_enabled and ensemble_statistics:
        result = score_windows_with_gev_ensemble(
            usable_windows,
            statistics=ensemble_statistics,
            alpha=alpha,
        )
        return result["windows"]

    result = score_windows_with_gev(
        usable_windows,
        alpha=alpha,
        statistic=statistic,
    )
    return result["windows"]
