from thesis_log_anomaly.stats.pot import (
    count_exceedances,
    extract_cluster_sizes,
    extract_event_novelty_scores,
    extract_template_burst_scores,
    refine_windows_with_burst_pot,
    refine_windows_with_event_pot,
    refine_windows_with_event_pot_per_window,
    refine_windows_with_pot,
)


def refine_with_pot(suspicious_windows, config: dict):
    """Optionally apply POT inside suspicious windows to narrow candidates."""
    pot_config = config.get("pot", {})
    if not pot_config.get("enabled", False):
        return suspicious_windows

    candidate_windows = [
        window for window in suspicious_windows if window.get("evt", {}).get("is_suspicious")
    ]
    if not candidate_windows:
        return suspicious_windows

    threshold_quantile = float(pot_config.get("threshold_quantile", 0.95))
    score_target = pot_config.get("score_target", "event_novelty")
    per_window = bool(pot_config.get("per_window", False))
    if not per_window:
        values = (
            extract_event_novelty_scores(
                candidate_windows,
                score_field="novelty_score" if score_target == "event_novelty" else "rarity_score",
            )
            if score_target in {"event_novelty", "historical_rarity"}
            else extract_template_burst_scores(candidate_windows)
            if score_target == "template_burst"
            else extract_cluster_sizes(candidate_windows)
        )
        if count_exceedances(values, threshold_quantile) < int(pot_config.get("min_exceedances", 20)):
            return suspicious_windows

    if score_target in {"event_novelty", "historical_rarity"}:
        try:
            if per_window:
                result = refine_windows_with_event_pot_per_window(
                    candidate_windows,
                    threshold_quantile=threshold_quantile,
                    tail_alpha=float(pot_config.get("tail_alpha", 0.10)),
                    score_field="novelty_score" if score_target == "event_novelty" else "rarity_score",
                    score_target=score_target,
                    min_exceedances=int(pot_config.get("min_exceedances", 20)),
                )
            else:
                result = refine_windows_with_event_pot(
                    candidate_windows,
                    threshold_quantile=threshold_quantile,
                    tail_alpha=float(pot_config.get("tail_alpha", 0.10)),
                    score_field="novelty_score" if score_target == "event_novelty" else "rarity_score",
                    score_target=score_target,
                )
        except ValueError:
            return suspicious_windows
    elif score_target == "template_burst":
        try:
            result = refine_windows_with_burst_pot(
                candidate_windows,
                threshold_quantile=threshold_quantile,
                tail_alpha=float(pot_config.get("tail_alpha", 0.10)),
            )
        except ValueError:
            return suspicious_windows
    else:
        try:
            result = refine_windows_with_pot(
                candidate_windows,
                threshold_quantile=threshold_quantile,
                tail_alpha=float(pot_config.get("tail_alpha", 0.10)),
            )
        except ValueError:
            return suspicious_windows
    by_window_id = {window["window_id"]: window for window in result["windows"]}

    refined = []
    for window in suspicious_windows:
        refined.append(by_window_id.get(window.get("window_id"), window))
    return refined
