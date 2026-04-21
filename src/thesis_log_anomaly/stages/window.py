import math

from thesis_log_anomaly.parsing.drain import parse_with_drain
from thesis_log_anomaly.stages.template import annotate_record_novelty, build_drain_settings
from thesis_log_anomaly.windowing.sliding import build_count_windows, build_group_windows, build_time_windows


def build_windows(parsed_stream, config: dict):
    """Construct overlapping sliding windows over ordered parsed records."""
    window_config = config.get("windowing", {})
    data_config = config.get("data", {})
    mode = window_config.get("mode", "count")
    grouping_key = window_config.get("grouping_key")
    records = parsed_stream["stream_records"] if isinstance(parsed_stream, dict) else parsed_stream
    history_template_counts = (
        parsed_stream.get("history_template_summary", {}).get("template_counts", {})
        if isinstance(parsed_stream, dict)
        else {}
    )
    rare_threshold = int(data_config.get("rare_template_count_threshold", 2))

    if grouping_key:
        windows = build_group_windows(records, grouping_key=str(grouping_key))
    elif mode == "time":
        windows = build_time_windows(
            records,
            duration_hours=int(window_config.get("size", 24)),
            stride_hours=int(window_config.get("stride", 12)),
        )
    else:
        windows = build_count_windows(
            records,
            size=int(window_config.get("size", 100)),
            stride=int(window_config.get("stride", 50)),
        )

    if config.get("parsing", {}).get("scope", "stream") == "per_window":
        windows = _apply_per_window_drain(
            windows,
            config,
            history_template_counts,
            rare_threshold,
        )

    return [
        _annotate_window(
            window,
            window_config.get("label_rule", "any_anomaly"),
            history_template_counts,
            rare_threshold,
        )
        for window in windows
    ]


def _apply_per_window_drain(
    windows: list[dict],
    config: dict,
    history_template_counts: dict[str, int],
    rare_threshold: int,
) -> list[dict]:
    settings = build_drain_settings(config)
    reparsed_windows: list[dict] = []

    for window in windows:
        parsed_records = parse_with_drain(
            [record.get("message") or record.get("log", "") for record in window.get("records", [])],
            settings=settings,
        )
        reparsed_records = []
        for record, parsed in zip(window.get("records", []), parsed_records):
            enriched = dict(record)
            enriched["template"] = parsed.get("template")
            enriched["drain"] = parsed
            reparsed_records.append(
                annotate_record_novelty(
                    enriched,
                    history_template_counts=history_template_counts,
                    rare_threshold=rare_threshold,
                )
            )

        updated_window = dict(window)
        updated_window["records"] = reparsed_records
        reparsed_windows.append(updated_window)

    return reparsed_windows


def _annotate_window(
    window: dict,
    label_rule: str,
    history_template_counts: dict[str, int],
    rare_threshold: int,
) -> dict:
    records = window.get("records", [])
    cluster_groups: dict[int, dict] = {}
    for record in records:
        cluster_id = record.get("drain", {}).get("cluster_id")
        if cluster_id is None:
            continue
        group = cluster_groups.setdefault(
            cluster_id,
            {
                "cluster_id": cluster_id,
                "size": 0,
                "template": record.get("template"),
                "examples": [],
                "event_ids": [],
                "num_anomalous_records": 0,
                "num_normal_records": 0,
            },
        )
        group["size"] += 1
        if len(group["examples"]) < 3:
            group["examples"].append(record.get("message") or record.get("log", ""))
        if record.get("event_id") is not None:
            group["event_ids"].append(record["event_id"])
        if record.get("label") == 1:
            group["num_anomalous_records"] += 1
        elif record.get("label") == 0:
            group["num_normal_records"] += 1

    clusters = sorted(cluster_groups.values(), key=lambda item: item["size"], reverse=True)
    labeled_records = [record for record in records if record.get("label") is not None]
    anomalous_records = [record for record in labeled_records if record.get("label") == 1]

    annotated = dict(window)
    annotated["clusters"] = clusters
    annotated["template_summary"] = _template_summary(records)
    annotated["feature_summary"] = _feature_summary(
        records,
        clusters,
        annotated["template_summary"],
        history_template_counts,
        rare_threshold,
    )
    annotated["ground_truth"] = {
        "label_rule": label_rule,
        "num_records": len(records),
        "num_labeled_records": len(labeled_records),
        "num_anomalous_records": len(anomalous_records),
        "window_label": _window_label(anomalous_records, records, label_rule),
    }
    return annotated


def _template_summary(records: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for record in records:
        template = record.get("template") or "<unknown>"
        counts[template] = counts.get(template, 0) + 1
    return {
        "count": len(records),
        "unique_templates": len(counts),
        "template_counts": dict(sorted(counts.items(), key=lambda item: item[1], reverse=True)),
    }


def _feature_summary(
    records: list[dict],
    clusters: list[dict],
    template_summary: dict,
    history_template_counts: dict[str, int],
    rare_threshold: int,
) -> dict:
    num_records = len(records)
    cluster_sizes = [cluster["size"] for cluster in clusters if cluster.get("size") is not None]
    new_template_count = sum(
        1
        for record in records
        if record.get("drain", {}).get("is_new_template")
    )
    novelty_scores = [
        float(record.get("novelty", {}).get("novelty_score", 0.0))
        for record in records
    ]
    rarity_scores = [
        float(record.get("novelty", {}).get("rarity_score", 0.0))
        for record in records
    ]
    historically_rare_count = 0
    unseen_in_history_count = 0
    for record in records:
        template = record.get("template") or "<unknown>"
        historical_count = history_template_counts.get(template, 0)
        if historical_count == 0:
            unseen_in_history_count += 1
        if historical_count <= rare_threshold:
            historically_rare_count += 1

    entropy = 0.0
    if num_records > 0:
        for count in template_summary["template_counts"].values():
            probability = count / num_records
            entropy -= probability * math.log2(probability)

    max_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    mean_cluster_size = (sum(cluster_sizes) / len(cluster_sizes)) if cluster_sizes else 0.0

    return {
        "num_records": num_records,
        "num_clusters": len(clusters),
        "unique_templates": template_summary["unique_templates"],
        "unique_template_ratio": (template_summary["unique_templates"] / num_records) if num_records else 0.0,
        "template_entropy": entropy,
        "max_cluster_size": max_cluster_size,
        "mean_cluster_size": mean_cluster_size,
        "max_cluster_ratio": (max_cluster_size / num_records) if num_records else 0.0,
        "new_template_count": new_template_count,
        "new_template_ratio": (new_template_count / num_records) if num_records else 0.0,
        "historically_rare_template_count": historically_rare_count,
        "historically_rare_template_ratio": (historically_rare_count / num_records) if num_records else 0.0,
        "unseen_in_history_count": unseen_in_history_count,
        "unseen_in_history_ratio": (unseen_in_history_count / num_records) if num_records else 0.0,
        "max_event_novelty_score": max(novelty_scores) if novelty_scores else 0.0,
        "mean_event_novelty_score": (sum(novelty_scores) / len(novelty_scores)) if novelty_scores else 0.0,
        "max_event_rarity_score": max(rarity_scores) if rarity_scores else 0.0,
        "mean_event_rarity_score": (sum(rarity_scores) / len(rarity_scores)) if rarity_scores else 0.0,
        "novelty_score_variance": _variance(novelty_scores),
        "rarity_score_variance": _variance(rarity_scores),
        "novelty_magnitude_score": (
            ((max_cluster_size / num_records) if num_records else 0.0)
            + ((template_summary["unique_templates"] / num_records) if num_records else 0.0)
            + ((historically_rare_count / num_records) if num_records else 0.0)
            + ((new_template_count / num_records) if num_records else 0.0)
        ),
    }


def _variance(values: list[float]) -> float:
    if not values:
        return 0.0
    mean_value = sum(values) / len(values)
    return sum((value - mean_value) ** 2 for value in values) / len(values)


def _window_label(anomalous_records: list[dict], records: list[dict], label_rule: str) -> int | None:
    labeled_records = [record for record in records if record.get("label") is not None]
    if not labeled_records:
        return None
    if label_rule == "any_anomaly":
        return 1 if anomalous_records else 0
    raise ValueError(f"Unsupported window label rule: {label_rule}")
