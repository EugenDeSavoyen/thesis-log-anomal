from thesis_log_anomaly.parsing.drain import (
    DrainSettings,
    parse_stream_with_drain,
    parse_stream_with_dual_drain,
    parse_with_drain,
)
from thesis_log_anomaly.parsing.templates import summarize_templates


def build_drain_settings(config: dict) -> DrainSettings:
    parsing_config = config.get("parsing", {})
    drain_config = parsing_config.get("drain", {})
    return DrainSettings(
        ini_path=parsing_config.get("ini_path"),
        depth=int(drain_config.get("depth", 4)),
        similarity_threshold=float(drain_config.get("similarity_threshold", 0.5)),
        max_children=int(drain_config.get("max_children", 100)),
        max_clusters=drain_config.get("max_clusters"),
        full_search_strategy=parsing_config.get("reference_full_search_strategy", "fallback"),
    )


def split_history_and_stream(records, config: dict) -> tuple[list[dict], list[dict]]:
    data_config = config.get("data", {})
    history_fraction = float(data_config.get("history_fraction", 0.3))
    split_index = max(1, int(len(records) * history_fraction)) if records else 0
    return records[:split_index], records[split_index:]


def parse_templates(records, config: dict):
    """Parse an ordered stream of log records with historical Drain warm-up."""
    parsing_config = config.get("parsing", {})
    data_config = config.get("data", {})
    settings = build_drain_settings(config)
    rare_threshold = int(data_config.get("rare_template_count_threshold", 2))
    history_records, stream_records = split_history_and_stream(records, config)
    history_parsed = parse_with_drain(
        [record.get("message") or record.get("log", "") for record in history_records],
        settings=settings,
    ) if history_records else []
    history_template_summary = summarize_templates(history_parsed)
    history_template_counts = history_template_summary["template_counts"]

    if parsing_config.get("scope", "stream") == "per_window":
        return {
            "history_records": history_records,
            "history_template_summary": history_template_summary,
            "stream_records": [dict(record) for record in stream_records],
            "stream_summary": summarize_templates([]),
            "history_size": len(history_records),
            "stream_size": len(stream_records),
        }

    if bool(parsing_config.get("dual_state_reference", True)):
        parsed_stream = parse_stream_with_dual_drain(
            stream_records,
            settings=settings,
            history_records=history_records,
            update_model_online=bool(parsing_config.get("update_model_online", True)),
        )
    else:
        parsed_stream = parse_stream_with_drain(
            stream_records,
            settings=settings,
            history_records=history_records,
            update_model_online=bool(parsing_config.get("update_model_online", True)),
        )
    enriched_stream = [
        annotate_record_novelty(record, history_template_counts, rare_threshold)
        for record in parsed_stream
    ]

    return {
        "history_records": history_records,
        "history_template_summary": history_template_summary,
        "stream_records": enriched_stream,
        "stream_summary": summarize_templates([record["drain"] for record in enriched_stream]),
        "history_size": len(history_records),
        "stream_size": len(enriched_stream),
    }


def annotate_record_novelty(
    record: dict,
    history_template_counts: dict[str, int],
    rare_threshold: int,
) -> dict:
    template = record.get("template") or "<unknown>"
    historical_count = int(history_template_counts.get(template, 0))
    unseen_in_history = historical_count == 0
    historically_rare = historical_count <= rare_threshold
    is_new_template = bool(record.get("drain", {}).get("is_new_template"))
    is_known_to_reference = bool(record.get("drain_reference", {}).get("is_known_template", not unseen_in_history))
    rarity_score = 1.0 / (historical_count + 1)
    novelty_score = (
        rarity_score
        + (1.0 if unseen_in_history else 0.0)
        + (0.75 if is_new_template else 0.0)
        + (0.25 if historically_rare else 0.0)
    )

    enriched = dict(record)
    enriched["novelty"] = {
        "historical_count": historical_count,
        "unseen_in_history": unseen_in_history,
        "historically_rare": historically_rare,
        "is_new_template": is_new_template,
        "is_known_to_reference": is_known_to_reference,
        "rarity_score": rarity_score,
        "novelty_score": novelty_score,
    }
    return enriched
