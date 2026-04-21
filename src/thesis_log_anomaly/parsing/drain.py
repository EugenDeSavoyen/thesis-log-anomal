from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig


@dataclass
class DrainSettings:
    ini_path: str | None = None
    depth: int = 4
    similarity_threshold: float = 0.5
    max_children: int = 100
    max_clusters: int | None = None
    full_search_strategy: str = "fallback"


def build_template_miner(settings: DrainSettings) -> TemplateMiner:
    return _build_template_miner(settings)


def parse_with_drain(log_lines: list[str], settings: DrainSettings) -> list[dict]:
    """Parse log lines with a fresh Drain miner.

    The legacy notebooks create a new `TemplateMiner` per window, which keeps
    cluster summaries local to the window. We preserve that behavior here.
    """
    miner = _build_template_miner(settings)
    parsed_lines: list[dict] = []

    for index, line in enumerate(log_lines, start=1):
        text = line.rstrip()
        result = miner.add_log_message(text)
        parsed_lines.append(
            {
                "line_number": index,
                "raw": text,
                "template": result.get("template_mined"),
                "cluster_id": result.get("cluster_id"),
                "cluster_size": result.get("cluster_size"),
                "change_type": result.get("change_type"),
            }
        )

    return parsed_lines


def warm_drain_with_history(log_lines: list[str], settings: DrainSettings) -> TemplateMiner:
    miner = _build_template_miner(settings)
    for line in log_lines:
        miner.add_log_message(line.rstrip())
    return miner


def parse_stream_with_drain(
    records: list[dict],
    settings: DrainSettings,
    history_records: list[dict] | None = None,
    update_model_online: bool = True,
) -> list[dict]:
    """Parse a stream of records while preserving record-level metadata.

    The miner is warmed on historical logs first, then reused across the stream.
    """
    history_records = history_records or []
    miner = warm_drain_with_history(
        [record.get("message") or record.get("log", "") for record in history_records],
        settings=settings,
    )

    parsed_records: list[dict] = []
    for index, record in enumerate(records, start=1):
        text = (record.get("message") or record.get("log", "")).rstrip()
        if update_model_online:
            result = miner.add_log_message(text)
        else:
            cluster = miner.match(text)
            if cluster is None:
                result = {
                    "change_type": "none",
                    "cluster_id": None,
                    "cluster_size": None,
                    "template_mined": None,
                    "cluster_count": len(list(miner.drain.clusters)),
                }
            else:
                result = {
                    "change_type": "none",
                    "cluster_id": cluster.cluster_id,
                    "cluster_size": cluster.size,
                    "template_mined": cluster.get_template(),
                    "cluster_count": len(list(miner.drain.clusters)),
                }

        enriched = dict(record)
        enriched["template"] = result.get("template_mined")
        enriched["drain"] = {
            "line_number": index,
            "raw": text,
            "template": result.get("template_mined"),
            "cluster_id": result.get("cluster_id"),
            "cluster_size": result.get("cluster_size"),
            "change_type": result.get("change_type"),
            "cluster_count": result.get("cluster_count"),
            "is_new_template": result.get("change_type") == "cluster_created",
        }
        parsed_records.append(enriched)

    return parsed_records


def parse_stream_with_dual_drain(
    records: list[dict],
    settings: DrainSettings,
    history_records: list[dict] | None = None,
    update_model_online: bool = True,
) -> list[dict]:
    """Parse stream with both stable reference and adaptive live Drain states."""
    history_records = history_records or []
    history_lines = [record.get("message") or record.get("log", "") for record in history_records]
    reference_miner = warm_drain_with_history(history_lines, settings=settings)
    live_miner = warm_drain_with_history(history_lines, settings=settings)

    parsed_records: list[dict] = []
    for index, record in enumerate(records, start=1):
        text = (record.get("message") or record.get("log", "")).rstrip()

        reference_cluster = reference_miner.match(text, full_search_strategy=settings.full_search_strategy)
        reference_template = reference_cluster.get_template() if reference_cluster else None

        if update_model_online:
            live_result = live_miner.add_log_message(text)
        else:
            live_cluster = live_miner.match(text, full_search_strategy=settings.full_search_strategy)
            if live_cluster is None:
                live_result = {
                    "change_type": "none",
                    "cluster_id": None,
                    "cluster_size": None,
                    "template_mined": None,
                    "cluster_count": len(list(live_miner.drain.clusters)),
                }
            else:
                live_result = {
                    "change_type": "none",
                    "cluster_id": live_cluster.cluster_id,
                    "cluster_size": live_cluster.size,
                    "template_mined": live_cluster.get_template(),
                    "cluster_count": len(list(live_miner.drain.clusters)),
                }

        enriched = dict(record)
        enriched["template"] = live_result.get("template_mined") or reference_template
        enriched["drain"] = {
            "line_number": index,
            "raw": text,
            "template": live_result.get("template_mined"),
            "cluster_id": live_result.get("cluster_id"),
            "cluster_size": live_result.get("cluster_size"),
            "change_type": live_result.get("change_type"),
            "cluster_count": live_result.get("cluster_count"),
            "is_new_template": live_result.get("change_type") == "cluster_created",
        }
        enriched["drain_reference"] = {
            "matched_template": reference_template,
            "matched_cluster_id": reference_cluster.cluster_id if reference_cluster else None,
            "matched_cluster_size": reference_cluster.size if reference_cluster else None,
            "is_known_template": reference_cluster is not None,
        }
        parsed_records.append(enriched)

    return parsed_records


def summarize_clusters(parsed_lines: list[dict]) -> list[dict]:
    """Build a cluster-style summary compatible with the legacy notebooks."""
    grouped: dict[int, dict] = {}
    for item in parsed_lines:
        cluster_id = item.get("cluster_id")
        if cluster_id is None:
            continue
        entry = grouped.setdefault(
            cluster_id,
            {
                "cluster_id": cluster_id,
                "size": 0,
                "template": item.get("template"),
                "examples": [],
            },
        )
        entry["size"] += 1
        if len(entry["examples"]) < 3 and item.get("raw"):
            entry["examples"].append(item["raw"])

    return sorted(grouped.values(), key=lambda item: item["size"], reverse=True)


def summarize_template_counts(parsed_lines: list[dict]) -> dict[str, int]:
    counts = Counter(item.get("template") or "<unknown>" for item in parsed_lines)
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def _build_template_miner(settings: DrainSettings) -> TemplateMiner:
    config = TemplateMinerConfig()
    ini_path = Path(settings.ini_path) if settings.ini_path else None
    if ini_path and ini_path.exists():
        config.load(str(ini_path))

    config.profiling_enabled = True
    config.depth = settings.depth
    config.sim_th = settings.similarity_threshold
    config.max_children = settings.max_children
    if settings.max_clusters is not None:
        config.drain_max_clusters = settings.max_clusters
    return TemplateMiner(config=config)
