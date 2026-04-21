from __future__ import annotations

from dataclasses import asdict
from datetime import timedelta
from typing import Any

from thesis_log_anomaly.datasets.schemas import LogRecord, LogWindow


def build_sliding_windows(items: list, size: int, stride: int) -> list[list]:
    """Create fixed-size overlapping windows."""
    if size <= 0 or stride <= 0:
        raise ValueError("size and stride must be positive")

    if len(items) < size:
        return [items[:]] if items else []

    windows = [items[i : i + size] for i in range(0, len(items) - size + 1, stride)]
    if windows:
        last_end = windows[-1]
        if last_end and items[-1] is not last_end[-1]:
            windows.append(items[-size:])
    return windows


def build_count_windows(records: list[dict], size: int, stride: int) -> list[dict]:
    typed_records = [_to_record(record) for record in records]
    windows = build_sliding_windows(typed_records, size=size, stride=stride)
    return [_window_to_dict(index + 1, window, start_index=index * stride) for index, window in enumerate(windows)]


def build_time_windows(records: list[dict], duration_hours: int, stride_hours: int) -> list[dict]:
    typed_records = [_to_record(record) for record in records]
    typed_records = [record for record in typed_records if record.timestamp is not None]
    typed_records.sort(key=lambda item: item.timestamp)
    if not typed_records:
        return []

    duration = timedelta(hours=duration_hours)
    stride = timedelta(hours=stride_hours)
    current_start = typed_records[0].timestamp
    max_time = typed_records[-1].timestamp

    windows: list[dict] = []
    window_id = 1
    while current_start <= max_time:
        current_end = current_start + duration
        window_records = [
            record
            for record in typed_records
            if record.timestamp is not None and current_start <= record.timestamp < current_end
        ]
        if window_records:
            windows.append(
                _serialize_window(
                    LogWindow(
                        window_id=window_id,
                        start_index=0,
                        end_index=len(window_records) - 1,
                        start_time=current_start,
                        end_time=current_end,
                        records=window_records,
                        metadata={"mode": "time"},
                    )
                )
            )
            window_id += 1
        current_start += stride

    return windows


def build_group_windows(records: list[dict], grouping_key: str) -> list[dict]:
    typed_records = [_to_record(record) for record in records]
    ordered_groups: dict[str, list[LogRecord]] = {}

    for record in typed_records:
        group_value = _resolve_group_value(record, grouping_key)
        if group_value is None:
            continue
        group_id = str(group_value)
        ordered_groups.setdefault(group_id, []).append(record)

    windows: list[dict] = []
    start_index = 0
    for window_id, (group_id, group_records) in enumerate(ordered_groups.items(), start=1):
        windows.append(
            _serialize_window(
                LogWindow(
                    window_id=window_id,
                    start_index=start_index,
                    end_index=start_index + len(group_records) - 1,
                    start_time=group_records[0].timestamp if group_records else None,
                    end_time=group_records[-1].timestamp if group_records else None,
                    records=group_records,
                    metadata={
                        "mode": "group",
                        "grouping_key": grouping_key,
                        "group_value": group_id,
                    },
                )
            )
        )
        start_index += len(group_records)

    return windows


def _window_to_dict(window_id: int, records: list[LogRecord], start_index: int) -> dict:
    block_ids = [
        record.metadata.get("block_id")
        for record in records
        if record.metadata.get("block_id") not in (None, "")
    ]
    return _serialize_window(
        LogWindow(
            window_id=window_id,
            start_index=start_index,
            end_index=start_index + len(records) - 1,
            start_time=records[0].timestamp if records else None,
            end_time=records[-1].timestamp if records else None,
            records=records,
            metadata={
                "mode": "count",
                "block_id": _majority_value(block_ids),
                "block_ids": sorted({str(block_id) for block_id in block_ids}),
            },
        )
    )


def _serialize_window(window: LogWindow) -> dict:
    data = asdict(window)
    data["records"] = [_serialize_record(record) for record in window.records]
    data["start_time"] = window.start_time.isoformat() if window.start_time else None
    data["end_time"] = window.end_time.isoformat() if window.end_time else None
    return data


def _serialize_record(record: LogRecord) -> dict:
    data = asdict(record)
    data["timestamp"] = record.timestamp.isoformat() if record.timestamp else None
    drain = data.get("metadata", {}).get("drain")
    if drain is not None:
        data["drain"] = drain
    drain_reference = data.get("metadata", {}).get("drain_reference")
    if drain_reference is not None:
        data["drain_reference"] = drain_reference
    novelty = data.get("metadata", {}).get("novelty")
    if novelty is not None:
        data["novelty"] = novelty
    return data


def _to_record(record: dict | LogRecord) -> LogRecord:
    if isinstance(record, LogRecord):
        return record
    timestamp = record.get("timestamp")
    if isinstance(timestamp, str) and timestamp:
        from datetime import datetime

        timestamp = datetime.fromisoformat(timestamp)
    return LogRecord(
        event_id=record.get("event_id"),
        timestamp=timestamp,
        message=record.get("message") or record.get("log", ""),
        label=record.get("label"),
        template=record.get("template"),
        source=record.get("source"),
        metadata={
            **record.get("metadata", {}),
            "drain": record.get("drain"),
            "drain_reference": record.get("drain_reference"),
            "novelty": record.get("novelty"),
            "block_id": record.get("block_id"),
            "source_line": record.get("source_line"),
        },
    )


def _majority_value(values: list) -> str | None:
    if not values:
        return None
    counts: dict[str, int] = {}
    for value in values:
        normalized = str(value)
        counts[normalized] = counts.get(normalized, 0) + 1
    return max(counts.items(), key=lambda item: item[1])[0]


def _resolve_group_value(record: LogRecord, grouping_key: str) -> Any:
    current: Any = {
        "event_id": record.event_id,
        "timestamp": record.timestamp,
        "message": record.message,
        "label": record.label,
        "template": record.template,
        "source": record.source,
        "metadata": record.metadata,
    }
    for part in grouping_key.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
        if current is None:
            return None
    return current
