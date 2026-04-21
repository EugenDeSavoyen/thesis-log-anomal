from __future__ import annotations

import csv
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from thesis_log_anomaly.datasets.schemas import LogRecord

BLOCK_ID_PATTERN = re.compile(r"(blk_-?\d+)")


def dataset_root() -> Path:
    return Path("data/raw/hdfs100k")


def expected_files() -> list[str]:
    return [
        "HDFS_100k.log_structured.csv",
        "anomaly_label.csv",
    ]


def load_hdfs100k_records(raw_dir: str | Path | None = None) -> list[LogRecord]:
    root = Path(raw_dir) if raw_dir is not None else dataset_root()
    csv_path = root / "HDFS_100k.log_structured.csv"
    label_path = root / "anomaly_label.csv"
    block_labels = _load_block_labels(label_path)
    return _load_hdfs100k_from_csv(csv_path, block_labels)


def to_rows(records: list[LogRecord]) -> list[dict]:
    return [asdict(record) for record in records]


def _load_hdfs100k_from_csv(csv_path: Path, block_labels: dict[str, int]) -> list[LogRecord]:
    records: list[LogRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            content = (row.get("Content") or "").strip()
            block_id = _extract_block_id(content)
            records.append(
                LogRecord(
                    event_id=_parse_int(row.get("LineId")),
                    timestamp=_parse_datetime(row.get("Date"), row.get("Time")),
                    message=content,
                    label=block_labels.get(block_id),
                    template=(row.get("EventTemplate") or "").strip() or None,
                    source="hdfs100k",
                    metadata={
                        "block_id": block_id,
                        "event_id_raw": row.get("EventId"),
                        "component": row.get("Component"),
                        "level": row.get("Level"),
                        "pid": _parse_int(row.get("Pid")),
                    },
                )
            )
    return records


def _load_block_labels(label_path: Path) -> dict[str, int]:
    labels: dict[str, int] = {}
    with label_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            block_id = row.get("BlockId")
            label = row.get("Label")
            if not block_id or not label:
                continue
            labels[block_id] = 1 if label.lower() == "anomaly" else 0
    return labels


def _extract_block_id(content: str) -> str | None:
    match = BLOCK_ID_PATTERN.search(content)
    if not match:
        return None
    return match.group(1)


def _parse_datetime(date_value: str | None, time_value: str | None) -> datetime | None:
    if not date_value or not time_value:
        return None
    try:
        return datetime.strptime(f"{date_value} {time_value}", "%y%m%d %H%M%S")
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
