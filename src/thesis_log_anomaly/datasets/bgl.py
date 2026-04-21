from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from thesis_log_anomaly.datasets.schemas import LogRecord


def dataset_root() -> Path:
    return Path("data/raw/bgl")


def expected_files() -> list[str]:
    return [
        "BGL.log",
        "BGL_ul.txt",
        "BGL_labeled.csv",
        "BGL_labeled_with_templates.csv",
        "anomaly_BGL.txt",
    ]


def load_bgl_records(raw_dir: str | Path | None = None) -> list[LogRecord]:
    root = Path(raw_dir) if raw_dir is not None else dataset_root()
    csv_path = root / "BGL_labeled.csv"
    if csv_path.exists():
        return _load_bgl_from_csv(csv_path)
    return _load_bgl_from_log(root / "BGL.log")


def to_rows(records: list[LogRecord]) -> list[dict]:
    rows = []
    for record in records:
        row = asdict(record)
        if record.metadata.get("block_id") not in (None, ""):
            row["block_id"] = record.metadata.get("block_id")
        if record.metadata.get("source_line") not in (None, ""):
            row["source_line"] = record.metadata.get("source_line")
        rows.append(row)
    return rows


def _load_bgl_from_csv(csv_path: Path) -> list[LogRecord]:
    records: list[LogRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                LogRecord(
                    event_id=int(row["id"]) if row.get("id") else None,
                    timestamp=_parse_datetime(row.get("datetime")),
                    message=(row.get("log") or "").strip(),
                    label=_parse_int(row.get("alert")),
                    template=(row.get("template") or "").strip() or None,
                    source="bgl",
                    metadata={
                        "alert_type": row.get("alert_type"),
                        "block_id": row.get("block_id"),
                        "source_line": _parse_int(row.get("source_line")),
                    },
                )
            )
    return records


def _load_bgl_from_log(log_path: Path) -> list[LogRecord]:
    records: list[LogRecord] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            label = 0 if line.startswith("-") else 1
            normalized = line[1:].strip() if line.startswith("-") else line
            parts = normalized.split(maxsplit=2)

            unix_seconds = None
            human_time = None
            message = normalized
            if len(parts) >= 3:
                unix_seconds = parts[0]
                human_time = parts[1]
                message = parts[2]

            timestamp = None
            if unix_seconds and unix_seconds.isdigit():
                timestamp = datetime.fromtimestamp(int(unix_seconds))

            records.append(
                LogRecord(
                    event_id=idx,
                    timestamp=timestamp,
                    message=message.strip(),
                    label=label,
                    source="bgl",
                    metadata={
                        "raw_line": raw_line.rstrip("\n"),
                        "unix_seconds": unix_seconds,
                        "human_time": human_time,
                    },
                )
            )
    return records


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


def _parse_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
