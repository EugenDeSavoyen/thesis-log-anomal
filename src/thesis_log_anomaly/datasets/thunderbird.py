from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from thesis_log_anomaly.datasets.schemas import LogRecord


def dataset_root() -> Path:
    return Path("data/raw/thunderbird")


def expected_files() -> list[str]:
    return [
        "Thunderbird_2k.log",
        "Thunderbird_2k.log_structured.csv",
        "Thunderbird_2k.log_templates.csv",
    ]


def load_thunderbird_records(raw_dir: str | Path | None = None) -> list[LogRecord]:
    root = Path(raw_dir) if raw_dir is not None else dataset_root()
    csv_path = root / "Thunderbird_2k.log_structured.csv"
    if csv_path.exists():
        return _load_thunderbird_from_csv(csv_path)
    return _load_thunderbird_from_log(root / "Thunderbird_2k.log")


def to_rows(records: list[LogRecord]) -> list[dict]:
    return [asdict(record) for record in records]


def _load_thunderbird_from_csv(csv_path: Path) -> list[LogRecord]:
    records: list[LogRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                LogRecord(
                    event_id=_parse_int(row.get("LineId")),
                    timestamp=_parse_timestamp(row.get("Timestamp")),
                    message=(row.get("Content") or "").strip(),
                    label=_parse_label(row.get("Label")),
                    template=(row.get("EventTemplate") or "").strip() or None,
                    source="thunderbird",
                    metadata={
                        "event_id_raw": row.get("EventId"),
                        "component": row.get("Component"),
                        "location": row.get("Location"),
                        "user": row.get("User"),
                        "pid": _parse_int(row.get("PID")),
                    },
                )
            )
    return records


def _load_thunderbird_from_log(log_path: Path) -> list[LogRecord]:
    records: list[LogRecord] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=8)
            label_token = parts[0] if parts else "-"
            unix_seconds = parts[1] if len(parts) > 1 else None
            message = parts[8] if len(parts) > 8 else line

            records.append(
                LogRecord(
                    event_id=idx,
                    timestamp=_parse_timestamp(unix_seconds),
                    message=message.strip(),
                    label=_parse_label(label_token),
                    source="thunderbird",
                    metadata={
                        "raw_line": raw_line.rstrip("\n"),
                        "unix_seconds": unix_seconds,
                    },
                )
            )
    return records


def _parse_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    if value.isdigit():
        return datetime.fromtimestamp(int(value))
    return None


def _parse_label(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return 0 if value == "-" else 1


def _parse_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
