from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from thesis_log_anomaly.datasets.schemas import LogRecord


def dataset_root() -> Path:
    return Path("data/raw/openstack2k")


def expected_files() -> list[str]:
    return [
        "OpenStack_2k.log",
        "OpenStack_2k.log_structured.csv",
        "OpenStack_2k.log_templates.csv",
    ]


def load_openstack2k_records(raw_dir: str | Path | None = None) -> list[LogRecord]:
    root = Path(raw_dir) if raw_dir is not None else dataset_root()
    csv_path = root / "OpenStack_2k.log_structured.csv"
    if csv_path.exists():
        return _load_openstack2k_from_csv(csv_path)
    return _load_openstack2k_from_log(root / "OpenStack_2k.log")


def to_rows(records: list[LogRecord]) -> list[dict]:
    return [asdict(record) for record in records]


def _load_openstack2k_from_csv(csv_path: Path) -> list[LogRecord]:
    records: list[LogRecord] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                LogRecord(
                    event_id=_parse_int(row.get("LineId")),
                    timestamp=_parse_datetime(row.get("Date"), row.get("Time")),
                    message=(row.get("Content") or "").strip(),
                    label=None,
                    template=(row.get("EventTemplate") or "").strip() or None,
                    source="openstack2k",
                    metadata={
                        "event_id_raw": row.get("EventId"),
                        "component": row.get("Component"),
                        "level": row.get("Level"),
                        "pid": _parse_int(row.get("Pid")),
                        "logrecord": row.get("Logrecord"),
                    },
                )
            )
    return records


def _load_openstack2k_from_log(log_path: Path) -> list[LogRecord]:
    records: list[LogRecord] = []
    with log_path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            records.append(
                LogRecord(
                    event_id=idx,
                    message=line,
                    label=None,
                    source="openstack2k",
                    metadata={"raw_line": raw_line.rstrip("\n")},
                )
            )
    return records


def _parse_datetime(date_value: str | None, time_value: str | None) -> datetime | None:
    if not date_value or not time_value:
        return None
    try:
        return datetime.strptime(f"{date_value} {time_value}", "%Y-%m-%d %H:%M:%S.%f")
    except ValueError:
        return None


def _parse_int(value: str | None) -> int | None:
    if value in (None, ""):
        return None
    return int(value)
