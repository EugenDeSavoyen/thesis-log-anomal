from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from thesis_log_anomaly.datasets.schemas import LogRecord


def dataset_root() -> Path:
    return Path("data/raw/linux2k")


def expected_files() -> list[str]:
    return [
        "linux2k.log",
        "lnx.txt",
        "processed_lnx.log.txt",
        "processed_lnx.txt",
        "example.txt",
    ]


def load_linux2k_records(raw_dir: str | Path | None = None) -> list[LogRecord]:
    root = Path(raw_dir) if raw_dir is not None else dataset_root()
    source_path = root / "lnx.txt"
    if not source_path.exists():
        source_path = root / "linux2k.log"

    records: list[LogRecord] = []
    with source_path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("[page:"):
                continue

            parts = line.split()
            if len(parts) < 4:
                continue

            timestamp = None
            try:
                timestamp = datetime.strptime(" ".join(parts[:3]), "%b %d %H:%M:%S")
            except ValueError:
                pass

            records.append(
                LogRecord(
                    event_id=idx,
                    timestamp=timestamp,
                    message=" ".join(parts[3:]).strip(),
                    source="linux2k",
                    metadata={"raw_line": raw_line.rstrip("\n")},
                )
            )
    return records


def to_rows(records: list[LogRecord]) -> list[dict]:
    return [asdict(record) for record in records]
