from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class LogRecord:
    message: str
    event_id: int | None = None
    timestamp: datetime | None = None
    label: str | int | None = None
    template: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LogWindow:
    window_id: int
    start_index: int
    end_index: int
    records: list[LogRecord]
    start_time: datetime | None = None
    end_time: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
