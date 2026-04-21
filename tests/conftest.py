from __future__ import annotations

import inspect
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RAW_DATA = ROOT / "data" / "raw"


def _has_local_raw_data() -> bool:
    return RAW_DATA.exists() and any(
        path.is_file() and path.name != ".gitkeep" for path in RAW_DATA.rglob("*")
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _has_local_raw_data():
        return

    skip_no_data = pytest.mark.skip(reason="requires local datasets under data/raw/")
    for item in items:
        function = getattr(item, "obj", None)
        if function is None:
            continue
        try:
            source = inspect.getsource(function)
        except OSError:
            continue
        if "load_logs(" in source or "run_pipeline(" in source:
            item.add_marker(skip_no_data)
