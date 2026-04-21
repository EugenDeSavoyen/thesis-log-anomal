from pathlib import Path
import re


def load_config(config_path: str | Path) -> dict:
    """Lightweight YAML-like config loader.

    This keeps the project dependency-light during early thesis refactoring.
    It supports the nested mappings used in the current `configs/base.yaml`.
    """
    path = Path(config_path)
    config: dict = {}
    stack: list[tuple[int, dict]] = [(-1, config)]

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if not line or line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()

        current = stack[-1][1]
        key, _, value = line.strip().partition(":")
        key = key.strip()
        value = value.strip()

        if not value:
            child: dict = {}
            current[key] = child
            stack.append((indent, child))
            continue

        current[key] = _parse_scalar(value)

    return config


def _parse_scalar(value: str):
    lowered = value.lower()
    if lowered == "null":
        return None
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"-?\d+", value):
        return int(value)
    if re.fullmatch(r"-?\d+\.\d+", value):
        return float(value)
    return value
