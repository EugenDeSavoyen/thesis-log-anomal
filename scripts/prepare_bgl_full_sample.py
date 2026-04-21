from __future__ import annotations

import argparse
import csv
import math
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from urllib.request import urlretrieve


ZENODO_BGL_URL = "https://zenodo.org/records/8196385/files/BGL.zip?download=1"


def main() -> None:
    parser = argparse.ArgumentParser(description="Download full LogHub BGL and create a chronological labeled sample.")
    parser.add_argument("--sample-lines", type=int, default=100_000)
    parser.add_argument("--blocks", type=int, default=1)
    parser.add_argument(
        "--block-selection",
        choices=["even", "anomaly_spread", "mixed_context", "interleaved_context"],
        default="even",
        help="Use evenly spaced blocks or choose chronological blocks that contain alerts.",
    )
    parser.add_argument("--archive", default="data/raw/downloads/BGL.zip")
    parser.add_argument("--output-dir", default="data/raw/bgl_full_sample")
    parser.add_argument("--source-log", default=None, help="Optional existing full BGL.log path.")
    args = parser.parse_args()

    archive = Path(args.archive)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.source_log:
        source_log = Path(args.source_log)
    else:
        archive.parent.mkdir(parents=True, exist_ok=True)
        if not archive.exists():
            print(f"Downloading {ZENODO_BGL_URL} -> {archive}")
            urlretrieve(ZENODO_BGL_URL, archive)
        source_log = _extract_bgl_log(archive, output_dir)

    output_csv = output_dir / "BGL_labeled.csv"
    if args.blocks <= 1:
        rows_written, anomalous_rows, block_metadata = _write_sample_csv(source_log, output_csv, args.sample_lines)
    else:
        rows_written, anomalous_rows, block_metadata = _write_multi_block_sample_csv(
            source_log,
            output_csv,
            sample_lines=args.sample_lines,
            blocks=args.blocks,
            block_selection=args.block_selection,
        )
    metadata_path = output_dir / "sample_metadata.json"
    _write_metadata(metadata_path, args, source_log, output_csv, rows_written, anomalous_rows, block_metadata)
    print(f"source_log={source_log}")
    print(f"output_csv={output_csv}")
    print(f"metadata={metadata_path}")
    print(f"rows_written={rows_written}")
    print(f"anomalous_rows={anomalous_rows}")
    print(f"anomaly_rate={(anomalous_rows / rows_written) if rows_written else 0.0:.6f}")


def _extract_bgl_log(archive: Path, output_dir: Path) -> Path:
    candidate = output_dir / "BGL.log"
    if candidate.exists():
        return candidate

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        log_names = [name for name in names if name.endswith("BGL.log")]
        if not log_names:
            log_names = [name for name in names if name.lower().endswith(".log")]
        if not log_names:
            raise FileNotFoundError(f"No .log file found inside {archive}")

        selected = log_names[0]
        print(f"Extracting {selected} -> {candidate}")
        with zf.open(selected) as source, candidate.open("wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)

    return candidate


def _write_sample_csv(source_log: Path, output_csv: Path, sample_lines: int) -> tuple[int, int, list[dict]]:
    rows_written = 0
    anomalous_rows = 0
    with source_log.open("r", encoding="utf-8", errors="ignore") as source, output_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target:
        writer = csv.DictWriter(target, fieldnames=["id", "datetime", "log", "alert", "alert_type", "template"])
        writer.writeheader()
        for line_number, raw_line in enumerate(source, start=1):
            if rows_written >= sample_lines:
                break
            parsed = _parse_bgl_line(raw_line)
            if parsed is None:
                continue
            rows_written += 1
            anomalous_rows += parsed["alert"]
            writer.writerow(_row(line_number, parsed, block_id=0, source_line=line_number))
    return rows_written, anomalous_rows, [
        {
            "block_id": 0,
            "source_start_line": 1,
            "source_end_line": rows_written,
            "rows": rows_written,
            "anomalous_rows": anomalous_rows,
        }
    ]


def _write_multi_block_sample_csv(
    source_log: Path,
    output_csv: Path,
    sample_lines: int,
    blocks: int,
    block_selection: str,
) -> tuple[int, int, list[dict]]:
    total_lines = _count_lines(source_log)
    block_size = max(1, sample_lines // blocks)
    if block_selection == "interleaved_context":
        starts = _interleaved_context_block_starts(source_log, total_lines, block_size, blocks)
    elif block_selection == "mixed_context":
        starts = _mixed_context_block_starts(source_log, total_lines, block_size, blocks)
    elif block_selection == "anomaly_spread":
        starts = _anomaly_spread_block_starts(source_log, total_lines, block_size, blocks)
    else:
        starts = _even_block_starts(total_lines, block_size, blocks)
    selected_ranges = [
        {
            "block_id": index,
            "start": start,
            "end": min(start + block_size - 1, total_lines),
        }
        for index, start in enumerate(starts)
    ]
    read_ranges = sorted(selected_ranges, key=lambda item: item["start"])

    rows_written = 0
    anomalous_rows = 0
    block_metadata = [
        {
            "block_id": index,
            "source_start_line": start,
            "source_end_line": end,
            "rows": 0,
            "anomalous_rows": 0,
        }
        for item in selected_ranges
        for index, start, end in [(item["block_id"], item["start"], item["end"])]
    ]

    current_read_index = 0
    rows_by_block: dict[int, list[dict]] = {item["block_id"]: [] for item in selected_ranges}
    with source_log.open("r", encoding="utf-8", errors="ignore") as source, output_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as target:
        writer = csv.DictWriter(
            target,
            fieldnames=["id", "datetime", "log", "alert", "alert_type", "template", "block_id", "source_line"],
        )
        writer.writeheader()
        for line_number, raw_line in enumerate(source, start=1):
            while current_read_index < len(read_ranges) and line_number > read_ranges[current_read_index]["end"]:
                current_read_index += 1
            if current_read_index >= len(read_ranges):
                break
            current_range = read_ranges[current_read_index]
            start = current_range["start"]
            end = current_range["end"]
            if line_number < start or line_number > end:
                continue
            parsed = _parse_bgl_line(raw_line)
            if parsed is None:
                continue
            rows_written += 1
            anomalous_rows += parsed["alert"]
            block_id = current_range["block_id"]
            block_metadata[block_id]["rows"] += 1
            block_metadata[block_id]["anomalous_rows"] += parsed["alert"]
            rows_by_block[block_id].append(_row(rows_written, parsed, block_id=block_id, source_line=line_number))

        ordered_row_id = 1
        for block_id in sorted(rows_by_block):
            for row in rows_by_block[block_id]:
                row["id"] = ordered_row_id
                ordered_row_id += 1
                writer.writerow(row)

    return rows_written, anomalous_rows, block_metadata


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def _even_block_starts(total_lines: int, block_size: int, blocks: int) -> list[int]:
    if blocks <= 1:
        return [1]
    max_start = max(1, total_lines - block_size + 1)
    return [
        1 + round(index * (max_start - 1) / (blocks - 1))
        for index in range(blocks)
    ]


def _anomaly_spread_block_starts(source_log: Path, total_lines: int, block_size: int, blocks: int) -> list[int]:
    anomaly_lines: list[int] = []
    block_counts = _scan_anomaly_blocks(source_log, total_lines, block_size)
    for block in block_counts:
        if block["anomalous_rows"] > 0:
            anomaly_lines.append(block["start"] + block_size // 2)

    if not anomaly_lines:
        return _even_block_starts(total_lines, block_size, blocks)

    anchors = _spread_values(anomaly_lines, blocks)
    max_start = max(1, total_lines - block_size + 1)
    starts = sorted(
        max(1, min(max_start, anchor - block_size // 2))
        for anchor in anchors
    )
    return _dedupe_overlapping_starts(starts, block_size, max_start)


def _mixed_context_block_starts(source_log: Path, total_lines: int, block_size: int, blocks: int) -> list[int]:
    scanned = _scan_anomaly_blocks(source_log, total_lines, block_size)
    anomaly_blocks = [block for block in scanned if block["anomalous_rows"] > 0]
    normal_blocks = [block for block in scanned if block["anomalous_rows"] == 0]
    if not anomaly_blocks or not normal_blocks:
        return _even_block_starts(total_lines, block_size, blocks)

    anomaly_count = max(1, blocks // 2)
    normal_count = max(0, blocks - anomaly_count)
    selected = _spread_blocks(normal_blocks, normal_count) + _spread_blocks(anomaly_blocks, anomaly_count)
    return sorted(block["start"] for block in selected)


def _interleaved_context_block_starts(source_log: Path, total_lines: int, block_size: int, blocks: int) -> list[int]:
    scanned = _scan_anomaly_blocks(source_log, total_lines, block_size)
    anomaly_blocks = [block for block in scanned if block["anomalous_rows"] > 0]
    normal_blocks = [block for block in scanned if block["anomalous_rows"] == 0]
    if not anomaly_blocks or not normal_blocks:
        return _even_block_starts(total_lines, block_size, blocks)

    pair_count = blocks // 2
    selected_normals = _spread_blocks(normal_blocks, pair_count)
    selected_anomalies = _spread_blocks(anomaly_blocks, blocks - pair_count)
    selected: list[dict] = []
    for index in range(max(len(selected_normals), len(selected_anomalies))):
        if index < len(selected_normals):
            selected.append(selected_normals[index])
        if index < len(selected_anomalies):
            selected.append(selected_anomalies[index])
    return [block["start"] for block in selected[:blocks]]


def _scan_anomaly_blocks(source_log: Path, total_lines: int, block_size: int) -> list[dict]:
    num_blocks = math.ceil(total_lines / block_size)
    blocks = [
        {
            "block_index": index,
            "start": index * block_size + 1,
            "end": min((index + 1) * block_size, total_lines),
            "rows": 0,
            "anomalous_rows": 0,
        }
        for index in range(num_blocks)
    ]

    with source_log.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            block_index = min((line_number - 1) // block_size, num_blocks - 1)
            parsed = _parse_bgl_line(raw_line)
            if parsed is None:
                continue
            blocks[block_index]["rows"] += 1
            blocks[block_index]["anomalous_rows"] += parsed["alert"]
    return blocks


def _spread_blocks(blocks: list[dict], count: int) -> list[dict]:
    if count <= 0:
        return []
    if len(blocks) <= count:
        return blocks
    indices = _spread_values(list(range(len(blocks))), count)
    return [blocks[index] for index in indices]


def _spread_values(values: list[int], count: int) -> list[int]:
    if count <= 1:
        return [values[len(values) // 2]]
    last = len(values) - 1
    return [values[round(index * last / (count - 1))] for index in range(count)]


def _dedupe_overlapping_starts(starts: list[int], block_size: int, max_start: int) -> list[int]:
    adjusted: list[int] = []
    for start in starts:
        if adjusted and start < adjusted[-1] + block_size:
            start = adjusted[-1] + block_size
        adjusted.append(min(start, max_start))

    # If right-shifting created duplicates near the file end, spread the tail backwards.
    for index in range(len(adjusted) - 2, -1, -1):
        if adjusted[index] + block_size > adjusted[index + 1]:
            adjusted[index] = max(1, adjusted[index + 1] - block_size)

    return adjusted


def _row(row_id: int, parsed: dict, block_id: int, source_line: int) -> dict:
    return {
        "id": row_id,
        "datetime": parsed["datetime"],
        "log": parsed["message"],
        "alert": parsed["alert"],
        "alert_type": parsed["alert_type"],
        "template": "",
        "block_id": block_id,
        "source_line": source_line,
    }


def _write_metadata(
    metadata_path: Path,
    args: argparse.Namespace,
    source_log: Path,
    output_csv: Path,
    rows_written: int,
    anomalous_rows: int,
    blocks: list[dict],
) -> None:
    import json

    metadata = {
        "source_log": str(source_log),
        "output_csv": str(output_csv),
        "sample_lines_requested": args.sample_lines,
        "blocks_requested": args.blocks,
        "rows_written": rows_written,
        "anomalous_rows": anomalous_rows,
        "anomaly_rate": (anomalous_rows / rows_written) if rows_written else 0.0,
        "blocks": blocks,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _parse_bgl_line(raw_line: str) -> dict | None:
    line = raw_line.strip()
    if not line:
        return None

    tokens = line.split(maxsplit=3)
    if not tokens:
        return None

    if tokens[0] == "-":
        alert = 0
        alert_type = "-"
        rest = tokens[1:]
    else:
        alert = 1
        alert_type = tokens[0]
        rest = tokens[1:]

    if len(rest) < 2:
        return None

    unix_seconds = rest[0]
    message = rest[2] if len(rest) >= 3 else rest[-1]
    try:
        timestamp = datetime.fromtimestamp(int(unix_seconds)).isoformat()
    except ValueError:
        timestamp = ""

    return {
        "datetime": timestamp,
        "message": message,
        "alert": alert,
        "alert_type": alert_type,
    }


if __name__ == "__main__":
    sys.exit(main())
