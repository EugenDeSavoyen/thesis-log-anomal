from __future__ import annotations

import json
from pathlib import Path


def evaluate_candidates(payload, config: dict):
    """Evaluate the pre-LLM pipeline after GEV and POT and optionally write a report."""
    evaluation_config = config.get("evaluation", {})
    pot_enabled = bool(config.get("pot", {}).get("enabled", False))
    windows_after_gev = payload["windows_after_gev"]
    windows_after_pot = payload["windows_after_pot"]
    parsed_stream = payload["parsed_stream"]

    report = {
        "dataset": config.get("data", {}).get("dataset"),
        "history_size": parsed_stream.get("history_size", 0),
        "stream_size": parsed_stream.get("stream_size", 0),
        "post_gev": _evaluate_post_gev(windows_after_gev),
        "post_pot": _evaluate_post_pot(windows_after_pot, parsed_stream["stream_records"]) if pot_enabled else None,
    }

    if evaluation_config.get("enabled", True):
        output_path = Path(evaluation_config.get("output_path", "outputs/reports/pre_llm_evaluation.json"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def _evaluate_post_gev(windows: list[dict]) -> dict:
    scored_windows = [window for window in windows if window.get("evt")]
    tp = fp = tn = fn = 0
    all_event_ids: set[int] = set()
    anomalous_event_ids: set[int] = set()
    forwarded_event_ids: set[int] = set()
    forwarded_anomalous_event_ids: set[int] = set()

    for window in scored_windows:
        truth = window.get("ground_truth", {}).get("window_label")
        pred = bool(window.get("evt", {}).get("is_suspicious"))
        if truth == 1 and pred:
            tp += 1
        elif truth == 0 and pred:
            fp += 1
        elif truth == 0 and not pred:
            tn += 1
        elif truth == 1 and not pred:
            fn += 1

        records = window.get("records", [])
        for record in records:
            event_id = record.get("event_id")
            if event_id is None:
                continue
            all_event_ids.add(event_id)
            if record.get("label") == 1:
                anomalous_event_ids.add(event_id)
        if pred:
            for record in records:
                event_id = record.get("event_id")
                if event_id is None:
                    continue
                forwarded_event_ids.add(event_id)
                if record.get("label") == 1:
                    forwarded_anomalous_event_ids.add(event_id)

    precision, recall, f1 = _prf(tp, fp, fn)
    return {
        "num_windows": len(scored_windows),
        "num_suspicious_windows": sum(1 for window in scored_windows if window.get("evt", {}).get("is_suspicious")),
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "window_reduction_ratio": _safe_ratio(
            sum(1 for window in scored_windows if window.get("evt", {}).get("is_suspicious")),
            len(scored_windows),
        ),
        "event_load_to_next_stage": len(forwarded_event_ids),
        "event_reduction_ratio": _safe_ratio(len(forwarded_event_ids), len(all_event_ids)),
        "anomaly_event_recall_to_next_stage": _safe_ratio(len(forwarded_anomalous_event_ids), len(anomalous_event_ids)),
    }


def _evaluate_post_pot(windows: list[dict], stream_records: list[dict]) -> dict:
    labeled_stream_records = [record for record in stream_records if record.get("label") is not None]
    total_events = len(labeled_stream_records)
    total_anomalous_events = sum(1 for record in labeled_stream_records if record.get("label") == 1)

    candidate_event_ids: set[int] = set()
    for window in windows:
        if window.get("pot", {}).get("candidate_event_ids"):
            candidate_event_ids.update(window["pot"]["candidate_event_ids"])
            continue
        for cluster in window.get("clusters", []):
            if cluster.get("pot", {}).get("is_anomaly"):
                candidate_event_ids.update(cluster.get("event_ids", []))

    if not candidate_event_ids:
        return {
            "candidate_events": 0,
            "candidate_anomalous_events": 0,
            "precision": None,
            "recall": 0.0 if total_anomalous_events else None,
            "f1": None,
            "event_reduction_ratio": 0.0 if total_events else None,
        }

    lookup = {record.get("event_id"): record for record in stream_records if record.get("event_id") is not None}
    candidates = [lookup[event_id] for event_id in candidate_event_ids if event_id in lookup]
    tp = sum(1 for record in candidates if record.get("label") == 1)
    fp = sum(1 for record in candidates if record.get("label") == 0)
    fn = total_anomalous_events - tp
    precision, recall, f1 = _prf(tp, fp, fn)

    return {
        "candidate_events": len(candidates),
        "candidate_anomalous_events": tp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "event_reduction_ratio": _safe_ratio(len(candidates), total_events),
        "anomaly_event_recall_to_next_stage": _safe_ratio(tp, total_anomalous_events),
        "windows_with_pot_anomaly": sum(1 for window in windows if window.get("pot", {}).get("has_pot_anomaly")),
    }


def _prf(tp: int, fp: int, fn: int) -> tuple[float | None, float | None, float | None]:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator
