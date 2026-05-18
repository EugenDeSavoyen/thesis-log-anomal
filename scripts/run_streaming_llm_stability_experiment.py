from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_streaming_llm_incidents import _counts, _read_decisions  # noqa: E402
from scripts.report_llm_incident_detector_comparison import (  # noqa: E402
    INSPECT_ACTIONS,
    count_anomalies,
    read_llm_decisions,
    safe_ratio,
)
from scripts.run_llm_triage import (  # noqa: E402
    LlmSettings,
    _sha256_text,
    build_run_metadata,
    read_jsonl,
    run_triage_packets,
    write_jsonl,
)
from scripts.simulate_streaming_incident_triage import (  # noqa: E402
    _deduplicate_events,
    _enrich_markov_scores,
    _read_true_clusters,
    evaluate_streaming_regions,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repeat the streaming incident LLM detector to measure decision stability."
    )
    parser.add_argument("--input-jsonl", default="outputs/reports/streaming_incident_packets_gap100.jsonl")
    parser.add_argument("--regions-csv", default="outputs/reports/streaming_incident_regions_gap100.csv")
    parser.add_argument("--streaming-report-json", default="outputs/reports/streaming_incident_triage_report_gap100.json")
    parser.add_argument("--input-csv", default="data/processed/event_level_candidates_bgl_multiblock.csv")
    parser.add_argument("--true-clusters-csv", default="outputs/reports/anomaly_cluster_density_clusters.csv")
    parser.add_argument("--prompt-template", default="prompts/log_triage_streaming_incident_v3.md")
    parser.add_argument("--prompt-template-id", default="log_triage_streaming_incident_v3_gap100_stability")
    parser.add_argument("--output-dir", default="outputs/reports/streaming_llm_stability_gap100_v3_n10")
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260501)
    parser.add_argument("--provider", default="ollama", choices=["ollama"])
    parser.add_argument("--endpoint", default="http://127.0.0.1:11434/api/generate")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--model-quantization", default="Q4_K_M")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--seed-policy",
        choices=["fixed", "increment"],
        default="fixed",
        help="Use one seed for all repeats, or increment it by run index to stress-test sampling stability.",
    )
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=240)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--max-packets", type=int, default=None)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = Path(args.prompt_template)
    prompt_template = prompt_path.read_text(encoding="utf-8")
    prompt_sha256 = _sha256_text(prompt_template)
    packets = read_jsonl(Path(args.input_jsonl))[args.start_offset :]
    if args.max_packets is not None:
        packets = packets[: args.max_packets]

    base_settings = LlmSettings(
        provider=args.provider,
        endpoint=args.endpoint,
        model=args.model,
        model_quantization=args.model_quantization,
        prompt_template_id=args.prompt_template_id,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        repeat_penalty=args.repeat_penalty,
        timeout_seconds=args.timeout_seconds,
    )

    df = _enrich_markov_scores(_deduplicate_events(pd.read_csv(args.input_csv)))
    regions = pd.read_csv(args.regions_csv)
    base_report = _read_json(Path(args.streaming_report_json))
    true_clusters = _read_true_clusters(Path(args.true_clusters_csv))
    actual_by_region = _actual_positive_by_region(regions, df)

    run_rows = []
    per_packet_rows = []
    for run_index in range(1, args.n_runs + 1):
        run_dir = output_dir / f"run_{run_index:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        output_jsonl = run_dir / "streaming_incident_llm.jsonl"
        output_metadata = run_dir / "streaming_incident_llm_run.json"
        output_eval_json = run_dir / "streaming_incident_llm_eval.json"
        run_settings = settings_for_run(base_settings, args.seed_policy, run_index)

        started_at = utc_now()
        results = run_triage_packets(
            packets,
            prompt_template=prompt_template,
            prompt_sha256=prompt_sha256,
            settings=run_settings,
            cache_dir=None,
            retries=args.retries,
            dry_run=args.dry_run,
        )
        write_jsonl(results, output_jsonl)
        metadata = build_run_metadata(
            results,
            input_jsonl=args.input_jsonl,
            output_jsonl=str(output_jsonl),
            prompt_template=str(prompt_path),
            prompt_sha256=prompt_sha256,
            settings=run_settings,
            started_at=started_at,
            dry_run=args.dry_run,
            cache_enabled=False,
            start_offset=args.start_offset,
            max_packets=args.max_packets,
        )
        output_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        run_eval = evaluate_one_run(
            run_index=run_index,
            llm_jsonl=output_jsonl,
            regions=regions,
            df=df,
            true_clusters=true_clusters,
            totals=base_report["totals"],
            actual_by_region=actual_by_region,
        )
        output_eval_json.write_text(json.dumps(run_eval, indent=2), encoding="utf-8")
        run_rows.append({**metadata, **run_eval["detector_metrics"], **run_eval["coverage_metrics"]})
        per_packet_rows.extend(run_eval["packet_decisions"])
        print(json.dumps(_console_run_summary(run_index, metadata, run_eval), indent=2))

    aggregate = build_aggregate_report(
        args=args,
        settings=base_settings,
        prompt_sha256=prompt_sha256,
        base_report=base_report,
        run_rows=run_rows,
        per_packet_rows=per_packet_rows,
        bootstrap_samples=args.bootstrap_samples,
        bootstrap_seed=args.bootstrap_seed,
    )
    (output_dir / "stability_summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    write_markdown(aggregate, output_dir / "stability_summary.md")
    print(json.dumps(aggregate["headline"], indent=2))


def evaluate_one_run(
    *,
    run_index: int,
    llm_jsonl: Path,
    regions: pd.DataFrame,
    df: pd.DataFrame,
    true_clusters: pd.DataFrame,
    totals: dict[str, Any],
    actual_by_region: dict[int, bool],
) -> dict[str, Any]:
    decisions = _read_decisions(llm_jsonl)
    merged = regions.merge(decisions, on="region_id", how="left")
    selected = merged[merged["recommended_action"].isin(INSPECT_ACTIONS)].copy()
    ignored = merged[merged["recommended_action"].eq("ignore")].copy()
    selected_metrics = evaluate_streaming_regions(selected, df, true_clusters, totals=totals)
    ignored_metrics = evaluate_streaming_regions(ignored, df, true_clusters, totals=totals)

    detector_rows = []
    for _, row in merged.iterrows():
        region_id = int(row["region_id"])
        predicted_positive = row.get("recommended_action") in INSPECT_ACTIONS
        actual_positive = actual_by_region[region_id]
        detector_rows.append(
            {
                "run_index": run_index,
                "region_id": region_id,
                "packet_id": row.get("packet_id"),
                "predicted_positive": bool(predicted_positive),
                "actual_positive": bool(actual_positive),
                "review_decision": row.get("review_decision"),
                "recommended_action": row.get("recommended_action"),
                "triage_score": row.get("triage_score"),
                "valid_json": bool(row.get("valid_json")),
            }
        )
    detail = pd.DataFrame(detector_rows)
    tp = int(((detail["predicted_positive"]) & (detail["actual_positive"])).sum())
    fp = int(((detail["predicted_positive"]) & (~detail["actual_positive"])).sum())
    tn = int(((~detail["predicted_positive"]) & (~detail["actual_positive"])).sum())
    fn = int(((~detail["predicted_positive"]) & (detail["actual_positive"])).sum())
    precision = safe_ratio(tp, tp + fp)
    recall = safe_ratio(tp, tp + fn)
    f1 = safe_ratio(2 * precision * recall, precision + recall)
    return {
        "detector_metrics": {
            "run_index": run_index,
            "llm_positive_packets": int(detail["predicted_positive"].sum()),
            "actual_positive_packets": int(detail["actual_positive"].sum()),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "packet_precision": precision,
            "packet_recall": recall,
            "packet_f1": f1,
        },
        "coverage_metrics": {
            "event_recall_in_llm_positive_packets": selected_metrics["anomaly_recall_against_all"],
            "cluster_recall_in_llm_positive_packets": selected_metrics["true_cluster_recall"],
            "stream_load_in_llm_positive_intervals": selected_metrics["stream_load"],
            "event_density_in_llm_positive_intervals": selected_metrics["region_weighted_density"],
            "ignored_anomaly_events": ignored_metrics["covered_anomaly_events"],
        },
        "counts": {
            "decision_counts": _counts(decisions, "review_decision"),
            "action_counts": _counts(decisions, "recommended_action"),
        },
        "packet_decisions": detector_rows,
    }


def build_aggregate_report(
    *,
    args: argparse.Namespace,
    settings: LlmSettings,
    prompt_sha256: str,
    base_report: dict[str, Any],
    run_rows: list[dict[str, Any]],
    per_packet_rows: list[dict[str, Any]],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    metrics = [
        "valid_json_rate",
        "llm_positive_packets",
        "packet_precision",
        "packet_recall",
        "packet_f1",
        "event_recall_in_llm_positive_packets",
        "cluster_recall_in_llm_positive_packets",
        "stream_load_in_llm_positive_intervals",
        "event_density_in_llm_positive_intervals",
        "ignored_anomaly_events",
        "mean_latency_ms_uncached",
        "p95_latency_ms_uncached",
        "total_tokens_reported",
    ]
    summaries = {metric: summarize_metric(run_rows, metric, bootstrap_samples, bootstrap_seed) for metric in metrics}
    agreement = summarize_agreement(per_packet_rows)
    headline = {
        "n_runs": len(run_rows),
        "packets_per_run": int(run_rows[0]["num_packets"]) if run_rows else 0,
        "packet_f1_mean": summaries["packet_f1"]["mean"],
        "packet_f1_ci95": summaries["packet_f1"]["bootstrap_ci95"],
        "packet_recall_min": summaries["packet_recall"]["min"],
        "event_recall_min": summaries["event_recall_in_llm_positive_packets"]["min"],
        "cluster_recall_min": summaries["cluster_recall_in_llm_positive_packets"]["min"],
        "mean_packet_agreement": agreement["mean_positive_vote_share"],
        "unanimous_packets": agreement["unanimous_packets"],
    }
    return {
        "experiment": {
            "name": "streaming_llm_stability_gap100_v3",
            "created_at": utc_now(),
            "purpose": "Estimate run-to-run stability of the LLM incident detector under fixed packets, prompt, and decoding settings.",
            "n_runs": len(run_rows),
            "cache_enabled": False,
            "same_seed_repeated": args.seed_policy == "fixed",
            "seed_policy": args.seed_policy,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": bootstrap_seed,
        },
        "fixed_inputs": {
            "input_jsonl": args.input_jsonl,
            "regions_csv": args.regions_csv,
            "streaming_report_json": args.streaming_report_json,
            "input_csv": args.input_csv,
            "true_clusters_csv": args.true_clusters_csv,
            "prompt_template": args.prompt_template,
            "prompt_template_sha256": prompt_sha256,
            "streaming_packetization_parameters": base_report.get("parameters"),
            "streaming_thresholds": base_report.get("thresholds"),
            "streaming_totals": base_report.get("totals"),
        },
        "llm_settings": settings.as_metadata(),
        "headline": headline,
        "metric_summaries": summaries,
        "agreement": agreement,
        "runs": compact_run_rows(run_rows),
    }


def summarize_metric(
    rows: list[dict[str, Any]],
    key: str,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    if not values:
        return {"count": 0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
    return {
        "count": len(values),
        "mean": mean,
        "std": math.sqrt(variance),
        "min": min(values),
        "max": max(values),
        "bootstrap_ci95": bootstrap_ci(values, bootstrap_samples, bootstrap_seed + stable_int(key)),
    }


def bootstrap_ci(values: list[float], samples: int, seed: int) -> list[float]:
    if len(values) == 1:
        return [values[0], values[0]]
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(sample) / len(sample))
    means.sort()
    low = means[int(0.025 * (len(means) - 1))]
    high = means[int(0.975 * (len(means) - 1))]
    return [low, high]


def summarize_agreement(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_region: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_region[int(row["region_id"])].append(row)
    packet_rows = []
    unanimous = 0
    for region_id, items in sorted(by_region.items()):
        positive_votes = sum(1 for item in items if item["predicted_positive"])
        total = len(items)
        vote_share = positive_votes / total if total else 0.0
        actions = Counter(str(item.get("recommended_action")) for item in items)
        decisions = Counter(str(item.get("review_decision")) for item in items)
        if positive_votes in {0, total}:
            unanimous += 1
        packet_rows.append(
            {
                "region_id": region_id,
                "actual_positive": bool(items[0]["actual_positive"]),
                "positive_votes": positive_votes,
                "total_votes": total,
                "positive_vote_share": vote_share,
                "majority_positive": positive_votes >= (total / 2),
                "action_counts": dict(actions),
                "decision_counts": dict(decisions),
            }
        )
    shares = [max(row["positive_vote_share"], 1.0 - row["positive_vote_share"]) for row in packet_rows]
    return {
        "packets": len(packet_rows),
        "unanimous_packets": unanimous,
        "non_unanimous_packets": len(packet_rows) - unanimous,
        "mean_positive_vote_share": sum(shares) / len(shares) if shares else None,
        "packet_vote_table": packet_rows,
    }


def compact_run_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        "run_index",
        "valid_json",
        "invalid_json",
        "valid_json_rate",
        "llm_positive_packets",
        "tp",
        "fp",
        "tn",
        "fn",
        "packet_precision",
        "packet_recall",
        "packet_f1",
        "event_recall_in_llm_positive_packets",
        "cluster_recall_in_llm_positive_packets",
        "stream_load_in_llm_positive_intervals",
        "ignored_anomaly_events",
        "mean_latency_ms_uncached",
        "total_tokens_reported",
    ]
    return [{key: row.get(key) for key in keys} for row in rows]


def write_markdown(report: dict[str, Any], path: Path) -> None:
    h = report["headline"]
    lines = [
        "# Streaming LLM Stability Experiment",
        "",
        "This repeats the same incident-level LLM detector on the fixed `gap100/close100` packet set. Cache is disabled, the prompt and decoding settings are held constant, and labels are used only after each run for offline evaluation.",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| repeated runs | {h['n_runs']} |",
        f"| packets per run | {h['packets_per_run']} |",
        f"| packet F1 mean | {fmt(h['packet_f1_mean'])} |",
        f"| packet F1 95% bootstrap CI | {fmt_ci(h['packet_f1_ci95'])} |",
        f"| minimum packet recall | {fmt(h['packet_recall_min'])} |",
        f"| minimum event recall | {fmt(h['event_recall_min'])} |",
        f"| minimum cluster recall | {fmt(h['cluster_recall_min'])} |",
        f"| unanimous packets | {h['unanimous_packets']} / {h['packets_per_run']} |",
        "",
        "## Fixed Parameters",
        "",
        "| Parameter | Value |",
        "| --- | --- |",
        f"| packet input | `{report['fixed_inputs']['input_jsonl']}` |",
        f"| regions | `{report['fixed_inputs']['regions_csv']}` |",
        f"| prompt | `{report['fixed_inputs']['prompt_template']}` |",
        f"| prompt sha256 | `{report['fixed_inputs']['prompt_template_sha256']}` |",
        f"| cache | disabled |",
    ]
    for key, value in report["llm_settings"].items():
        lines.append(f"| LLM `{key}` | `{value}` |")
    for key, value in (report["fixed_inputs"].get("streaming_packetization_parameters") or {}).items():
        lines.append(f"| packetizer `{key}` | `{value}` |")
    for key, value in (report["fixed_inputs"].get("streaming_thresholds") or {}).items():
        lines.append(f"| threshold `{key}` | `{value}` |")

    lines.extend(
        [
            "",
            "## Metric Stability",
            "",
            "| Metric | Mean | Std | Min | Max | 95% bootstrap CI |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, summary in report["metric_summaries"].items():
        if not summary.get("count"):
            continue
        lines.append(
            f"| `{key}` | {fmt(summary['mean'])} | {fmt(summary['std'])} | {fmt(summary['min'])} | {fmt(summary['max'])} | {fmt_ci(summary['bootstrap_ci95'])} |"
        )

    lines.extend(
        [
            "",
            "## Per-Run Results",
            "",
            "| Run | Valid JSON | LLM-positive | TP | FP | TN | FN | Precision | Recall | F1 | Event recall | Cluster recall | Stream load | Tokens |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in report["runs"]:
        lines.append(
            "| {run_index} | {valid_json} | {llm_positive_packets} | {tp} | {fp} | {tn} | {fn} | {precision} | {recall} | {f1} | {event_recall} | {cluster_recall} | {stream_load} | {tokens} |".format(
                run_index=row["run_index"],
                valid_json=row["valid_json"],
                llm_positive_packets=row["llm_positive_packets"],
                tp=row["tp"],
                fp=row["fp"],
                tn=row["tn"],
                fn=row["fn"],
                precision=fmt(row["packet_precision"]),
                recall=fmt(row["packet_recall"]),
                f1=fmt(row["packet_f1"]),
                event_recall=fmt(row["event_recall_in_llm_positive_packets"]),
                cluster_recall=fmt(row["cluster_recall_in_llm_positive_packets"]),
                stream_load=fmt(row["stream_load_in_llm_positive_intervals"]),
                tokens=f"{int(row['total_tokens_reported']):,}",
            )
        )

    lines.extend(
        [
            "",
            "## Decision Agreement",
            "",
            f"- Unanimous inspect/ignore decision: `{report['agreement']['unanimous_packets']} / {report['agreement']['packets']}` packets.",
            f"- Non-unanimous packets: `{report['agreement']['non_unanimous_packets']}`.",
            "- `mean_positive_vote_share` reports the mean majority-vote share, so higher is more stable.",
            "",
            "## Artifacts",
            "",
            f"- `{path.parent / 'stability_summary.json'}`",
            f"- per-run JSONL/metadata/evaluation files under `{path.parent}`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _actual_positive_by_region(regions: pd.DataFrame, df: pd.DataFrame) -> dict[int, bool]:
    out = {}
    for _, row in regions.iterrows():
        out[int(row["region_id"])] = count_anomalies(df, row) > 0
    return out


def settings_for_run(settings: LlmSettings, seed_policy: str, run_index: int) -> LlmSettings:
    if seed_policy == "fixed" or settings.seed is None:
        return settings
    return LlmSettings(
        provider=settings.provider,
        endpoint=settings.endpoint,
        model=settings.model,
        model_quantization=settings.model_quantization,
        prompt_template_id=settings.prompt_template_id,
        temperature=settings.temperature,
        top_p=settings.top_p,
        top_k=settings.top_k,
        seed=settings.seed + run_index - 1,
        num_ctx=settings.num_ctx,
        num_predict=settings.num_predict,
        repeat_penalty=settings.repeat_penalty,
        think_mode=settings.think_mode,
        json_mode=settings.json_mode,
        timeout_seconds=settings.timeout_seconds,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def stable_int(text: str) -> int:
    return sum((index + 1) * ord(char) for index, char in enumerate(text))


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def fmt_ci(values: list[float] | None) -> str:
    if not values:
        return "-"
    return f"[{fmt(values[0])}, {fmt(values[1])}]"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _console_run_summary(run_index: int, metadata: dict[str, Any], run_eval: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": run_index,
        "valid_json": metadata["valid_json"],
        "decision_counts": metadata["decision_counts"],
        "action_counts": metadata["action_counts"],
        "detector_metrics": run_eval["detector_metrics"],
    }


if __name__ == "__main__":
    main()
