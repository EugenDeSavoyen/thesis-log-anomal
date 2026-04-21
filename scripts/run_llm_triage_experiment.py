from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.augment_cluster_packets_with_retrieval import (  # noqa: E402
    compact_cluster_packet,
    encode_queries,
    retrieve_context_for_packet,
)
from scripts.build_log_retrieval_index import DEFAULT_MODEL_PATH  # noqa: E402
from scripts.build_llm_cluster_packets import build_template_cluster_packets, write_jsonl as write_packet_jsonl  # noqa: E402
from scripts.evaluate_llm_triage import (  # noqa: E402
    build_baseline_comparison,
    build_llm_summary,
    infer_totals,
    load_llm_outputs,
    merge_llm_with_labels,
)
from scripts.run_classical_triage_baseline import DEFAULT_METHODS, build_classical_triage_summary  # noqa: E402
from scripts.run_llm_triage import (  # noqa: E402
    LlmSettings,
    build_run_metadata,
    run_triage_packets,
    utc_now,
)


DEFAULT_VARIANTS = ["cluster", "cluster_retrieval", "cluster_rerank"]
DEFAULT_TOP_K = [20, 100, 500]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run reproducible local LLM triage experiments and compare them with classical rank baselines."
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/event_level_candidates_bgl_multiblock.csv",
    )
    parser.add_argument("--dataset", default="bgl_multiblock")
    parser.add_argument("--selection-method", default="template_burst", choices=DEFAULT_METHODS)
    parser.add_argument(
        "--top-k",
        default=",".join(str(value) for value in DEFAULT_TOP_K),
        help="Comma-separated candidate budgets. Cluster packets are built separately for each budget.",
    )
    parser.add_argument(
        "--eval-top-k",
        default=None,
        help="Comma-separated review budgets evaluated inside each candidate budget. Defaults to --top-k.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants: cluster, cluster_retrieval, cluster_rerank.",
    )
    parser.add_argument("--max-clusters", type=int, default=None)
    parser.add_argument("--max-events-per-cluster", type=int, default=8)
    parser.add_argument("--prompt-cluster", default="prompts/log_triage_cluster_v1.md")
    parser.add_argument("--prompt-retrieval", default="prompts/log_triage_cluster_contrast_v1.md")
    parser.add_argument("--prompt-rerank", default="prompts/log_triage_cluster_rerank_v1.md")
    parser.add_argument("--baseline-csv", default="outputs/reports/classical_triage_baseline_summary.csv")
    parser.add_argument(
        "--baseline-method",
        default=None,
        help="Classical method used for delta rows. Defaults to --selection-method.",
    )
    parser.add_argument(
        "--classical-methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated classical methods to recompute for the same budgets.",
    )
    parser.add_argument("--event-cv-json", default="outputs/reports/event_level_cv_bgl_multiblock.json")
    parser.add_argument("--total-stream-events", type=int, default=None)
    parser.add_argument("--total-anomalies", type=int, default=None)
    parser.add_argument("--retrieval-index-meta", default="outputs/reports/log_retrieval_index_meta.csv")
    parser.add_argument("--retrieval-index-embeddings", default="outputs/models/log_retrieval_index_embeddings.npy")
    parser.add_argument("--retrieval-model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--nearest-k", type=int, default=1)
    parser.add_argument("--same-template-k", type=int, default=2)
    parser.add_argument("--low-score-k", type=int, default=2)
    parser.add_argument("--max-query-events", type=int, default=3)
    parser.add_argument("--max-representative-events", type=int, default=3)
    parser.add_argument("--max-message-chars", type=int, default=180)
    parser.add_argument("--drop-nearest-examples", action="store_true")
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--model-quantization", default="Q4_K_M")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k-sampling", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Render requests and write evaluation-shaped outputs without calling the model.")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output-dir", default="outputs/reports/llm_triage_experiment")
    args = parser.parse_args()
    if args.baseline_method is None:
        args.baseline_method = args.selection_method

    top_k_values = parse_int_list(args.top_k)
    eval_top_k_values = parse_int_list(args.eval_top_k) if args.eval_top_k else top_k_values
    variants = parse_variants(args.variants)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.input_csv)
    totals = infer_totals(
        candidates,
        event_cv_json=Path(args.event_cv_json),
        total_stream_events=args.total_stream_events,
        total_anomalies=args.total_anomalies,
    )
    classical_methods = parse_methods(args.classical_methods)
    classical_summary = build_classical_triage_summary(
        candidates,
        methods=classical_methods,
        top_k=eval_top_k_values,
        totals=totals,
    )
    classical_summary_path = output_dir / "classical_same_budget_summary.csv"
    classical_summary.to_csv(classical_summary_path, index=False)
    retrieval_assets = load_retrieval_assets(args) if "cluster_retrieval" in variants and not args.dry_run else None

    run_rows = []
    summary_frames = []
    comparison_frames = []
    for budget in top_k_values:
        packets = build_template_cluster_packets(
            candidates,
            dataset=args.dataset,
            selection_method=args.selection_method,
            top_k=budget,
            max_clusters=args.max_clusters,
            max_events_per_cluster=args.max_events_per_cluster,
            prompt_strategy="template_cluster_v1",
        )
        packet_path = output_dir / f"cluster_packets_top{budget}.jsonl"
        packet_report_path = output_dir / f"cluster_packets_top{budget}.json"
        write_packet_jsonl(packets, packet_path)
        packet_report_path.write_text(
            json.dumps(build_packet_report(packets, args=args, budget=budget, packet_path=packet_path), indent=2),
            encoding="utf-8",
        )

        for variant in variants:
            variant_packets = packets
            source_packet_path = packet_path
            prompt_path = Path(args.prompt_cluster)
            if variant == "cluster_retrieval":
                prompt_path = Path(args.prompt_retrieval)
                variant_packets = (
                    dry_run_retrieval_packets(packets, args=args)
                    if args.dry_run
                    else augment_packets_with_retrieval(packets, retrieval_assets, args=args)
                )
                source_packet_path = output_dir / f"cluster_packets_top{budget}_retrieval.jsonl"
                write_packet_jsonl(variant_packets, source_packet_path)
            elif variant == "cluster_rerank":
                prompt_path = Path(args.prompt_rerank)

            prompt_template = prompt_path.read_text(encoding="utf-8")
            llm_path = output_dir / f"{variant}_top{budget}_llm.jsonl"
            llm_meta_path = output_dir / f"{variant}_top{budget}_llm_run.json"
            eval_summary_path = output_dir / f"{variant}_top{budget}_evaluation_summary.csv"
            eval_events_path = output_dir / f"{variant}_top{budget}_evaluation_events.csv"

            settings = LlmSettings(
                model=args.model,
                model_quantization=args.model_quantization,
                prompt_template_id=prompt_path.stem,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k_sampling,
                seed=args.seed,
                num_ctx=args.num_ctx,
                num_predict=args.num_predict,
                repeat_penalty=args.repeat_penalty,
                timeout_seconds=args.timeout_seconds,
            )
            started_at = utc_now()
            cache_dir = None if args.no_cache else output_dir / f"{variant}_top{budget}_cache"
            if cache_dir is not None:
                cache_dir.mkdir(parents=True, exist_ok=True)
            results = run_triage_packets(
                variant_packets,
                prompt_template=prompt_template,
                prompt_sha256=sha256_text(prompt_template),
                settings=settings,
                cache_dir=cache_dir,
                retries=args.retries,
                dry_run=args.dry_run,
            )
            write_result_jsonl(results, llm_path)
            metadata = build_run_metadata(
                results,
                input_jsonl=str(source_packet_path),
                output_jsonl=str(llm_path),
                prompt_template=str(prompt_path),
                prompt_sha256=sha256_text(prompt_template),
                settings=settings,
                started_at=started_at,
                dry_run=args.dry_run,
                cache_enabled=not args.no_cache,
                start_offset=0,
                max_packets=None,
            )
            metadata.update({"experiment_variant": variant, "candidate_top_k": budget})
            llm_meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            evaluated, summary, comparison = evaluate_variant(
                llm_path,
                source_packet_path,
                candidates,
                top_k_values=[value for value in eval_top_k_values if value <= budget],
                totals=totals,
                baseline=classical_summary,
                baseline_method=args.baseline_method,
            )
            summary.insert(0, "variant", variant)
            summary.insert(1, "candidate_top_k", budget)
            evaluated.to_csv(eval_events_path, index=False)
            summary.to_csv(eval_summary_path, index=False)
            summary_frames.append(summary)
            if not comparison.empty:
                comparison.insert(0, "variant", variant)
                comparison.insert(1, "candidate_top_k", budget)
                comparison_frames.append(comparison)
            run_rows.append(
                {
                    "variant": variant,
                    "candidate_top_k": budget,
                    "packet_jsonl": str(source_packet_path),
                    "llm_jsonl": str(llm_path),
                    "llm_metadata_json": str(llm_meta_path),
                    "evaluation_summary_csv": str(eval_summary_path),
                    "evaluation_events_csv": str(eval_events_path),
                    "num_packets": len(variant_packets),
                    "valid_json": metadata["valid_json"],
                    "invalid_json": metadata["invalid_json"],
                    "total_tokens": metadata["total_tokens_reported"],
                    "dry_run": args.dry_run,
                }
            )

    combined_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    combined_comparison = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()
    run_table = pd.DataFrame(run_rows)
    combined_summary_path = output_dir / "llm_experiment_summary.csv"
    combined_comparison_path = output_dir / "llm_experiment_classical_comparison.csv"
    run_table_path = output_dir / "llm_experiment_runs.csv"
    report_json_path = output_dir / "llm_experiment_report.json"
    report_md_path = output_dir / "llm_experiment_report.md"
    combined_summary.to_csv(combined_summary_path, index=False)
    combined_comparison.to_csv(combined_comparison_path, index=False)
    run_table.to_csv(run_table_path, index=False)
    report = build_experiment_report(
        args=args,
        totals=totals,
        run_table=run_table,
        combined_summary=combined_summary,
        combined_comparison=combined_comparison,
        classical_summary=classical_summary,
        output_dir=output_dir,
        outputs={
            "classical_summary_csv": str(classical_summary_path),
            "summary_csv": str(combined_summary_path),
            "comparison_csv": str(combined_comparison_path),
            "runs_csv": str(run_table_path),
            "report_json": str(report_json_path),
            "report_md": str(report_md_path),
        },
    )
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, combined_summary, combined_comparison, report_md_path)
    print(json.dumps(report, indent=2))


def load_retrieval_assets(args: argparse.Namespace) -> dict[str, Any]:
    meta = pd.read_csv(args.retrieval_index_meta)
    embeddings = np.load(args.retrieval_index_embeddings).astype(np.float32)
    if len(meta) != len(embeddings):
        raise ValueError(f"Retrieval metadata rows ({len(meta)}) do not match embeddings ({len(embeddings)}).")
    return {"meta": meta, "embeddings": embeddings}


def augment_packets_with_retrieval(
    packets: list[dict[str, Any]],
    retrieval_assets: dict[str, Any] | None,
    *,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if retrieval_assets is None:
        raise ValueError("Retrieval assets are required for cluster_retrieval unless --dry-run is used.")
    query_texts = [build_cluster_query_text(packet, max_query_events=args.max_query_events) for packet in packets]
    query_embeddings = encode_queries(query_texts, model_path=args.retrieval_model_path)
    augmented = []
    for packet, query_embedding in zip(packets, query_embeddings, strict=True):
        updated = compact_cluster_packet(
            packet,
            max_representative_events=args.max_representative_events,
            max_member_ids=1000,
            max_message_chars=args.max_message_chars,
        )
        updated["prompt_strategy"] = "template_cluster_retrieval_v1"
        updated["retrieved_context"] = retrieve_context_for_packet(
            packet,
            retrieval_assets["meta"],
            retrieval_assets["embeddings"],
            query_embedding,
            nearest_k=args.nearest_k,
            same_template_k=args.same_template_k,
            low_score_k=args.low_score_k,
            max_message_chars=args.max_message_chars,
            include_nearest=not args.drop_nearest_examples,
        )
        augmented.append(updated)
    return augmented


def dry_run_retrieval_packets(packets: list[dict[str, Any]], *, args: argparse.Namespace) -> list[dict[str, Any]]:
    augmented = []
    for packet in packets:
        updated = compact_cluster_packet(
            packet,
            max_representative_events=args.max_representative_events,
            max_member_ids=1000,
            max_message_chars=args.max_message_chars,
        )
        updated["prompt_strategy"] = "template_cluster_retrieval_v1"
        updated["retrieved_context"] = {
            "retrieval_context_schema_version": "retrieval_context_v1",
            "nearest_examples": [],
            "same_template_examples": [],
            "lower_score_contrast_examples": [],
            "notes": ["Dry run placeholder; retrieval index was not queried."],
        }
        augmented.append(updated)
    return augmented


def build_cluster_query_text(packet: dict[str, Any], *, max_query_events: int) -> str:
    cluster = packet.get("cluster", {})
    parts = [f"Template: {cluster.get('template', '')}", "Representative messages:"]
    for event in packet.get("representative_events", [])[:max_query_events]:
        parts.append(str(event.get("message", "")))
    return "\n".join(parts)


def evaluate_variant(
    llm_path: Path,
    packet_path: Path,
    candidates: pd.DataFrame,
    *,
    top_k_values: list[int],
    totals: dict[str, int],
    baseline: pd.DataFrame,
    baseline_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    packet_rows = read_jsonl(packet_path)
    llm_df = load_llm_outputs(llm_path, packet_rows=packet_rows, evaluation_mode="cluster")
    evaluated = merge_llm_with_labels(llm_df, candidates)
    summary = build_llm_summary(evaluated, top_k_values=top_k_values, totals=totals)
    baseline_subset = baseline[
        (baseline["method"] == baseline_method) & (baseline["top_k"].isin(top_k_values))
    ].copy()
    comparison = build_baseline_comparison(summary, baseline_subset)
    return evaluated, summary, comparison


def build_packet_report(
    packets: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    budget: int,
    packet_path: Path,
) -> dict[str, Any]:
    event_counts = [int(packet["cluster"]["event_count"]) for packet in packets]
    return {
        "dataset": args.dataset,
        "selection_method": args.selection_method,
        "candidate_top_k": budget,
        "max_clusters": args.max_clusters,
        "max_events_per_cluster": args.max_events_per_cluster,
        "num_packets": len(packets),
        "covered_events": sum(event_counts),
        "max_cluster_events": max(event_counts) if event_counts else 0,
        "output_jsonl": str(packet_path),
        "label_policy": "labels and evaluation metrics are excluded from LLM packets",
    }


def build_experiment_report(
    *,
    args: argparse.Namespace,
    totals: dict[str, int],
    run_table: pd.DataFrame,
    combined_summary: pd.DataFrame,
    combined_comparison: pd.DataFrame,
    classical_summary: pd.DataFrame,
    output_dir: Path,
    outputs: dict[str, str],
) -> dict[str, Any]:
    best_f1 = best_row(combined_summary, "f1")
    best_precision = best_row(combined_summary, "precision")
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "local_llm_triage_on_suspicious_logs",
        "input_csv": args.input_csv,
        "dataset": args.dataset,
        "selection_method": args.selection_method,
        "candidate_top_k": parse_int_list(args.top_k),
        "eval_top_k": parse_int_list(args.eval_top_k) if args.eval_top_k else parse_int_list(args.top_k),
        "variants": parse_variants(args.variants),
        "dry_run": args.dry_run,
        "totals": totals,
        "output_dir": str(output_dir),
        "num_runs": int(len(run_table)),
        "best_precision_row": best_precision,
        "best_f1_row": best_f1,
        "best_classical_precision_row": best_row(classical_summary, "precision"),
        "best_classical_f1_row": best_row(classical_summary, "f1"),
        "outputs": outputs,
        "comparison_rows": combined_comparison.to_dict("records") if not combined_comparison.empty else [],
        "notes": [
            "The LLM receives only suspicious logs selected by the classical candidate generator.",
            "Labels are merged only during offline evaluation.",
            "cluster_retrieval uses unlabeled retrieval context and should be checked for recall loss.",
        ],
    }


def write_markdown_report(
    report: dict[str, Any],
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# LLM Triage Experiment",
        "",
        "This experiment compares local no-thinking LLM triage on suspicious log clusters against the classical rank baseline.",
        "",
        "## Run Summary",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Candidate score: `{report['selection_method']}`",
        f"- Top-K budgets: `{report['candidate_top_k']}`",
        f"- Evaluation budgets: `{report['eval_top_k']}`",
        f"- Variants: `{report['variants']}`",
        f"- Dry run: `{report['dry_run']}`",
        "",
        "## LLM Policy Metrics",
        "",
        "| Variant | Budget | Policy | Selected | Precision | Recall all anomalies | F1 | Stream load | Valid JSON | Tokens |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            "| {variant} | {budget} | {policy} | {selected} | {precision} | {recall} | {f1} | {load} | {valid} | {tokens} |".format(
                variant=row["variant"],
                budget=int(row["candidate_top_k"]),
                policy=row["policy"],
                selected=int(row["selected_events"]),
                precision=format_optional(row["precision"]),
                recall=format_optional(row["recall_against_all_anomalies"]),
                f1=format_optional(row["f1"]),
                load=format_optional(row["event_load_ratio_against_stream"]),
                valid=format_optional(row["valid_json_rate"]),
                tokens=int(row["total_tokens"]),
            )
        )
    if not comparison.empty:
        lines.extend(
            [
                "",
                "## Delta Vs Classical",
                "",
                "| Variant | Budget | Policy | Precision delta | Recall delta | F1 delta | Stream-load delta |",
                "| --- | ---: | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in comparison.to_dict("records"):
            lines.append(
                "| {variant} | {budget} | {policy} | {precision} | {recall} | {f1} | {load} |".format(
                    variant=row["variant"],
                    budget=int(row["candidate_top_k"]),
                    policy=row["policy"],
                    precision=format_optional(row.get("precision_delta_vs_classical")),
                    recall=format_optional(row.get("recall_against_all_anomalies_delta_vs_classical")),
                    f1=format_optional(row.get("f1_delta_vs_classical")),
                    load=format_optional(row.get("event_load_ratio_against_stream_delta_vs_classical")),
                )
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Use this report to decide whether the LLM triage layer reduces review load or improves analyst-facing explanations without losing too much recall.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_result_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def parse_int_list(raw: str) -> list[int]:
    return sorted({int(item.strip()) for item in raw.split(",") if item.strip() and int(item.strip()) > 0})


def parse_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    allowed = {"cluster", "cluster_retrieval", "cluster_rerank"}
    unknown = [variant for variant in variants if variant not in allowed]
    if unknown:
        raise ValueError(f"Unknown variant(s): {', '.join(unknown)}")
    return variants


def parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [method for method in methods if method not in DEFAULT_METHODS]
    if unknown:
        raise ValueError(f"Unknown classical method(s): {', '.join(unknown)}")
    return methods


def best_row(df: pd.DataFrame, metric: str) -> dict[str, Any] | None:
    if df.empty or metric not in df:
        return None
    ranked = df.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if ranked.empty:
        return None
    return ranked.iloc[0].to_dict()


def format_optional(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def sha256_text(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
