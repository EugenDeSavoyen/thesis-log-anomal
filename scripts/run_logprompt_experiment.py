from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_llm_review_packets import (  # noqa: E402
    DEFAULT_LOCAL_CONTEXT_COLUMNS,
    DEFAULT_SCORE_COLUMNS,
    DEFAULT_WINDOW_CONTEXT_COLUMNS,
    assert_no_label_leakage,
    write_jsonl as write_packet_jsonl,
)
from scripts.evaluate_llm_triage import (  # noqa: E402
    build_baseline_comparison,
    build_llm_summary,
    infer_totals,
    load_llm_outputs,
    merge_llm_with_labels,
)
from scripts.run_classical_triage_baseline import (  # noqa: E402
    DEFAULT_METHODS,
    build_classical_triage_summary,
    rank_events,
)
from scripts.run_llm_triage import (  # noqa: E402
    LlmSettings,
    build_run_metadata,
    run_triage_packets,
    utc_now,
)


LOGPROMPT_PACKET_SCHEMA_VERSION = "logprompt_event_packet_v1"
DEFAULT_VARIANTS = ["direct", "semantic_sequence", "fewshot_retrieval"]
PROMPT_BY_VARIANT = {
    "direct": "prompts/logprompt_direct_v1.md",
    "semantic_sequence": "prompts/logprompt_semantic_sequence_v1.md",
    "fewshot_retrieval": "prompts/logprompt_fewshot_retrieval_v1.md",
}
RETRIEVAL_SCORE_COLUMNS = [
    "classical_score",
    "template_burst_score",
    "template_count_deviation_score",
    "novelty_score",
    "rarity_score",
    "local_sequence_context_score",
    "markov_bigram_surprise",
    "markov_trigram_surprise",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LogPrompt-style local LLM triage on top of GEV/event-level suspicious candidates."
    )
    parser.add_argument("--input-csv", default="data/processed/event_level_candidates_bgl_multiblock.csv")
    parser.add_argument("--dataset", default="bgl_multiblock")
    parser.add_argument("--selection-method", default="novelty", choices=DEFAULT_METHODS)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument(
        "--eval-top-k",
        default="20,50,100",
        help="Comma-separated review budgets evaluated inside the LLM candidate set.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated variants: direct, semantic_sequence, fewshot_retrieval.",
    )
    parser.add_argument("--baseline-method", default=None)
    parser.add_argument(
        "--classical-methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated classical methods to recompute for the same evaluation budgets.",
    )
    parser.add_argument("--event-cv-json", default="outputs/reports/event_level_cv_bgl_multiblock.json")
    parser.add_argument("--total-stream-events", type=int, default=None)
    parser.add_argument("--total-anomalies", type=int, default=None)
    parser.add_argument("--neighbor-events", type=int, default=2)
    parser.add_argument("--same-template-examples", type=int, default=2)
    parser.add_argument("--score-neighbor-examples", type=int, default=2)
    parser.add_argument("--lower-score-examples", type=int, default=2)
    parser.add_argument("--max-message-chars", type=int, default=240)
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output-dir", default="outputs/reports/logprompt_experiment")
    args = parser.parse_args()

    if args.baseline_method is None:
        args.baseline_method = args.selection_method

    variants = parse_variants(args.variants)
    eval_top_k = [value for value in parse_int_list(args.eval_top_k) if value <= args.top_k]
    if not eval_top_k:
        raise ValueError("At least one --eval-top-k value must be <= --top-k.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(args.input_csv)
    totals = infer_totals(
        candidates,
        event_cv_json=Path(args.event_cv_json),
        total_stream_events=args.total_stream_events,
        total_anomalies=args.total_anomalies,
    )
    classical_summary = build_classical_triage_summary(
        candidates,
        methods=parse_methods(args.classical_methods),
        top_k=eval_top_k,
        totals=totals,
    )
    classical_summary_path = output_dir / "classical_same_budget_summary.csv"
    classical_summary.to_csv(classical_summary_path, index=False)

    packets = build_logprompt_packets(
        candidates,
        dataset=args.dataset,
        selection_method=args.selection_method,
        top_k=args.top_k,
        neighbor_events=args.neighbor_events,
        same_template_examples=args.same_template_examples,
        score_neighbor_examples=args.score_neighbor_examples,
        lower_score_examples=args.lower_score_examples,
        max_message_chars=args.max_message_chars,
    )
    packet_path = output_dir / f"logprompt_packets_{args.selection_method}_top{args.top_k}.jsonl"
    packet_report_path = output_dir / f"logprompt_packets_{args.selection_method}_top{args.top_k}.json"
    write_packet_jsonl(packets, packet_path)
    packet_report_path.write_text(
        json.dumps(build_packet_report(packets, args=args, packet_path=packet_path), indent=2),
        encoding="utf-8",
    )

    run_rows = []
    summary_frames = []
    comparison_frames = []
    for variant in variants:
        prompt_path = Path(PROMPT_BY_VARIANT[variant])
        prompt_template = prompt_path.read_text(encoding="utf-8")
        llm_path = output_dir / f"{variant}_top{args.top_k}_llm.jsonl"
        metadata_path = output_dir / f"{variant}_top{args.top_k}_llm_run.json"
        events_path = output_dir / f"{variant}_top{args.top_k}_evaluation_events.csv"
        summary_path = output_dir / f"{variant}_top{args.top_k}_evaluation_summary.csv"
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
        cache_dir = None if args.no_cache else output_dir / f"{variant}_top{args.top_k}_cache"
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
        results = run_triage_packets(
            packets,
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
            input_jsonl=str(packet_path),
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
        metadata.update(
            {
                "experiment": "logprompt_on_gev_event_candidates",
                "variant": variant,
                "selection_method": args.selection_method,
                "candidate_top_k": args.top_k,
                "eval_top_k": eval_top_k,
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        evaluated, summary, comparison = evaluate_variant(
            llm_path,
            packet_path,
            candidates,
            top_k_values=eval_top_k,
            totals=totals,
            baseline=classical_summary,
            baseline_method=args.baseline_method,
        )
        summary.insert(0, "variant", variant)
        summary.insert(1, "candidate_top_k", args.top_k)
        evaluated.to_csv(events_path, index=False)
        summary.to_csv(summary_path, index=False)
        summary_frames.append(summary)
        if not comparison.empty:
            comparison.insert(0, "variant", variant)
            comparison.insert(1, "candidate_top_k", args.top_k)
            comparison_frames.append(comparison)
        run_rows.append(
            {
                "variant": variant,
                "candidate_top_k": args.top_k,
                "prompt_template": str(prompt_path),
                "packet_jsonl": str(packet_path),
                "llm_jsonl": str(llm_path),
                "llm_metadata_json": str(metadata_path),
                "evaluation_summary_csv": str(summary_path),
                "evaluation_events_csv": str(events_path),
                "num_packets": len(packets),
                "valid_json": metadata["valid_json"],
                "invalid_json": metadata["invalid_json"],
                "total_tokens": metadata["total_tokens_reported"],
                "dry_run": args.dry_run,
            }
        )

    run_table = pd.DataFrame(run_rows)
    combined_summary = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    combined_comparison = pd.concat(comparison_frames, ignore_index=True) if comparison_frames else pd.DataFrame()

    run_table_path = output_dir / "logprompt_run_table.csv"
    combined_summary_path = output_dir / "logprompt_combined_summary.csv"
    combined_comparison_path = output_dir / "logprompt_vs_classical_comparison.csv"
    report_json_path = output_dir / "logprompt_experiment_report.json"
    report_md_path = output_dir / "logprompt_experiment_report.md"
    run_table.to_csv(run_table_path, index=False)
    combined_summary.to_csv(combined_summary_path, index=False)
    combined_comparison.to_csv(combined_comparison_path, index=False)

    report = build_experiment_report(
        args=args,
        totals=totals,
        run_table=run_table,
        combined_summary=combined_summary,
        combined_comparison=combined_comparison,
        output_dir=output_dir,
        outputs={
            "packets_jsonl": str(packet_path),
            "packet_report_json": str(packet_report_path),
            "classical_summary_csv": str(classical_summary_path),
            "run_table_csv": str(run_table_path),
            "combined_summary_csv": str(combined_summary_path),
            "comparison_csv": str(combined_comparison_path),
            "report_json": str(report_json_path),
            "report_md": str(report_md_path),
        },
    )
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown_report(report, combined_summary, combined_comparison, report_md_path)
    print(json.dumps(report, indent=2))


def build_logprompt_packets(
    df: pd.DataFrame,
    *,
    dataset: str,
    selection_method: str,
    top_k: int,
    neighbor_events: int,
    same_template_examples: int,
    score_neighbor_examples: int,
    lower_score_examples: int,
    max_message_chars: int,
) -> list[dict[str, Any]]:
    scored_candidates = rank_events(df, selection_method).copy()
    ranked = scored_candidates.head(top_k).copy()
    ranked["selection_rank"] = range(1, len(ranked) + 1)
    if "classical_score" not in ranked:
        ranked["classical_score"] = 0.0
    ordered = ordered_candidates(scored_candidates)
    packets = []
    for _, row in ranked.iterrows():
        packet = packet_from_row(
            scored_candidates,
            ordered,
            row,
            dataset=dataset,
            selection_method=selection_method,
            top_k=top_k,
            neighbor_events=neighbor_events,
            same_template_examples=same_template_examples,
            score_neighbor_examples=score_neighbor_examples,
            lower_score_examples=lower_score_examples,
            max_message_chars=max_message_chars,
        )
        assert_no_label_leakage(packet)
        packets.append(packet)
    return packets


def packet_from_row(
    full_df: pd.DataFrame,
    ordered: pd.DataFrame,
    row: pd.Series,
    *,
    dataset: str,
    selection_method: str,
    top_k: int,
    neighbor_events: int,
    same_template_examples: int,
    score_neighbor_examples: int,
    lower_score_examples: int,
    max_message_chars: int,
) -> dict[str, Any]:
    event_id = as_int(row.get("event_id"))
    selection_rank = as_int(row.get("selection_rank"))
    packet = {
        "packet_schema_version": LOGPROMPT_PACKET_SCHEMA_VERSION,
        "packet_id": f"{dataset}:logprompt:{selection_method}:top{top_k}:{selection_rank:06d}:event:{event_id}",
        "dataset": dataset,
        "candidate_source": "gev_suspicious_event_candidates",
        "selection_method": selection_method,
        "top_k_budget": int(top_k),
        "selection_rank": selection_rank,
        "event_id": event_id,
        "event_order": as_int(row.get("event_order")),
        "block_id": clean_value(row.get("block_id")),
        "template": clean_text(row.get("template"), max_chars=max_message_chars),
        "message": clean_text(row.get("message"), max_chars=max_message_chars),
        "semantic_prompt_fields": semantic_prompt_fields(row),
        "scores": pick_columns(row, ["classical_score", *DEFAULT_SCORE_COLUMNS]),
        "window_context": pick_columns(row, DEFAULT_WINDOW_CONTEXT_COLUMNS),
        "sequence_context": {
            "local_features": pick_columns(row, DEFAULT_LOCAL_CONTEXT_COLUMNS),
            "neighbor_events": neighbor_examples(
                ordered,
                row,
                radius=neighbor_events,
                max_message_chars=max_message_chars,
            ),
        },
        "retrieval_lite_examples": {
            "same_template_history": same_template_examples_for_row(
                ordered,
                row,
                limit=same_template_examples,
                max_message_chars=max_message_chars,
            ),
            "score_neighbor_examples": score_neighbor_examples_for_row(
                full_df,
                row,
                limit=score_neighbor_examples,
                max_message_chars=max_message_chars,
            ),
            "lower_score_contrast_examples": lower_score_examples_for_row(
                full_df,
                row,
                limit=lower_score_examples,
                max_message_chars=max_message_chars,
            ),
            "notes": [
                "Examples are label-free and may include normal or anomalous events.",
                "They are retrieval-lite context, not ground truth.",
            ],
        },
        "prompt_strategy": "logprompt_semantic_sequence_retrieval_v1",
    }
    return packet


def semantic_prompt_fields(row: pd.Series) -> dict[str, Any]:
    text = f"{row.get('template', '')} {row.get('message', '')}".lower()
    keywords = {
        "failure_terms": ["fail", "failed", "failure", "fatal", "error", "exception"],
        "communication_terms": ["socket", "connection", "link", "network", "control stream"],
        "filesystem_terms": ["mount", "read", "write", "filesystem", "file system", "io"],
        "hardware_terms": ["machine check", "corrected", "uncorrected", "memory", "processor", "hardware"],
    }
    matched = {}
    for group, terms in keywords.items():
        hits = [term for term in terms if term in text]
        if hits:
            matched[group] = hits
    return {
        "matched_semantic_terms": matched,
        "explicitly_corrected": "corrected" in text and "uncorrected" not in text,
        "failure_like_text": any(
            term in text
            for term in ["fail", "failed", "failure", "fatal", "error", "exception", "uncorrected", "severed"]
        ),
    }


def ordered_candidates(df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ["block_id", "event_order", "event_id"] if column in df.columns]
    if not sort_columns:
        return df
    return df.sort_values(sort_columns)


def neighbor_examples(
    ordered: pd.DataFrame,
    row: pd.Series,
    *,
    radius: int,
    max_message_chars: int,
) -> list[dict[str, Any]]:
    if radius <= 0 or "event_id" not in ordered:
        return []
    event_id = row.get("event_id")
    matches = ordered.index[ordered["event_id"] == event_id].tolist()
    if not matches:
        return []
    position = ordered.index.get_loc(matches[0])
    if isinstance(position, slice):
        position = position.start
    start = max(0, int(position) - radius)
    end = min(len(ordered), int(position) + radius + 1)
    examples = []
    for _, candidate in ordered.iloc[start:end].iterrows():
        if candidate.get("event_id") == event_id:
            continue
        relation = "previous_event" if candidate.get("event_order", 0) < row.get("event_order", 0) else "next_event"
        examples.append(compact_event(candidate, relation=relation, max_message_chars=max_message_chars))
    return examples


def same_template_examples_for_row(
    ordered: pd.DataFrame,
    row: pd.Series,
    *,
    limit: int,
    max_message_chars: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or "template" not in ordered:
        return []
    same = ordered[(ordered["template"] == row.get("template")) & (ordered["event_id"] != row.get("event_id"))].copy()
    if "event_order" in same and pd.notna(row.get("event_order")):
        earlier = same[same["event_order"] < row.get("event_order")]
        if not earlier.empty:
            same = earlier
    return [
        compact_event(candidate, relation="same_template_history", max_message_chars=max_message_chars)
        for _, candidate in same.tail(limit).iterrows()
    ]


def score_neighbor_examples_for_row(
    df: pd.DataFrame,
    row: pd.Series,
    *,
    limit: int,
    max_message_chars: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    columns = [column for column in RETRIEVAL_SCORE_COLUMNS if column in df.columns and column in row.index]
    if not columns:
        return []
    scored = df[df["event_id"] != row.get("event_id")].copy()
    for column in columns:
        row_value = numeric(row.get(column), default=0.0)
        scored[f"distance__{column}"] = (pd.to_numeric(scored[column], errors="coerce").fillna(0.0) - row_value).abs()
    distance_cols = [f"distance__{column}" for column in columns]
    scored["score_profile_distance"] = scored[distance_cols].sum(axis=1)
    nearest = scored.sort_values(["score_profile_distance", "event_order"], ascending=[True, True]).head(limit)
    return [
        compact_event(candidate, relation="score_neighbor", max_message_chars=max_message_chars)
        for _, candidate in nearest.iterrows()
    ]


def lower_score_examples_for_row(
    df: pd.DataFrame,
    row: pd.Series,
    *,
    limit: int,
    max_message_chars: int,
) -> list[dict[str, Any]]:
    if limit <= 0 or "classical_score" not in df:
        return []
    threshold = numeric(row.get("classical_score"), default=0.0)
    lower = df[(df["event_id"] != row.get("event_id")) & (pd.to_numeric(df["classical_score"], errors="coerce") < threshold)].copy()
    if lower.empty:
        return []
    lower = lower.sort_values("classical_score", ascending=False).head(limit)
    return [
        compact_event(candidate, relation="lower_score_contrast", max_message_chars=max_message_chars)
        for _, candidate in lower.iterrows()
    ]


def compact_event(row: pd.Series, *, relation: str, max_message_chars: int) -> dict[str, Any]:
    return {
        "relation": relation,
        "event_id": as_int(row.get("event_id")),
        "event_order": as_int(row.get("event_order")),
        "block_id": clean_value(row.get("block_id")),
        "template": clean_text(row.get("template"), max_chars=max_message_chars),
        "message": clean_text(row.get("message"), max_chars=max_message_chars),
        "scores": pick_columns(row, ["classical_score", "template_burst_score", "novelty_score", "rarity_score"]),
    }


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
    llm_df = load_llm_outputs(llm_path, packet_rows=packet_rows, evaluation_mode="event")
    evaluated = merge_llm_with_labels(llm_df, candidates)
    summary = build_llm_summary(evaluated, top_k_values=top_k_values, totals=totals)
    baseline_subset = baseline[
        (baseline["method"] == baseline_method) & (baseline["top_k"].isin(top_k_values))
    ].copy()
    comparison = build_baseline_comparison(summary, baseline_subset)
    return evaluated, summary, comparison


def build_packet_report(packets: list[dict[str, Any]], *, args: argparse.Namespace, packet_path: Path) -> dict[str, Any]:
    return {
        "packet_schema_version": LOGPROMPT_PACKET_SCHEMA_VERSION,
        "experiment": "logprompt_on_gev_event_candidates",
        "dataset": args.dataset,
        "input_csv": args.input_csv,
        "output_jsonl": str(packet_path),
        "selection_method": args.selection_method,
        "top_k": args.top_k,
        "num_packets": len(packets),
        "neighbor_events": args.neighbor_events,
        "same_template_examples": args.same_template_examples,
        "score_neighbor_examples": args.score_neighbor_examples,
        "lower_score_examples": args.lower_score_examples,
        "max_message_chars": args.max_message_chars,
        "packet_sha256": sha256_json(packets),
        "label_policy": "labels and evaluation metrics are excluded from LogPrompt packets",
        "notes": [
            "LogPrompt packets are built only from GEV/event-level suspicious candidates.",
            "Retrieval-lite examples are label-free context, not examples of known normal/anomalous classes.",
        ],
    }


def build_experiment_report(
    *,
    args: argparse.Namespace,
    totals: dict[str, int],
    run_table: pd.DataFrame,
    combined_summary: pd.DataFrame,
    combined_comparison: pd.DataFrame,
    output_dir: Path,
    outputs: dict[str, str],
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "logprompt_on_gev_event_candidates",
        "dataset": args.dataset,
        "input_csv": args.input_csv,
        "selection_method": args.selection_method,
        "candidate_top_k": args.top_k,
        "eval_top_k": parse_int_list(args.eval_top_k),
        "variants": parse_variants(args.variants),
        "dry_run": args.dry_run,
        "totals": totals,
        "output_dir": str(output_dir),
        "num_runs": int(len(run_table)),
        "best_precision_row": best_row(combined_summary, "precision"),
        "best_f1_row": best_row(combined_summary, "f1"),
        "outputs": outputs,
        "comparison_rows": combined_comparison.to_dict("records") if not combined_comparison.empty else [],
        "notes": [
            "All LLM calls are on top of GEV/event-level suspicious candidates.",
            "Labels are merged only during offline evaluation.",
            "This experiment adapts LogPrompt as zero/few-shot prompt strategies, not fine-tuning.",
        ],
    }


def write_markdown_report(
    report: dict[str, Any],
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# LogPrompt Experiment",
        "",
        "This experiment applies LogPrompt-style prompts on top of GEV/event-level suspicious candidates.",
        "",
        "## Run Summary",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Candidate score: `{report['selection_method']}`",
        f"- Candidate top-K: `{report['candidate_top_k']}`",
        f"- Evaluation top-K: `{report['eval_top_k']}`",
        f"- Variants: `{report['variants']}`",
        f"- Dry run: `{report['dry_run']}`",
        "",
        "## LLM Policy Metrics",
        "",
        "| Variant | Policy | Top-K input | Selected | Positives | Precision | Recall all anomalies | Recall in LLM budget | F1 | Stream load | Valid JSON | Tokens |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.to_dict("records"):
        lines.append(
            "| {variant} | {policy} | {top_k} | {selected} | {positives} | {precision} | {recall} | {budget_recall} | {f1} | {load} | {valid} | {tokens} |".format(
                variant=row["variant"],
                policy=row["policy"],
                top_k=int(row["top_k"]),
                selected=int(row["selected_events"]),
                positives=int(row["positive_events"]),
                precision=format_optional(row["precision"]),
                recall=format_optional(row["recall_against_all_anomalies"]),
                budget_recall=format_optional(row["recall_against_llm_budget_anomalies"]),
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
                "| Variant | Policy | Top-K | Precision delta | Recall delta | F1 delta | Stream-load delta |",
                "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in comparison.to_dict("records"):
            lines.append(
                "| {variant} | {policy} | {top_k} | {precision} | {recall} | {f1} | {load} |".format(
                    variant=row["variant"],
                    policy=row["policy"],
                    top_k=int(row["top_k"]),
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
            "Use this report to compare prompt strategies under the same GEV/event-candidate input budget. Do not treat LogPrompt as a raw-stream detector.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def pick_columns(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    values = {}
    for column in columns:
        if column in row.index:
            value = clean_value(row.get(column))
            if value is not None:
                values[column] = value
    return values


def clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        value = value.item()
    if isinstance(value, float):
        return float(value)
    if isinstance(value, int):
        return int(value)
    return value


def clean_text(value: Any, *, max_chars: int) -> str:
    cleaned = clean_value(value)
    if cleaned is None:
        return ""
    text = str(cleaned)
    if max_chars > 0 and len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def as_int(value: Any) -> int | None:
    cleaned = clean_value(value)
    if cleaned is None:
        return None
    return int(cleaned)


def numeric(value: Any, *, default: float) -> float:
    cleaned = clean_value(value)
    if cleaned is None:
        return default
    try:
        return float(cleaned)
    except (TypeError, ValueError):
        return default


def parse_int_list(raw: str) -> list[int]:
    return sorted({int(item.strip()) for item in raw.split(",") if item.strip() and int(item.strip()) > 0})


def parse_variants(raw: str) -> list[str]:
    variants = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [variant for variant in variants if variant not in PROMPT_BY_VARIANT]
    if unknown:
        raise ValueError(f"Unknown LogPrompt variant(s): {', '.join(unknown)}")
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_result_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


if __name__ == "__main__":
    main()
