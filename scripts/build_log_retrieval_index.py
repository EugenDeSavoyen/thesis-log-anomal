from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INDEX_SCHEMA_VERSION = "log_retrieval_index_v1"
DEFAULT_MODEL_PATH = "models/Qwen3-Embedding-0.6B"
META_COLUMNS = [
    "event_id",
    "event_order",
    "block_id",
    "template",
    "message",
    "template_burst_score",
    "template_count_deviation_score",
    "novelty_score",
    "rarity_score",
    "historical_count_log",
    "local_sequence_context_score",
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a local retrieval-lite embedding index for log examples."
    )
    parser.add_argument(
        "--input-csv",
        default="data/processed/event_level_candidates_bgl_multiblock.csv",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--text-mode", default="template_message", choices=["template", "message", "template_message"])
    parser.add_argument(
        "--output-meta",
        default="outputs/reports/log_retrieval_index_meta.csv",
    )
    parser.add_argument(
        "--output-embeddings",
        default="outputs/models/log_retrieval_index_embeddings.npy",
    )
    parser.add_argument(
        "--output-report",
        default="outputs/reports/log_retrieval_index.json",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    corpus = build_retrieval_corpus(df, max_rows=args.max_rows, text_mode=args.text_mode)
    texts = corpus["retrieval_text"].tolist()
    embeddings = encode_texts(texts, model_path=args.model_path, batch_size=args.batch_size)

    meta_path = Path(args.output_meta)
    embeddings_path = Path(args.output_embeddings)
    report_path = Path(args.output_report)
    for path in [meta_path, embeddings_path, report_path]:
        path.parent.mkdir(parents=True, exist_ok=True)

    corpus.to_csv(meta_path, index=False)
    np.save(embeddings_path, embeddings.astype(np.float32))
    report = {
        "index_schema_version": INDEX_SCHEMA_VERSION,
        "input_csv": args.input_csv,
        "model_path": args.model_path,
        "text_mode": args.text_mode,
        "num_rows": int(len(corpus)),
        "embedding_shape": list(embeddings.shape),
        "output_meta": str(meta_path),
        "output_embeddings": str(embeddings_path),
        "meta_sha256": _sha256_dataframe(corpus),
        "embedding_sha256": _sha256_array(embeddings),
        "label_policy": "labels are excluded from retrieval metadata and embeddings",
        "notes": [
            "This is retrieval-lite: metadata CSV plus normalized embedding matrix.",
            "Use scripts/augment_cluster_packets_with_retrieval.py to add retrieved context to LLM packets.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def build_retrieval_corpus(
    df: pd.DataFrame,
    *,
    max_rows: int | None = None,
    text_mode: str = "template_message",
) -> pd.DataFrame:
    if max_rows is not None:
        df = df.head(max_rows).copy()
    columns = [column for column in META_COLUMNS if column in df.columns]
    corpus = df[columns].copy()
    if "event_id" in corpus:
        corpus = corpus.drop_duplicates(subset=["event_id"]).copy()
    corpus["retrieval_text"] = [
        build_retrieval_text(row, text_mode=text_mode)
        for _, row in corpus.iterrows()
    ]
    label_like = [column for column in corpus.columns if _is_label_like(column)]
    if label_like:
        raise ValueError(f"Label-like columns would leak into retrieval index: {label_like}")
    return corpus.reset_index(drop=True)


def build_retrieval_text(row: pd.Series, *, text_mode: str) -> str:
    template = _clean_text(row.get("template"))
    message = _clean_text(row.get("message"))
    if text_mode == "template":
        return f"Template: {template}"
    if text_mode == "message":
        return f"Message: {message}"
    return f"Template: {template}\nMessage: {message}"


def encode_texts(texts: list[str], *, model_path: str, batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = _load_sentence_transformer(model_path)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def _load_sentence_transformer(model_path: str):
    from sentence_transformers import SentenceTransformer

    try:
        return SentenceTransformer(model_path, local_files_only=True)
    except TypeError:
        return SentenceTransformer(model_path)


def _clean_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value)


def _is_label_like(column: str) -> bool:
    lowered = column.lower()
    return lowered in {"label", "labels", "is_anomaly", "ground_truth"} or lowered.startswith("label_")


def _sha256_dataframe(df: pd.DataFrame) -> str:
    raw = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


if __name__ == "__main__":
    main()
