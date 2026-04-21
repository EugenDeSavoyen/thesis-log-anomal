from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


RESPONSE_SCHEMA_VERSION = "llm_triage_response_v1"
DEFAULT_ENDPOINT = "http://127.0.0.1:11434/api/generate"
INSPECT_ACTIONS = {"inspect_event", "inspect_window", "inspect_template_cluster"}
REQUIRED_RESPONSE_KEYS = {
    "packet_id",
    "review_decision",
    "severity",
    "confidence",
    "reason_codes",
    "rationale",
    "suspicious_evidence",
    "benign_evidence",
    "recommended_action",
    "needs_more_context",
}
DECISIONS = {"likely_anomaly", "likely_normal", "uncertain"}
SEVERITIES = {"low", "medium", "high", "unknown"}
REASON_CODES = {
    "burst",
    "template_burst",
    "rare_template",
    "novel_template",
    "sequence_context",
    "benign_repetition",
    "insufficient_context",
}
REASON_CODE_ALIASES = {
    "corrected_hardware_error": "benign_repetition",
    "communication_failure": "sequence_context",
    "filesystem_failure": "sequence_context",
    "fatal_failure": "sequence_context",
    "template_burst_score": "template_burst",
    "novelty": "novel_template",
    "novelty_score": "novel_template",
    "is_new_template": "novel_template",
    "unseen_in_history": "novel_template",
    "rarity": "rare_template",
    "rarity_score": "rare_template",
    "historically_rare": "rare_template",
    "local_sequence_context_score": "sequence_context",
}
ACTIONS = {"ignore", "inspect_event", "inspect_window", "inspect_template_cluster"}
REVIEW_PRIORITIES = {"ignore", "low", "medium", "high"}
FORBIDDEN_RESPONSE_TERMS = {
    "ground truth",
    "true positive",
    "false positive",
    "false negative",
    "label",
    "labels",
    "precision",
    "recall",
}


@dataclass(frozen=True)
class LlmSettings:
    provider: str = "ollama"
    endpoint: str = DEFAULT_ENDPOINT
    model: str = "qwen3:8b"
    model_quantization: str = "Q4_K_M"
    prompt_template_id: str = "log_triage_event_v2"
    temperature: float = 0.0
    top_p: float = 0.95
    top_k: int = 20
    seed: int | None = 7
    num_ctx: int = 4096
    num_predict: int = 512
    repeat_penalty: float = 1.0
    think_mode: str = "disabled"
    json_mode: bool = True
    timeout_seconds: int = 120

    def ollama_options(self) -> dict[str, Any]:
        options: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.seed is not None:
            options["seed"] = self.seed
        return options

    def as_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "endpoint": self.endpoint,
            "model": self.model,
            "model_quantization": self.model_quantization,
            "prompt_template_id": self.prompt_template_id,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "repeat_penalty": self.repeat_penalty,
            "think_mode": self.think_mode,
            "json_mode": self.json_mode,
            "timeout_seconds": self.timeout_seconds,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local no-thinking LLM triage over JSONL review packets."
    )
    parser.add_argument(
        "--input-jsonl",
        default="outputs/reports/llm_review_packets_bgl_multiblock.jsonl",
    )
    parser.add_argument(
        "--prompt-template",
        default="prompts/log_triage_event_v2.md",
    )
    parser.add_argument(
        "--prompt-template-id",
        default=None,
        help="Prompt id stored in run metadata. Defaults to the prompt template file stem.",
    )
    parser.add_argument(
        "--output-jsonl",
        default="outputs/reports/llm_triage_bgl_multiblock.jsonl",
    )
    parser.add_argument(
        "--output-metadata",
        default="outputs/reports/llm_triage_bgl_multiblock_run.json",
    )
    parser.add_argument("--cache-dir", default="outputs/reports/llm_triage_cache")
    parser.add_argument("--provider", default="ollama", choices=["ollama"])
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--model", default="qwen3:8b")
    parser.add_argument("--model-quantization", default="Q4_K_M")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--repeat-penalty", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    parser.add_argument("--max-packets", type=int, default=None)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Render and validate requests without calling the model.",
    )
    args = parser.parse_args()

    prompt_template_path = Path(args.prompt_template)
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    prompt_sha256 = _sha256_text(prompt_template)
    settings = LlmSettings(
        provider=args.provider,
        endpoint=args.endpoint,
        model=args.model,
        model_quantization=args.model_quantization,
        prompt_template_id=args.prompt_template_id or prompt_template_path.stem,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
        num_ctx=args.num_ctx,
        num_predict=args.num_predict,
        repeat_penalty=args.repeat_penalty,
        timeout_seconds=args.timeout_seconds,
    )
    packets = read_jsonl(Path(args.input_jsonl))
    selected_packets = packets[args.start_offset :]
    if args.max_packets is not None:
        selected_packets = selected_packets[: args.max_packets]

    output_jsonl = Path(args.output_jsonl)
    output_metadata = Path(args.output_metadata)
    cache_dir = Path(args.cache_dir)
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    if not args.no_cache:
        cache_dir.mkdir(parents=True, exist_ok=True)

    started_at = utc_now()
    results = run_triage_packets(
        selected_packets,
        prompt_template=prompt_template,
        prompt_sha256=prompt_sha256,
        settings=settings,
        cache_dir=None if args.no_cache else cache_dir,
        retries=args.retries,
        dry_run=args.dry_run,
    )
    write_jsonl(results, output_jsonl)
    metadata = build_run_metadata(
        results,
        input_jsonl=args.input_jsonl,
        output_jsonl=str(output_jsonl),
        prompt_template=str(prompt_template_path),
        prompt_sha256=prompt_sha256,
        settings=settings,
        started_at=started_at,
        dry_run=args.dry_run,
        cache_enabled=not args.no_cache,
        start_offset=args.start_offset,
        max_packets=args.max_packets,
    )
    output_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


def run_triage_packets(
    packets: list[dict[str, Any]],
    *,
    prompt_template: str,
    prompt_sha256: str,
    settings: LlmSettings,
    cache_dir: Path | None,
    retries: int,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    results = []
    for packet in packets:
        prompt = render_prompt(prompt_template, packet)
        request_payload = build_ollama_payload(prompt, settings)
        request_sha256 = _sha256_json(
            {
                "packet": packet,
                "prompt_sha256": prompt_sha256,
                "request_payload": request_payload,
                "settings": settings.as_metadata(),
            }
        )
        started_at = utc_now()
        if dry_run:
            result = _dry_run_result(packet, request_sha256, started_at)
        else:
            result = run_one_packet(
                packet,
                request_payload=request_payload,
                request_sha256=request_sha256,
                settings=settings,
                cache_dir=cache_dir,
                retries=retries,
                started_at=started_at,
            )
        results.append(result)
    return results


def run_one_packet(
    packet: dict[str, Any],
    *,
    request_payload: dict[str, Any],
    request_sha256: str,
    settings: LlmSettings,
    cache_dir: Path | None,
    retries: int,
    started_at: str,
) -> dict[str, Any]:
    cache_path = cache_dir / f"{request_sha256}.json" if cache_dir is not None else None
    if cache_path is not None and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return {**cached, "cache_hit": True}

    response_payload: dict[str, Any] | None = None
    raw_response = ""
    parsed_response: dict[str, Any] | None = None
    validation_errors: list[str] = []
    call_error = None
    latency_ms = 0
    retry_count = 0

    for attempt in range(retries + 1):
        retry_count = attempt
        call_started = time.perf_counter()
        try:
            response_payload = call_ollama(
                settings.endpoint,
                request_payload,
                timeout_seconds=settings.timeout_seconds,
            )
            latency_ms = int((time.perf_counter() - call_started) * 1000)
            raw_response = str(response_payload.get("response") or "")
            parsed_response, validation_errors = parse_and_validate_response(raw_response, packet)
            call_error = None
            if not validation_errors:
                break
        except Exception as exc:  # noqa: BLE001 - surfaced in output metadata.
            latency_ms = int((time.perf_counter() - call_started) * 1000)
            call_error = f"{type(exc).__name__}: {exc}"
            validation_errors = [call_error]

    result = build_result_record(
        packet,
        request_sha256=request_sha256,
        started_at=started_at,
        latency_ms=latency_ms,
        raw_response=raw_response,
        parsed_response=parsed_response,
        validation_errors=validation_errors,
        response_payload=response_payload or {},
        retry_count=retry_count,
        cache_hit=False,
        call_error=call_error,
    )
    if cache_path is not None:
        cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def render_prompt(prompt_template: str, packet: dict[str, Any]) -> str:
    packet_json = json.dumps(packet, ensure_ascii=False, sort_keys=True, indent=2)
    return prompt_template.replace("{{PACKET_JSON}}", packet_json)


def build_ollama_payload(prompt: str, settings: LlmSettings) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": settings.model,
        "prompt": prompt,
        "stream": False,
        "options": settings.ollama_options(),
    }
    if settings.json_mode:
        payload["format"] = "json"
    return payload


def call_ollama(endpoint: str, payload: dict[str, Any], *, timeout_seconds: int) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        endpoint,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Ollama connection failed: {exc.reason}") from exc


def parse_and_validate_response(
    raw_response: str,
    packet: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    try:
        parsed = extract_json_object(raw_response)
    except ValueError as exc:
        return None, [str(exc)]
    normalize_reason_codes(parsed)
    normalize_response_consistency(parsed)
    errors = validate_response(parsed, expected_packet_id=str(packet.get("packet_id")))
    return parsed, errors


def normalize_reason_codes(response: dict[str, Any]) -> None:
    reason_codes = response.get("reason_codes")
    if not isinstance(reason_codes, list):
        return
    normalized = []
    for code in reason_codes:
        normalized_code = REASON_CODE_ALIASES.get(str(code), code)
        if normalized_code not in normalized:
            normalized.append(normalized_code)
    response["reason_codes"] = normalized


def normalize_response_consistency(response: dict[str, Any]) -> None:
    if response.get("review_decision") != "likely_normal":
        return
    if response.get("needs_more_context") is True:
        return
    if response.get("recommended_action") in INSPECT_ACTIONS:
        response["recommended_action"] = "ignore"
        if response.get("review_priority") in {None, "medium", "high"}:
            response["review_priority"] = "low"
    triage_score = response.get("triage_score")
    low_score = triage_score is None
    if isinstance(triage_score, (int, float)):
        low_score = float(triage_score) <= 45
    low_priority = response.get("review_priority") in {None, "ignore", "low"}
    if low_score and low_priority and response.get("recommended_action") in INSPECT_ACTIONS:
        response["recommended_action"] = "ignore"


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("Empty model response")
    try:
        loaded = json.loads(stripped)
        if isinstance(loaded, dict):
            return loaded
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        raise ValueError("No JSON object found in model response")
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(stripped)):
        char = stripped[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : index + 1]
                try:
                    loaded = json.loads(candidate)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSON object: {exc}") from exc
                if not isinstance(loaded, dict):
                    raise ValueError("Model response JSON is not an object")
                return loaded
    raise ValueError("Unterminated JSON object in model response")


def validate_response(response: dict[str, Any], *, expected_packet_id: str) -> list[str]:
    errors = []
    missing = sorted(REQUIRED_RESPONSE_KEYS - set(response))
    if missing:
        errors.append(f"Missing required keys: {', '.join(missing)}")
    extra_decision = response.get("review_decision")
    if extra_decision is not None and extra_decision not in DECISIONS:
        errors.append(f"Invalid review_decision: {extra_decision}")
    severity = response.get("severity")
    if severity is not None and severity not in SEVERITIES:
        errors.append(f"Invalid severity: {severity}")
    action = response.get("recommended_action")
    if action is not None and action not in ACTIONS:
        errors.append(f"Invalid recommended_action: {action}")
    if response.get("packet_id") != expected_packet_id:
        errors.append("packet_id does not match input packet")
    confidence = response.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0 <= float(confidence) <= 1:
        errors.append("confidence must be a number between 0 and 1")
    reason_codes = response.get("reason_codes")
    if not isinstance(reason_codes, list):
        errors.append("reason_codes must be a list")
    else:
        unknown = sorted({str(item) for item in reason_codes if item not in REASON_CODES})
        if unknown:
            errors.append(f"Unknown reason_codes: {', '.join(unknown)}")
    for key in ["suspicious_evidence", "benign_evidence"]:
        if key in response and not isinstance(response[key], list):
            errors.append(f"{key} must be a list")
    if "needs_more_context" in response and not isinstance(response["needs_more_context"], bool):
        errors.append("needs_more_context must be a boolean")
    if (
        response.get("review_decision") == "likely_normal"
        and response.get("recommended_action") in INSPECT_ACTIONS
        and response.get("needs_more_context") is not True
    ):
        errors.append("Inconsistent response: likely_normal should recommend ignore unless more context is needed")
    if response.get("recommended_action") in INSPECT_ACTIONS and response.get("review_decision") == "likely_normal":
        errors.append("Inconsistent response: inspect actions require likely_anomaly or uncertain")
    triage_score = response.get("triage_score")
    if triage_score is not None and (
        not isinstance(triage_score, (int, float)) or not 0 <= float(triage_score) <= 100
    ):
        errors.append("triage_score must be a number between 0 and 100")
    review_priority = response.get("review_priority")
    if review_priority is not None and review_priority not in REVIEW_PRIORITIES:
        errors.append(f"Invalid review_priority: {review_priority}")
    rationale = response.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        errors.append("rationale must be a non-empty string")
    elif len(rationale.split()) > 80:
        errors.append("rationale is too long")
    response_text = json.dumps(response, ensure_ascii=False).lower()
    forbidden = sorted(term for term in FORBIDDEN_RESPONSE_TERMS if term in response_text)
    if forbidden:
        errors.append(f"Forbidden evaluation terms in response: {', '.join(forbidden)}")
    return errors


def build_result_record(
    packet: dict[str, Any],
    *,
    request_sha256: str,
    started_at: str,
    latency_ms: int,
    raw_response: str,
    parsed_response: dict[str, Any] | None,
    validation_errors: list[str],
    response_payload: dict[str, Any],
    retry_count: int,
    cache_hit: bool,
    call_error: str | None,
) -> dict[str, Any]:
    finished_at = utc_now()
    response_sha256 = _sha256_text(raw_response) if raw_response else None
    return {
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "packet_id": packet.get("packet_id"),
        "event_id": packet.get("event_id"),
        "selection_rank": packet.get("selection_rank"),
        "request_sha256": request_sha256,
        "response_sha256": response_sha256,
        "cache_hit": cache_hit,
        "started_at": started_at,
        "finished_at": finished_at,
        "latency_ms": latency_ms,
        "prompt_tokens": response_payload.get("prompt_eval_count"),
        "completion_tokens": response_payload.get("eval_count"),
        "total_tokens": _sum_optional_ints(
            response_payload.get("prompt_eval_count"),
            response_payload.get("eval_count"),
        ),
        "ollama_total_duration_ns": response_payload.get("total_duration"),
        "ollama_load_duration_ns": response_payload.get("load_duration"),
        "valid_json": parsed_response is not None and not validation_errors,
        "validation_errors": validation_errors,
        "retry_count": retry_count,
        "call_error": call_error,
        "review_decision": None if parsed_response is None else parsed_response.get("review_decision"),
        "confidence": None if parsed_response is None else parsed_response.get("confidence"),
        "triage_score": None if parsed_response is None else parsed_response.get("triage_score"),
        "review_priority": None if parsed_response is None else parsed_response.get("review_priority"),
        "recommended_action": None if parsed_response is None else parsed_response.get("recommended_action"),
        "parsed_response": parsed_response,
        "raw_response": raw_response,
    }


def build_run_metadata(
    results: list[dict[str, Any]],
    *,
    input_jsonl: str,
    output_jsonl: str,
    prompt_template: str,
    prompt_sha256: str,
    settings: LlmSettings,
    started_at: str,
    dry_run: bool,
    cache_enabled: bool,
    start_offset: int,
    max_packets: int | None,
) -> dict[str, Any]:
    finished_at = utc_now()
    valid = sum(1 for row in results if row.get("valid_json"))
    cache_hits = sum(1 for row in results if row.get("cache_hit"))
    latencies = [int(row.get("latency_ms") or 0) for row in results if not row.get("cache_hit")]
    total_tokens = sum(int(row.get("total_tokens") or 0) for row in results)
    return {
        "run_id": f"llm_triage_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "response_schema_version": RESPONSE_SCHEMA_VERSION,
        "created_at": started_at,
        "finished_at": finished_at,
        "input_path": input_jsonl,
        "output_path": output_jsonl,
        "prompt_template": prompt_template,
        "prompt_template_sha256": prompt_sha256,
        "settings": settings.as_metadata(),
        "dry_run": dry_run,
        "cache_enabled": cache_enabled,
        "start_offset": start_offset,
        "max_packets": max_packets,
        "num_packets": len(results),
        "valid_json": valid,
        "invalid_json": len(results) - valid,
        "valid_json_rate": None if not results else valid / len(results),
        "cache_hits": cache_hits,
        "mean_latency_ms_uncached": None if not latencies else sum(latencies) / len(latencies),
        "p95_latency_ms_uncached": percentile(latencies, 0.95),
        "total_tokens_reported": total_tokens,
        "decision_counts": _value_counts(row.get("review_decision") for row in results),
        "action_counts": _value_counts(row.get("recommended_action") for row in results),
    }


def _dry_run_result(packet: dict[str, Any], request_sha256: str, started_at: str) -> dict[str, Any]:
    return build_result_record(
        packet,
        request_sha256=request_sha256,
        started_at=started_at,
        latency_ms=0,
        raw_response="",
        parsed_response=None,
        validation_errors=["dry_run: model was not called"],
        response_payload={},
        retry_count=0,
        cache_hit=False,
        call_error=None,
    )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return float(ordered[index])


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sha256_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sum_optional_ints(*values: Any) -> int | None:
    present = [int(value) for value in values if value is not None]
    if not present:
        return None
    return sum(present)


def _value_counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


if __name__ == "__main__":
    main()
