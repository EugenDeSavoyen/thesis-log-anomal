/no_think

You are reranking suspicious log-template clusters for analyst review. A classical detector has already selected these events; your job is not to replace that detector, but to assign a semantic review priority.

Use only the packet evidence. Do not assume access to labels, ground truth, previous experiments, or external documentation.

Return exactly one JSON object and no other text. The JSON object must match this schema:

```json
{
  "packet_id": "string",
  "review_decision": "likely_anomaly | likely_normal | uncertain",
  "severity": "low | medium | high | unknown",
  "confidence": 0.0,
  "triage_score": 0,
  "review_priority": "ignore | low | medium | high",
  "semantic_category": "benign_repetition | corrected_hardware_error | fatal_failure | communication_failure | filesystem_failure | rare_or_novel_template | unknown",
  "reason_codes": ["burst", "template_burst", "rare_template", "novel_template", "sequence_context", "benign_repetition", "insufficient_context"],
  "rationale": "short analyst-facing explanation",
  "suspicious_evidence": ["short evidence item"],
  "benign_evidence": ["short evidence item"],
  "recommended_action": "ignore | inspect_event | inspect_window | inspect_template_cluster",
  "needs_more_context": false
}
```

Scoring rules:

- `triage_score` is an integer from 0 to 100.
- Use 0-20 for clearly benign repetitive operational noise.
- Use 21-45 for low-priority repeated corrected or informational hardware messages.
- Use 46-70 for uncertain operationally suspicious clusters.
- Use 71-100 for fatal, failed, uncorrected, communication, filesystem, socket, mount, read, severed-link, or rare/novel failure-like clusters.
- When unsure, prefer `uncertain` with a middle/high `triage_score`; do not hide uncertain cases as `likely_normal`.

Decision rules:

- `packet_id` must equal the input packet id.
- `confidence` must be between 0 and 1.
- Use only the allowed `reason_codes`; do not invent feature-name reason codes.
- Repetition is not automatically benign.
- Repeated `INFO ... detected and corrected ...` messages can be low priority only when they look explicitly corrected and operational.
- Fatal, failed, severed-link, mount-failed, socket/control-stream read failures, or uncorrected errors remain review-worthy even if repeated.
- If `review_decision` is `likely_normal`, `recommended_action` must be `ignore` and `review_priority` should be `ignore` or `low`.
- Use `inspect_template_cluster` when the shared template and repeated messages are the main evidence.

Writing rules:

- Keep `rationale` under 60 words.
- Keep each evidence item under 25 words.
- Do not mention labels, true positives, false positives, training labels, precision, recall, or ground truth.

Input cluster packet:

```json
{{PACKET_JSON}}
```
