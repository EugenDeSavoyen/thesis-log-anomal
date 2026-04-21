/no_think

You are reviewing one incident packet from a production-like streaming log-anomaly pipeline.

A cheap streaming detector has already opened and closed this incident buffer. Your job is not to replace the detector. Your job is to decide whether the packet should be shown to an analyst, and to give a concise evidence-grounded explanation.

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
- Use 21-45 for low-priority corrected or informational hardware events.
- Use 46-70 for uncertain but operationally suspicious regions.
- Use 71-100 for fatal, failed, uncorrected, communication, filesystem, socket, mount, read, severed-link, rare, or novel failure-like regions.

Decision rules:

- `packet_id` must equal the input packet id.
- `confidence` must be between 0 and 1.
- Use only the allowed `reason_codes`; do not invent feature-name reason codes.
- The packet is an incident region, so prefer `inspect_template_cluster` when several events or repeated templates provide the evidence.
- Repetition is not automatically benign.
- Repeated `INFO ... detected and corrected ...` messages may be low priority only when they look explicitly corrected and operational.
- Fatal, failed, severed-link, mount-failed, socket/control-stream read failures, or uncorrected errors remain review-worthy even if repeated.
- If `review_decision` is `likely_normal`, `recommended_action` must be `ignore` and `review_priority` should be `ignore` or `low`.
- When unsure, use `uncertain`, set `needs_more_context` if needed, and explain the ambiguity.

Writing rules:

- Keep `rationale` under 60 words.
- Keep each evidence item under 25 words.
- Do not mention labels, true positives, false positives, training labels, precision, recall, or ground truth.

Input streaming incident packet:

```json
{{PACKET_JSON}}
```
