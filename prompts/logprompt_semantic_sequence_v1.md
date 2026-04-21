/no_think

You are applying a LogPrompt-style review to one suspicious log event. A classical GEV/statistical pipeline has already filtered the raw stream. Your job is to combine log semantics with local sequence context and decide whether this suspicious candidate deserves analyst review.

Use only the packet evidence. Do not assume access to labels, ground truth, previous experiment results, or external documentation.

Return exactly one JSON object and no other text:

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

Review method:

1. Semantic check: inspect the message/template for failure, fatal, error, warning, corrected/uncorrected, network, filesystem, mount, socket, read, link, or hardware language.
2. Sequence check: inspect neighbor events, local switches, novelty around the event, repeated templates, and window context.
3. Score check: use scores only as supporting evidence. Do not treat a numeric score alone as proof.
4. Decision: prioritize review when semantic and sequence evidence agree, or when either one is strongly suspicious.

Scoring rules:

- `triage_score` is 0-20 for clearly benign repetition.
- `triage_score` is 21-45 for low-priority corrected/informational operational messages.
- `triage_score` is 46-70 for uncertain suspicious context.
- `triage_score` is 71-100 for fatal, failed, uncorrected, rare/novel, bursty, or sequence-disruptive messages.
- When unsure, prefer `uncertain` with a middle/high score instead of hiding cases as `likely_normal`.

Consistency rules:

- `packet_id` must equal the input packet id.
- `confidence` must be between 0 and 1.
- Use only the allowed `reason_codes`.
- If `review_decision` is `likely_normal`, `recommended_action` must be `ignore`.
- Use `inspect_window` when sequence context is important.
- Use `inspect_template_cluster` when repeated same-template behavior is the main evidence.

Writing rules:

- Keep `rationale` under 60 words.
- Keep each evidence item under 25 words.
- Do not mention labels, true positives, false positives, training labels, precision, recall, or ground truth.

Input packet:

```json
{{PACKET_JSON}}
```
