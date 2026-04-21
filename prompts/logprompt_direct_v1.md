/no_think

You are reviewing one suspicious event from a log-anomaly pipeline. The event was already selected by GEV/statistical filtering and event-level ranking. Your task is only semantic triage inside this suspicious set.

Use only the input packet. Do not assume access to labels, ground truth, previous experiment results, or external documentation.

Return exactly one JSON object and no other text:

```json
{
  "packet_id": "string",
  "review_decision": "likely_anomaly | likely_normal | uncertain",
  "severity": "low | medium | high | unknown",
  "confidence": 0.0,
  "triage_score": 0,
  "review_priority": "ignore | low | medium | high",
  "reason_codes": ["burst", "template_burst", "rare_template", "novel_template", "sequence_context", "benign_repetition", "insufficient_context"],
  "rationale": "short analyst-facing explanation",
  "suspicious_evidence": ["short evidence item"],
  "benign_evidence": ["short evidence item"],
  "recommended_action": "ignore | inspect_event | inspect_window | inspect_template_cluster",
  "needs_more_context": false
}
```

Decision rules:

- `packet_id` must equal the input packet id.
- `confidence` must be between 0 and 1.
- `triage_score` must be an integer from 0 to 100.
- Use only the allowed `reason_codes`; do not invent feature-name reason codes.
- Use `likely_anomaly` for fatal, failed, uncorrected, communication, filesystem, socket, mount, read, severed-link, rare, novel, or burst-like evidence.
- Use `likely_normal` only for clearly benign repetition with no strong suspicious semantics.
- Use `uncertain` when suspicious and benign evidence both appear plausible.
- If `review_decision` is `likely_normal`, `recommended_action` must be `ignore`.
- Repetition is not automatically benign.

Writing rules:

- Keep `rationale` under 60 words.
- Keep each evidence item under 25 words.
- Do not mention labels, true positives, false positives, training labels, precision, recall, or ground truth.

Input packet:

```json
{{PACKET_JSON}}
```
