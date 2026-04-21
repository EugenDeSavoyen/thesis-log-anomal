/no_think

You are reviewing a cluster of events that share the same parsed log template. A statistical detector and event-level ranker selected these events for possible analyst review. Your job is to decide whether the cluster should be treated as a likely anomaly, likely normal repetition, or uncertain case.

Use only the packet evidence. Do not assume access to labels, ground truth, previous experiments, or external documentation.

Return exactly one JSON object and no other text. The JSON object must match this schema:

```json
{
  "packet_id": "string",
  "review_decision": "likely_anomaly | likely_normal | uncertain",
  "severity": "low | medium | high | unknown",
  "confidence": 0.0,
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
- Use only the allowed `reason_codes`; do not invent feature-name reason codes.
- Use `likely_anomaly` when the cluster has failure/error/fatal semantics, unusually high burst evidence, rare-template evidence, novelty, or suspicious sequence context.
- Use `likely_normal` only when the cluster looks like common benign repetition with weak suspicious evidence.
- Use `uncertain` when the template looks operationally suspicious but the packet lacks enough contrastive context.
- Repetition is not automatically benign. Repeated fatal, failure, socket, mount, read, severed-link, or corrected-error patterns can be more suspicious as a cluster.

Action rules:

- Prefer `inspect_template_cluster` when the shared template and repeated messages are the main evidence.
- Use `inspect_window` when surrounding sequence context is the main evidence.
- Use `ignore` only for likely-normal clusters.
- If `review_decision` is `likely_normal`, `recommended_action` should be `ignore` unless `needs_more_context` is true.

Writing rules:

- Keep `rationale` under 60 words.
- Keep each evidence item under 25 words.
- Do not mention labels, true positives, false positives, training labels, precision, recall, or ground truth.

Input cluster packet:

```json
{{PACKET_JSON}}
```
