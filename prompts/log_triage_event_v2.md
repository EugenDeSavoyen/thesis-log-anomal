/no_think

You are reviewing one event from a log-anomaly pipeline. A statistical detector has already selected this event for possible review. Your job is to help an analyst decide whether it looks like a likely anomaly, likely normal event, or uncertain case.

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
- Use only the allowed `reason_codes`; do not invent feature-name reason codes such as `novelty_score`, `rarity_score`, or `unseen_in_history`.
- Use `likely_anomaly` when the event has failure/error/fatal semantics, high burst evidence, rare-template evidence, novelty, or suspicious sequence context.
- Use `likely_normal` only when the event is common or repetitive and there is no strong suspicious evidence.
- Use `uncertain` when benign repetition and suspicious evidence both appear plausible.
- Repetition is not automatically benign. A repeated fatal, failure, socket, mount, read, severed-link, or corrected-error pattern can still be review-worthy.

Action consistency rules:

- If `review_decision` is `likely_normal`, `recommended_action` should be `ignore` unless `needs_more_context` is true.
- If `recommended_action` is `inspect_event`, `inspect_window`, or `inspect_template_cluster`, then `review_decision` should be `likely_anomaly` or `uncertain`.
- Use `inspect_template_cluster` when many same-template events should be reviewed together.
- Use `inspect_window` when neighbor events or local sequence context matter.
- Use `inspect_event` when the single event is enough evidence.

Writing rules:

- Keep `rationale` under 60 words.
- Keep each evidence item under 25 words.
- Do not mention labels, true positives, false positives, training labels, precision, recall, or ground truth.

Input packet:

```json
{{PACKET_JSON}}
```
