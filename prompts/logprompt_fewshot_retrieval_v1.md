/no_think

You are applying a LogPrompt-style few-shot review to one suspicious log event. A GEV/statistical pipeline has already selected this candidate. Your task is to compare the candidate with unlabeled retrieval-lite examples and decide whether the candidate deserves analyst review.

Use only the packet evidence. Retrieved examples are context only; they are not guaranteed normal or anomalous. Do not assume access to labels, ground truth, previous experiment results, or external documentation.

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

1. Candidate semantics: identify whether the current event message/template describes failure, communication problems, filesystem problems, corrected hardware events, or routine operational noise.
2. Retrieval contrast: compare the candidate with same-template, score-neighbor, and lower-score examples. Use them to judge whether the candidate looks routine or unusually severe.
3. Sequence context: consider nearby events and local context scores.
4. Final triage: choose the smallest review action that still protects recall inside the suspicious candidate set.

Important cautions:

- Retrieved examples are unlabeled and may include anomalies.
- Similarity to another event is not proof of normality.
- Repeated fatal/failure messages are still review-worthy.
- Corrected informational hardware messages may be low priority when they look explicitly corrected and routine.

Consistency rules:

- `packet_id` must equal the input packet id.
- `confidence` must be between 0 and 1.
- `triage_score` must be an integer from 0 to 100.
- Use only the allowed `reason_codes`.
- If `review_decision` is `likely_normal`, `recommended_action` must be `ignore`.
- Use `inspect_window` when neighbor events change the interpretation.
- Use `inspect_template_cluster` when the same-template examples are central.

Writing rules:

- Keep `rationale` under 60 words.
- Keep each evidence item under 25 words.
- Do not mention labels, true positives, false positives, training labels, precision, recall, or ground truth.

Input packet:

```json
{{PACKET_JSON}}
```
