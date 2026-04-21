/no_think

You are reviewing a cluster of events that share the same parsed log template. A statistical detector selected this cluster for possible analyst review. Retrieved examples are provided only as unlabeled contrastive context; they are not ground truth.

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
- Use only the allowed `reason_codes`.
- Treat retrieved lower-score examples as contrast, not proof of normality.
- If the suspicious cluster closely resembles lower-score or common same-template examples, reduce confidence and consider `likely_normal` or `uncertain`.
- Repeated `INFO ... detected and corrected ...` hardware messages can be benign operational bursts when retrieved context shows similar lower-score patterns.
- Fatal, failed, severed-link, mount-failed, socket/control-stream read failures, or uncorrected errors remain review-worthy even if repeated.
- If `review_decision` is `likely_normal`, `recommended_action` must be `ignore`.
- Use `inspect_template_cluster` only when `review_decision` is `likely_anomaly` or `uncertain`.
- Prefer `inspect_template_cluster` for cluster-level suspicious patterns.

Writing rules:

- Keep `rationale` under 60 words.
- Keep each evidence item under 25 words.
- Do not mention labels, true positives, false positives, training labels, precision, recall, or ground truth.

Input cluster packet:

```json
{{PACKET_JSON}}
```
