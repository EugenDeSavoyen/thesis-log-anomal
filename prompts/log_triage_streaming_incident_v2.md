/no_think

You are reviewing one incident packet from a production-like streaming log-anomaly pipeline.

A cheap streaming detector has already opened and closed this incident buffer. Your job is not to replace the detector. Your job is to make a decisive analyst triage decision from the provided evidence.

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

Decision policy:

- Do not use `uncertain` as the default. Use it only when the evidence is genuinely balanced or the packet lacks enough context.
- Use `likely_normal` when the packet is mostly repeated informational or corrected hardware messages and there is no fatal, failed, severed, missing, mount, filesystem, socket, read, panic, or uncorrected symptom.
- Use `likely_anomaly` when any representative event contains fatal, failed, failure, panic, severed, missing program image, mount failed, filesystem failure, socket/control-stream read failure, or similarly operationally severe wording.
- Use `likely_anomaly` for high-density incident bursts with many seed events and high template burst evidence, even if some individual messages are repetitive.
- Use `likely_normal` for corrected hardware-noise clusters when messages explicitly say detected and corrected, total corrected, CE/EDRAM/DDR corrected, and no severe failure wording appears.

Scoring policy:

- `triage_score` is an integer from 0 to 100.
- 0-20: ignore; common benign repetition.
- 21-45: likely normal or low-priority corrected hardware noise.
- 46-60: uncertain; needs more context.
- 61-80: likely anomaly; review-worthy suspicious incident.
- 81-100: high-priority likely anomaly with fatal/failure/panic/filesystem/socket/mount/severed symptoms.

Action policy:

- If `review_decision` is `likely_normal`, set `recommended_action` to `ignore`, and `review_priority` to `ignore` or `low`.
- If `review_decision` is `likely_anomaly`, use `inspect_template_cluster` unless one single event is clearly sufficient.
- If `review_decision` is `uncertain`, set `needs_more_context` to true unless the packet is already review-worthy.

Reason-code policy:

- Use only the allowed `reason_codes`.
- Use `benign_repetition` for corrected repeated INFO events.
- Use `sequence_context` for Markov/sequence surprise.
- Use `template_burst` for repeated same-template incident evidence.
- Use `novel_template` or `rare_template` when novelty/rarity is a main reason.
- Do not invent feature-name reason codes.

Writing rules:

- Keep `rationale` under 60 words.
- Keep each evidence item under 25 words.
- Do not mention labels, true positives, false positives, training labels, precision, recall, or ground truth.

Input streaming incident packet:

```json
{{PACKET_JSON}}
```
