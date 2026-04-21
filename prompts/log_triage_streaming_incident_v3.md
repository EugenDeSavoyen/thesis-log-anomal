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

Hard output constraints:

- `packet_id` must equal the input packet id.
- `reason_codes` must contain only values from the schema list. Do not put semantic categories in `reason_codes`.
- If `review_decision` is `likely_normal`, `triage_score` must be 0-45, `recommended_action` must be `ignore`, and `review_priority` must be `ignore` or `low`.
- If `review_decision` is `likely_anomaly`, `triage_score` must be 61-100 and `recommended_action` should usually be `inspect_template_cluster`.
- If `review_decision` is `uncertain`, `triage_score` must be 46-60 and `needs_more_context` should usually be true.

Decision policy:

- Do not use `uncertain` as the default. Choose `likely_normal` or `likely_anomaly` whenever the packet has a clear direction.
- Use `likely_normal` when the packet is mostly repeated informational or corrected hardware messages and there is no fatal, failed, severed, missing, mount, filesystem, socket, read, panic, uncorrected, or similar severe symptom.
- Use `likely_anomaly` when any representative event contains fatal, failed, failure, panic, severed, missing program image, mount failed, filesystem failure, socket/control-stream read failure, uncorrected error, or similarly operationally severe wording.
- Use `likely_anomaly` for high-density incident bursts with many seed events and high template-burst evidence, unless the text clearly says errors were detected and corrected.
- Use `likely_normal` for corrected hardware-noise clusters when messages explicitly say detected and corrected, total corrected, CE/EDRAM/DDR corrected, and no severe failure wording appears.

Reason-code policy:

- Use `benign_repetition` for corrected repeated INFO events.
- Use `sequence_context` for Markov/sequence surprise or communication-flow symptoms.
- Use `template_burst` for repeated same-template incident evidence.
- Use `novel_template` or `rare_template` when novelty/rarity is a main reason.
- Use `insufficient_context` only with `uncertain`.
- Do not invent reason codes such as corrected_hardware_error, fatal_failure, communication_failure, or filesystem_failure. Those are semantic categories, not reason codes.

Writing rules:

- Keep `rationale` under 60 words.
- Keep each evidence item under 25 words.
- Do not mention labels, true positives, false positives, training labels, precision, recall, or ground truth.

Input streaming incident packet:

```json
{{PACKET_JSON}}
```
