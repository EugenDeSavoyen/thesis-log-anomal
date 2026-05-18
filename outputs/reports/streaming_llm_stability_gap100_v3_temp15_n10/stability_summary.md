# Streaming LLM Stability Experiment

This repeats the same incident-level LLM detector on the fixed `gap100/close100` packet set. Cache is disabled, the prompt and decoding settings are held constant, and labels are used only after each run for offline evaluation.

## Headline

| Metric | Value |
| --- | ---: |
| repeated runs | 10 |
| packets per run | 20 |
| packet F1 mean | 0.800 |
| packet F1 95% bootstrap CI | [0.800, 0.800] |
| minimum packet recall | 1.000 |
| minimum event recall | 1.000 |
| minimum cluster recall | 1.000 |
| unanimous packets | 20 / 20 |

## Fixed Parameters

| Parameter | Value |
| --- | --- |
| packet input | `outputs/reports/streaming_incident_packets_gap100.jsonl` |
| regions | `outputs/reports/streaming_incident_regions_gap100.csv` |
| prompt | `prompts/log_triage_streaming_incident_v3.md` |
| prompt sha256 | `3fe0b60fa70aecc62ff14fc8e4ea6a235bb7f84cadaa762379bd94b6cc13aac9` |
| cache | disabled |
| LLM `provider` | `ollama` |
| LLM `endpoint` | `http://127.0.0.1:11434/api/generate` |
| LLM `model` | `qwen3:8b` |
| LLM `model_quantization` | `Q4_K_M` |
| LLM `prompt_template_id` | `log_triage_streaming_incident_v3_gap100_stability` |
| LLM `temperature` | `1.5` |
| LLM `top_p` | `0.95` |
| LLM `top_k` | `20` |
| LLM `seed` | `7` |
| LLM `num_ctx` | `8192` |
| LLM `num_predict` | `512` |
| LLM `repeat_penalty` | `1.0` |
| LLM `think_mode` | `disabled` |
| LLM `json_mode` | `True` |
| LLM `timeout_seconds` | `240` |
| packetizer `calibration_fraction` | `0.2` |
| packetizer `dense_quantile` | `0.98` |
| packetizer `diversity_quantile` | `0.98` |
| packetizer `enable_dense_channel` | `True` |
| packetizer `enable_diversity_channel` | `True` |
| packetizer `max_gap` | `100` |
| packetizer `context_before` | `10` |
| packetizer `context_after` | `10` |
| packetizer `close_after` | `100` |
| packetizer `max_representative_events` | `8` |
| packetizer `max_packets` | `200` |
| threshold `template_burst_score` | `100.0` |
| threshold `novelty_score` | `2.25` |
| threshold `markov_bigram_surprise` | `3.597766405033929` |

## Metric Stability

| Metric | Mean | Std | Min | Max | 95% bootstrap CI |
| --- | ---: | ---: | ---: | ---: | ---: |
| `valid_json_rate` | 0.970 | 0.079 | 0.750 | 1.000 | [0.920, 1.000] |
| `llm_positive_packets` | 15.000 | 0.000 | 15.000 | 15.000 | [15.000, 15.000] |
| `packet_precision` | 0.667 | 0.000 | 0.667 | 0.667 | [0.667, 0.667] |
| `packet_recall` | 1.000 | 0.000 | 1.000 | 1.000 | [1.000, 1.000] |
| `packet_f1` | 0.800 | 0.000 | 0.800 | 0.800 | [0.800, 0.800] |
| `event_recall_in_llm_positive_packets` | 1.000 | 0.000 | 1.000 | 1.000 | [1.000, 1.000] |
| `cluster_recall_in_llm_positive_packets` | 1.000 | 0.000 | 1.000 | 1.000 | [1.000, 1.000] |
| `stream_load_in_llm_positive_intervals` | 0.237 | 0.000 | 0.237 | 0.237 | [0.237, 0.237] |
| `event_density_in_llm_positive_intervals` | 0.182 | 0.000 | 0.182 | 0.182 | [0.182, 0.182] |
| `ignored_anomaly_events` | 0.000 | 0.000 | 0.000 | 0.000 | [0.000, 0.000] |
| `mean_latency_ms_uncached` | 4849.940 | 88.959 | 4691.000 | 5033.500 | [4799.045, 4903.300] |
| `p95_latency_ms_uncached` | 5202.700 | 200.371 | 5006.000 | 5674.000 | [5098.900, 5332.700] |
| `total_tokens_reported` | 86604.200 | 122.862 | 86523.000 | 86941.000 | [86551.400, 86687.000] |

## Per-Run Results

| Run | Valid JSON | LLM-positive | TP | FP | TN | FN | Precision | Recall | F1 | Event recall | Cluster recall | Stream load | Tokens |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 20 | 15 | 10 | 5 | 5 | 0 | 0.667 | 1.000 | 0.800 | 1.000 | 1.000 | 0.237 | 86,626 |
| 2 | 20 | 15 | 10 | 5 | 5 | 0 | 0.667 | 1.000 | 0.800 | 1.000 | 1.000 | 0.237 | 86,524 |
| 3 | 20 | 15 | 10 | 5 | 5 | 0 | 0.667 | 1.000 | 0.800 | 1.000 | 1.000 | 0.237 | 86,574 |
| 4 | 19 | 15 | 10 | 5 | 5 | 0 | 0.667 | 1.000 | 0.800 | 1.000 | 1.000 | 0.237 | 86,523 |
| 5 | 20 | 15 | 10 | 5 | 5 | 0 | 0.667 | 1.000 | 0.800 | 1.000 | 1.000 | 0.237 | 86,547 |
| 6 | 15 | 15 | 10 | 5 | 5 | 0 | 0.667 | 1.000 | 0.800 | 1.000 | 1.000 | 0.237 | 86,941 |
| 7 | 20 | 15 | 10 | 5 | 5 | 0 | 0.667 | 1.000 | 0.800 | 1.000 | 1.000 | 0.237 | 86,605 |
| 8 | 20 | 15 | 10 | 5 | 5 | 0 | 0.667 | 1.000 | 0.800 | 1.000 | 1.000 | 0.237 | 86,588 |
| 9 | 20 | 15 | 10 | 5 | 5 | 0 | 0.667 | 1.000 | 0.800 | 1.000 | 1.000 | 0.237 | 86,560 |
| 10 | 20 | 15 | 10 | 5 | 5 | 0 | 0.667 | 1.000 | 0.800 | 1.000 | 1.000 | 0.237 | 86,554 |

## Decision Agreement

- Unanimous inspect/ignore decision: `20 / 20` packets.
- Non-unanimous packets: `0`.
- `mean_positive_vote_share` reports the mean majority-vote share, so higher is more stable.

## Artifacts

- `outputs\reports\streaming_llm_stability_gap100_v3_temp15_n10\stability_summary.json`
- per-run JSONL/metadata/evaluation files under `outputs\reports\streaming_llm_stability_gap100_v3_temp15_n10`
