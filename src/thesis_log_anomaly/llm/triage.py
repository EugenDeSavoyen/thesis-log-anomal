from thesis_log_anomaly.llm.prompts import build_triage_prompt


def prepare_llm_review_payload(candidates: list[dict]) -> list[dict]:
    return [{"prompt": build_triage_prompt(candidate), "candidate": candidate} for candidate in candidates]
