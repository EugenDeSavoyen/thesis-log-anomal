def build_triage_prompt(window_summary: dict) -> str:
    return (
        "Analyze the following suspicious log window and explain whether it looks "
        f"anomalous: {window_summary}"
    )
