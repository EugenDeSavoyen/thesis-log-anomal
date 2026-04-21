def compute_template_counts(window_templates: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in window_templates:
        template = item.get("template") or "<unknown>"
        counts[template] = counts.get(template, 0) + 1
    return counts
