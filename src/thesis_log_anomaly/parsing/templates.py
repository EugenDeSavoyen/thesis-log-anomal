from thesis_log_anomaly.parsing.drain import summarize_template_counts


def summarize_templates(parsed_lines: list[dict]) -> dict:
    """Build template-level summary statistics for one window or dataset."""
    template_counts = summarize_template_counts(parsed_lines)
    return {
        "count": len(parsed_lines),
        "unique_templates": len(template_counts),
        "template_counts": template_counts,
    }
