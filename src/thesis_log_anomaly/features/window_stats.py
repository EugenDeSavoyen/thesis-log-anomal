def compute_window_statistics(window_templates: list[dict]) -> dict:
    """Compute numeric window statistics for EVT input."""
    return {
        "window_size": len(window_templates),
        "unique_templates": len({item.get("template") for item in window_templates}),
    }
