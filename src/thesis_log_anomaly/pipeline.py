from thesis_log_anomaly.config import load_config
from thesis_log_anomaly.stages.evaluation import evaluate_candidates
from thesis_log_anomaly.stages.evt import score_windows_with_evt
from thesis_log_anomaly.stages.llm_review import review_with_llm
from thesis_log_anomaly.stages.load import load_logs
from thesis_log_anomaly.stages.pre_model import prepare_pre_model_artifacts
from thesis_log_anomaly.stages.template import parse_templates
from thesis_log_anomaly.stages.window import build_windows


def run_pipeline(config_path: str) -> dict:
    """Orchestrate the thesis pipeline stage by stage.

    The current implementation is a thin scaffold so we can fill each stage in
    with real dataset and model logic incrementally.
    """
    config = load_config(config_path)
    logs = load_logs(config)
    parsed_stream = parse_templates(logs, config)
    windows = build_windows(parsed_stream, config)
    windows_after_gev = score_windows_with_evt(windows, config)
    review_with_llm(windows_after_gev, config)
    pre_model_report = prepare_pre_model_artifacts(
        {
            "parsed_stream": parsed_stream,
            "windows_after_gev": windows_after_gev,
        },
        config,
    )
    return evaluate_candidates(
        {
            "parsed_stream": parsed_stream,
            "windows_after_gev": windows_after_gev,
            "windows_after_pot": windows_after_gev,
            "pre_model_report": pre_model_report,
        },
        config,
    )
