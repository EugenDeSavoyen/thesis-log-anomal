import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.select_template_burst_review_sample import _infer_totals, _threshold_sweep, _topn_sweep
from scripts.run_classical_triage_baseline import build_classical_triage_summary, rank_events, _markov_surprise
from scripts.build_llm_review_packets import assert_no_label_leakage, build_review_packets
from scripts.build_llm_cluster_packets import build_template_cluster_packets
from scripts.build_log_retrieval_index import build_retrieval_corpus
from scripts.augment_cluster_packets_with_retrieval import retrieve_context_for_packet
from scripts.run_llm_triage import (
    LlmSettings,
    extract_json_object,
    parse_and_validate_response,
    render_prompt,
    run_triage_packets,
    validate_response,
)
from scripts.evaluate_llm_triage import build_llm_summary, load_llm_outputs, merge_llm_with_labels
from thesis_log_anomaly.pipeline import run_pipeline
from thesis_log_anomaly.baselines.event_level import build_event_candidate_rows, run_event_level_baselines
from thesis_log_anomaly.parsing.drain import DrainSettings, parse_with_drain, summarize_clusters
from thesis_log_anomaly.stats.gev import extract_window_statistics, fit_gev_block_maxima, score_windows_with_gev
from thesis_log_anomaly.stats.pot import (
    fit_peak_over_threshold,
    refine_windows_with_burst_pot,
    refine_windows_with_event_pot,
    refine_windows_with_event_pot_per_window,
    refine_windows_with_pot,
)
from thesis_log_anomaly.stages.evaluation import evaluate_candidates
from thesis_log_anomaly.stages.load import load_logs
from thesis_log_anomaly.stages.evt import score_windows_with_evt
from thesis_log_anomaly.stages.pot import refine_with_pot
from thesis_log_anomaly.stages.template import parse_templates
from thesis_log_anomaly.stages.window import build_windows


def test_pipeline_smoke():
    run_pipeline("configs/base.yaml")


def test_bgl_loader_reads_records():
    logs = load_logs({"data": {"dataset": "bgl", "raw_dir": "data/raw"}})
    assert logs
    assert logs[0]["message"]
    assert logs[0]["source"] == "bgl"


def test_thunderbird_loader_reads_records():
    logs = load_logs({"data": {"dataset": "thunderbird", "raw_dir": "data/raw"}})
    assert logs
    assert logs[0]["message"]
    assert logs[0]["source"] == "thunderbird"


def test_thunderbird_pipeline_smoke():
    report = run_pipeline("configs/thunderbird_demo.yaml")
    assert report["dataset"] == "thunderbird"
    assert report["stream_size"] > 0


def test_hdfs2k_loader_reads_records():
    logs = load_logs({"data": {"dataset": "hdfs2k", "raw_dir": "data/raw"}})
    assert logs
    assert logs[0]["message"]
    assert logs[0]["source"] == "hdfs2k"


def test_hdfs100k_loader_reads_records():
    logs = load_logs({"data": {"dataset": "hdfs100k", "raw_dir": "data/raw"}})
    assert logs
    assert logs[0]["message"]
    assert logs[0]["source"] == "hdfs100k"
    assert logs[0]["label"] in (0, 1)


def test_openstack2k_loader_reads_records():
    logs = load_logs({"data": {"dataset": "openstack2k", "raw_dir": "data/raw"}})
    assert logs
    assert logs[0]["message"]
    assert logs[0]["source"] == "openstack2k"


def test_hdfs2k_pipeline_smoke():
    report = run_pipeline("configs/hdfs2k_demo.yaml")
    assert report["dataset"] == "hdfs2k"
    assert report["stream_size"] > 0


def test_hdfs100k_pipeline_smoke():
    report = run_pipeline("configs/hdfs100k_demo.yaml")
    assert report["dataset"] == "hdfs100k"
    assert report["stream_size"] > 0


def test_hdfs100k_grouped_pipeline_smoke():
    report = run_pipeline("configs/hdfs100k_grouped.yaml")
    assert report["dataset"] == "hdfs100k"
    assert report["stream_size"] > 0


def test_openstack2k_pipeline_smoke():
    report = run_pipeline("configs/openstack2k_demo.yaml")
    assert report["dataset"] == "openstack2k"
    assert report["stream_size"] > 0


def test_count_windows_overlap():
    logs = load_logs({"data": {"dataset": "bgl", "raw_dir": "data/raw"}})[:10]
    parsed = {
        "stream_records": logs,
    }
    windows = build_windows(parsed, {"windowing": {"mode": "count", "size": 4, "stride": 2}})
    assert len(windows) >= 4
    assert len(windows[0]["records"]) == 4
    first_ids = [item["event_id"] for item in windows[0]["records"]]
    second_ids = [item["event_id"] for item in windows[1]["records"]]
    assert first_ids[2:] == second_ids[:2]


def test_drain_parser_groups_similar_lines():
    parsed = parse_with_drain(
        [
            "authentication failure for user root from 1.2.3.4",
            "authentication failure for user admin from 5.6.7.8",
            "accepted password for user root from 1.2.3.4",
        ],
        settings=DrainSettings(ini_path="configs/legacy/drain3.ini"),
    )
    assert parsed[0]["template"]
    assert parsed[0]["cluster_id"] == parsed[1]["cluster_id"]
    assert parsed[0]["cluster_id"] != parsed[2]["cluster_id"]
    clusters = summarize_clusters(parsed)
    assert clusters[0]["size"] == 2


def test_template_stage_adds_clusters_and_templates():
    logs = load_logs({"data": {"dataset": "bgl", "raw_dir": "data/raw"}})[:20]
    parsed_stream = parse_templates(
        logs,
        {
            "data": {"history_fraction": 0.25},
            "parsing": {
                "ini_path": "configs/legacy/drain3.ini",
                "drain": {"depth": 4, "similarity_threshold": 0.4, "max_children": 100},
            }
        },
    )
    assert parsed_stream["stream_records"]
    assert parsed_stream["stream_records"][0]["template"]
    windows = build_windows(parsed_stream, {"windowing": {"mode": "count", "size": 10, "stride": 10}})
    assert "clusters" in windows[0]
    assert "template_summary" in windows[0]


def test_template_stage_per_window_scope_keeps_raw_stream_until_windowing():
    logs = load_logs({"data": {"dataset": "bgl", "raw_dir": "data/raw"}})[:20]
    parsed_stream = parse_templates(
        logs,
        {
            "data": {"history_fraction": 0.25},
            "parsing": {
                "scope": "per_window",
                "ini_path": "configs/legacy/drain3.ini",
                "drain": {"depth": 4, "similarity_threshold": 0.4, "max_children": 100},
            },
        },
    )
    assert parsed_stream["stream_records"]
    assert "drain" not in parsed_stream["stream_records"][0]
    assert parsed_stream["history_template_summary"]["count"] > 0


def test_gev_fit_returns_threshold():
    fit = fit_gev_block_maxima([4.0, 5.0, 8.0, 12.0, 20.0], alpha=0.9)
    assert fit["scale"] > 0
    assert fit["threshold"] > 0


def test_gev_scores_suspicious_windows():
    parsed_windows = [
        {"window_id": 1, "clusters": [{"cluster_id": 1, "size": 4}], "records": []},
        {"window_id": 2, "clusters": [{"cluster_id": 1, "size": 5}], "records": []},
        {"window_id": 3, "clusters": [{"cluster_id": 1, "size": 6}], "records": []},
        {"window_id": 4, "clusters": [{"cluster_id": 1, "size": 25}], "records": []},
    ]
    scored = score_windows_with_gev(parsed_windows, alpha=0.75, statistic="max_cluster_size")
    suspicious = [window for window in scored["windows"] if window["evt"]["is_suspicious"]]
    assert suspicious
    assert suspicious[-1]["window_id"] == 4


def test_evt_stage_scores_parsed_windows():
    logs = load_logs({"data": {"dataset": "bgl", "raw_dir": "data/raw"}})[:200]
    parsed_stream = parse_templates(
        logs,
        {
            "data": {"history_fraction": 0.25},
            "parsing": {
                "ini_path": "configs/legacy/drain3.ini",
                "drain": {"depth": 4, "similarity_threshold": 0.4, "max_children": 100},
            }
        },
    )
    windows = build_windows(parsed_stream, {"windowing": {"mode": "count", "size": 50, "stride": 25}})
    scored = score_windows_with_evt(
        windows,
        {
            "evt": {
                "method": "gev",
                "target_statistic": "max_cluster_size",
                "significance_level": 0.8,
                "min_windows_for_fit": 3,
            }
        },
    )
    assert scored
    assert "evt" in scored[0]
    assert "threshold" in scored[0]["evt"]


def test_evt_ensemble_scores_windows():
    logs = load_logs({"data": {"dataset": "bgl", "raw_dir": "data/raw"}})[:200]
    parsed_stream = parse_templates(
        logs,
        {
            "data": {"history_fraction": 0.25, "rare_template_count_threshold": 2},
            "parsing": {
                "ini_path": "configs/legacy/drain3.ini",
                "drain": {"depth": 4, "similarity_threshold": 0.4, "max_children": 100},
            },
        },
    )
    windows = build_windows(parsed_stream, {"windowing": {"mode": "count", "size": 50, "stride": 25}})
    scored = score_windows_with_evt(
        windows,
        {
            "evt": {
                "method": "gev",
                "ensemble_enabled": True,
                "ensemble_statistics": "unique_templates,template_entropy,historically_rare_template_ratio",
                "significance_level": 0.8,
                "min_windows_for_fit": 3,
            }
        },
    )
    assert scored
    assert "evt_ensemble" in scored[0]


def test_pot_fit_returns_absolute_threshold():
    fit = fit_peak_over_threshold(
        [1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0],
        threshold_quantile=0.75,
        tail_alpha=0.20,
    )
    assert fit["threshold_u"] > 0
    assert fit["threshold_z"] >= fit["threshold_u"]


def test_pot_refines_clusters_inside_windows():
    windows = [
        {
            "window_id": 1,
            "evt": {"is_suspicious": True},
            "clusters": [
                {"cluster_id": 1, "size": 2},
                {"cluster_id": 2, "size": 3},
                {"cluster_id": 3, "size": 50},
            ],
        },
        {
            "window_id": 2,
            "evt": {"is_suspicious": True},
            "clusters": [
                {"cluster_id": 4, "size": 2},
                {"cluster_id": 5, "size": 4},
                {"cluster_id": 6, "size": 6},
                {"cluster_id": 7, "size": 80},
            ],
        },
    ]
    refined = refine_windows_with_pot(windows, threshold_quantile=0.7, tail_alpha=0.2)
    flagged = [
        cluster
        for window in refined["windows"]
        for cluster in window["clusters"]
        if cluster["pot"]["is_anomaly"]
    ]
    assert flagged


def test_pot_stage_marks_suspicious_windows_only():
    suspicious_windows = [
        {
            "window_id": 1,
            "evt": {"is_suspicious": True},
            "records": [],
            "clusters": [{"cluster_id": i, "size": size} for i, size in enumerate([1, 2, 3, 40], start=1)],
        },
        {
            "window_id": 2,
            "evt": {"is_suspicious": True},
            "records": [],
            "clusters": [{"cluster_id": i, "size": size} for i, size in enumerate([2, 3, 5, 60], start=10)],
        },
        {
            "window_id": 3,
            "evt": {"is_suspicious": False},
            "records": [],
            "clusters": [{"cluster_id": i, "size": size} for i, size in enumerate([1, 1, 2], start=20)],
        },
    ]
    refined = refine_with_pot(
        suspicious_windows,
        {
            "pot": {
                "enabled": True,
                "score_target": "cluster_size",
                "threshold_quantile": 0.7,
                "tail_alpha": 0.2,
                "min_exceedances": 1,
            }
        },
    )
    assert "pot" in refined[0]
    assert "pot" in refined[1]
    assert "pot" not in refined[2]


def test_evaluation_reports_post_gev_and_post_pot():
    logs = load_logs({"data": {"dataset": "bgl", "raw_dir": "data/raw"}})[:200]
    parsed_stream = parse_templates(
        logs,
        {
            "data": {"history_fraction": 0.25},
            "parsing": {
                "ini_path": "configs/legacy/drain3.ini",
                "drain": {"depth": 4, "similarity_threshold": 0.4, "max_children": 100},
            },
        },
    )
    windows = build_windows(parsed_stream, {"windowing": {"mode": "count", "size": 50, "stride": 25}})
    scored = score_windows_with_evt(
        windows,
        {"evt": {"method": "gev", "target_statistic": "max_cluster_size", "significance_level": 0.8, "min_windows_for_fit": 3}},
    )
    refined = refine_with_pot(
        scored,
        {"pot": {"enabled": True, "threshold_quantile": 0.7, "tail_alpha": 0.2, "min_exceedances": 1}},
    )
    report = evaluate_candidates(
        {"parsed_stream": parsed_stream, "windows_after_gev": scored, "windows_after_pot": refined},
        {"data": {"dataset": "bgl"}, "evaluation": {"enabled": False}},
    )
    assert "post_gev" in report
    assert "post_pot" in report


def test_window_features_include_entropy_and_novelty():
    logs = load_logs({"data": {"dataset": "bgl", "raw_dir": "data/raw"}})[:120]
    parsed_stream = parse_templates(
        logs,
        {
            "data": {"history_fraction": 0.25},
            "parsing": {
                "ini_path": "configs/legacy/drain3.ini",
                "drain": {"depth": 4, "similarity_threshold": 0.4, "max_children": 100},
            },
        },
    )
    windows = build_windows(parsed_stream, {"windowing": {"mode": "count", "size": 40, "stride": 20}})
    features = windows[0]["feature_summary"]
    assert "template_entropy" in features
    assert "new_template_ratio" in features
    assert "max_cluster_ratio" in features
    assert "unique_template_ratio" in features
    assert "historically_rare_template_ratio" in features
    assert "novelty_magnitude_score" in features
    assert "max_event_novelty_score" in features


def test_build_windows_reparses_records_per_window_when_enabled():
    logs = [
        {"event_id": 1, "message": "accepted password for user root from 1.2.3.4", "label": 0},
        {"event_id": 2, "message": "accepted password for user admin from 5.6.7.8", "label": 0},
        {"event_id": 3, "message": "failed password for invalid user guest from 8.8.8.8", "label": 1},
        {"event_id": 4, "message": "failed password for invalid user test from 9.9.9.9", "label": 1},
    ]
    parsed_stream = parse_templates(
        logs,
        {
            "data": {"history_fraction": 0.25, "rare_template_count_threshold": 2},
            "parsing": {
                "scope": "per_window",
                "ini_path": "configs/legacy/drain3.ini",
                "drain": {"depth": 4, "similarity_threshold": 0.4, "max_children": 100},
            },
        },
    )
    windows = build_windows(
        parsed_stream,
        {
            "data": {"rare_template_count_threshold": 2},
            "parsing": {
                "scope": "per_window",
                "ini_path": "configs/legacy/drain3.ini",
                "drain": {"depth": 4, "similarity_threshold": 0.4, "max_children": 100},
            },
            "windowing": {"mode": "count", "size": 2, "stride": 2},
        },
    )
    assert windows
    assert windows[0]["records"][0]["drain"]["cluster_id"] is not None
    assert "novelty" in windows[0]["records"][0]
    assert windows[0]["clusters"]


def test_stream_records_receive_novelty_scores():
    logs = load_logs({"data": {"dataset": "bgl", "raw_dir": "data/raw"}})[:100]
    parsed_stream = parse_templates(
        logs,
        {
            "data": {"history_fraction": 0.25, "rare_template_count_threshold": 2},
            "parsing": {
                "ini_path": "configs/legacy/drain3.ini",
                "drain": {"depth": 4, "similarity_threshold": 0.4, "max_children": 100},
            },
        },
    )
    novelty = parsed_stream["stream_records"][0]["novelty"]
    assert "novelty_score" in novelty
    assert novelty["novelty_score"] >= 0
    assert "drain_reference" in parsed_stream["stream_records"][0]


def test_event_novelty_pot_marks_candidate_events():
    windows = [
        {
            "window_id": 1,
            "records": [
                {"event_id": 1, "novelty": {"novelty_score": 0.2}},
                {"event_id": 2, "novelty": {"novelty_score": 0.4}},
                {"event_id": 3, "novelty": {"novelty_score": 3.0}},
            ],
            "clusters": [],
        },
        {
            "window_id": 2,
            "records": [
                {"event_id": 4, "novelty": {"novelty_score": 0.1}},
                {"event_id": 5, "novelty": {"novelty_score": 4.0}},
            ],
            "clusters": [],
        },
    ]
    refined = refine_windows_with_event_pot(windows, threshold_quantile=0.7, tail_alpha=0.2)
    candidate_ids = [event_id for window in refined["windows"] for event_id in window["pot"]["candidate_event_ids"]]
    assert candidate_ids


def test_event_rarity_pot_marks_candidate_events():
    windows = [
        {
            "window_id": 1,
            "records": [
                {"event_id": 1, "novelty": {"rarity_score": 0.1}},
                {"event_id": 2, "novelty": {"rarity_score": 0.2}},
                {"event_id": 3, "novelty": {"rarity_score": 0.9}},
            ],
            "clusters": [],
        },
        {
            "window_id": 2,
            "records": [
                {"event_id": 4, "novelty": {"rarity_score": 0.15}},
                {"event_id": 5, "novelty": {"rarity_score": 1.0}},
            ],
            "clusters": [],
        },
    ]
    refined = refine_windows_with_event_pot(
        windows,
        threshold_quantile=0.7,
        tail_alpha=0.2,
        score_field="rarity_score",
        score_target="historical_rarity",
    )
    candidate_ids = [event_id for window in refined["windows"] for event_id in window["pot"]["candidate_event_ids"]]
    assert candidate_ids


def test_event_novelty_pot_per_window_marks_candidate_events():
    windows = [
        {
            "window_id": 1,
            "records": [
                {"event_id": 1, "novelty": {"novelty_score": 0.1}},
                {"event_id": 2, "novelty": {"novelty_score": 0.2}},
                {"event_id": 3, "novelty": {"novelty_score": 4.0}},
            ],
            "clusters": [],
        },
        {
            "window_id": 2,
            "records": [
                {"event_id": 4, "novelty": {"novelty_score": 0.1}},
                {"event_id": 5, "novelty": {"novelty_score": 3.0}},
                {"event_id": 6, "novelty": {"novelty_score": 3.5}},
            ],
            "clusters": [],
        },
    ]
    refined = refine_windows_with_event_pot_per_window(
        windows,
        threshold_quantile=0.6,
        tail_alpha=0.3,
        score_field="novelty_score",
        score_target="event_novelty",
        min_exceedances=1,
    )
    candidate_ids = [event_id for window in refined["windows"] for event_id in window["pot"]["candidate_event_ids"]]
    assert candidate_ids


def test_template_burst_pot_marks_candidate_events():
    windows = [
        {
            "window_id": 1,
            "records": [
                {"event_id": 1, "novelty": {"unseen_in_history": False, "is_new_template": False}},
                {"event_id": 2, "novelty": {"unseen_in_history": True, "is_new_template": False}},
                {"event_id": 3, "novelty": {"unseen_in_history": True, "is_new_template": True}},
                {"event_id": 4, "novelty": {"unseen_in_history": True, "is_new_template": False}},
                {"event_id": 5, "novelty": {"unseen_in_history": False, "is_new_template": False}},
            ],
            "clusters": [],
        },
        {
            "window_id": 2,
            "records": [
                {"event_id": 6, "novelty": {"unseen_in_history": False, "is_new_template": False}},
                {"event_id": 7, "novelty": {"unseen_in_history": True, "is_new_template": False}},
            ],
            "clusters": [],
        },
    ]
    refined = refine_windows_with_burst_pot(
        windows,
        threshold_quantile=0.6,
        tail_alpha=0.2,
    )
    candidate_ids = [event_id for window in refined["windows"] for event_id in window["pot"]["candidate_event_ids"]]
    assert candidate_ids


def test_event_level_baseline_runs_inside_suspicious_windows():
    windows = []
    stream_records = []
    event_id = 1
    for block_id in ["1", "2", "3"]:
        records = []
        for label in [0, 1]:
            record = {
                "event_id": event_id,
                "message": f"event {event_id}",
                "template": "normal template" if label == 0 else "rare failure template",
                "label": label,
                "metadata": {"block_id": block_id},
                "drain": {"is_new_template": label == 1},
                "novelty": {
                    "historical_count": 10 if label == 0 else 0,
                    "unseen_in_history": label == 1,
                    "historically_rare": label == 1,
                    "is_new_template": label == 1,
                    "rarity_score": 0.1 if label == 0 else 1.0,
                    "novelty_score": 0.1 if label == 0 else 3.0,
                },
            }
            records.append(record)
            stream_records.append(record)
            event_id += 1
        windows.append(
            {
                "window_id": int(block_id),
                "evt": {
                    "method": "gev",
                    "is_suspicious": True,
                    "value": 3.0,
                    "threshold": 1.0,
                },
                "records": records,
                "template_summary": {
                    "template_counts": {
                        "normal template": 1,
                        "rare failure template": 1,
                    }
                },
                "feature_summary": {
                    "num_records": 2,
                    "template_entropy": 1.0,
                    "historically_rare_template_ratio": 0.5,
                    "unseen_in_history_ratio": 0.5,
                    "new_template_ratio": 0.5,
                    "mean_event_novelty_score": 1.55,
                },
            }
        )

    rows = build_event_candidate_rows(windows)
    assert len(rows) == 6
    assert "template_count_past_z" in rows[0]
    assert "template_count_deviation_score" in rows[0]
    assert "local_sequence_context_score" in rows[0]
    assert "neighbor_max_novelty_score" in rows[0]

    report = run_event_level_baselines(windows, stream_records, min_validation_recall=0.9)
    assert report["best"]
    assert report["total_event_summary"]["candidate_anomalous_events_inside_suspicious_windows"] == 3
    assert report["results"]


def test_template_burst_selector_sweeps_use_configurable_totals():
    df = pd.DataFrame(
        [
            {"event_id": 10, "event_order": 10, "label": 1, "template_burst_score": 100.0},
            {"event_id": 20, "event_order": 20, "label": 0, "template_burst_score": 75.0},
            {"event_id": 30, "event_order": 30, "label": 1, "template_burst_score": 25.0},
        ]
    )
    totals = _infer_totals(df, total_stream_events=100, total_anomalies=4)
    threshold_sweep = _threshold_sweep(df, [50.0, 100.0], totals)
    ranked = df.sort_values("template_burst_score", ascending=False)
    topn_sweep = _topn_sweep(ranked, [1, 2], totals)

    assert totals["total_stream_events"] == 100
    assert threshold_sweep.loc[threshold_sweep["threshold"] == 50.0, "num_events"].iloc[0] == 2
    assert threshold_sweep.loc[threshold_sweep["threshold"] == 100.0, "recall_against_candidate_anomalies"].iloc[0] == 0.25
    assert topn_sweep.loc[topn_sweep["top_n"] == 1, "precision"].iloc[0] == 1.0
    assert topn_sweep.loc[topn_sweep["top_n"] == 2, "event_load_ratio_against_stream"].iloc[0] == 0.02


def test_classical_triage_baseline_reports_rank_metrics_and_templates():
    df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "event_order": 1,
                "label": 1,
                "template": "fatal a",
                "template_burst_score": 10.0,
                "template_count_deviation_score": 0.5,
                "novelty_score": 1.0,
                "rarity_score": 1.0,
                "template_count_past_z": 0.0,
                "local_sequence_context_score": 0.0,
            },
            {
                "event_id": 2,
                "event_order": 2,
                "label": 0,
                "template": "normal b",
                "template_burst_score": 5.0,
                "template_count_deviation_score": 2.0,
                "novelty_score": 0.2,
                "rarity_score": 0.2,
                "template_count_past_z": 1.0,
                "local_sequence_context_score": 0.5,
            },
            {
                "event_id": 3,
                "event_order": 3,
                "label": 1,
                "template": "fatal c",
                "template_burst_score": 1.0,
                "template_count_deviation_score": 4.0,
                "novelty_score": 0.1,
                "rarity_score": 0.1,
                "template_count_past_z": 2.0,
                "local_sequence_context_score": 1.0,
            },
        ]
    )
    markov_scores = _markov_surprise(df, order=2)
    summary = build_classical_triage_summary(
        df,
        methods=["template_burst", "template_count_deviation", "markov_bigram_surprise"],
        top_k=[1, 2],
        totals={"total_stream_events": 10, "total_anomalies": 2},
    )
    burst_top_1 = summary[(summary["method"] == "template_burst") & (summary["top_k"] == 1)].iloc[0]
    deviation_top_1 = summary[
        (summary["method"] == "template_count_deviation") & (summary["top_k"] == 1)
    ].iloc[0]

    assert rank_events(df, "template_burst").iloc[0]["event_id"] == 1
    assert burst_top_1["precision"] == 1.0
    assert burst_top_1["recall_against_all_anomalies"] == 0.5
    assert burst_top_1["unique_templates"] == 1
    assert deviation_top_1["positive_events"] == 1
    assert len(markov_scores) == len(df)
    assert markov_scores.iloc[0] == 0.0


def test_llm_review_packets_are_ranked_and_label_free():
    df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "event_order": 1,
                "block_id": 1,
                "label": 1,
                "template": "fatal socket disconnect",
                "message": "RAS APP FATAL socket disconnected",
                "template_burst_score": 10.0,
                "template_count_deviation_score": 1.0,
                "novelty_score": 2.0,
                "rarity_score": 3.0,
                "historical_count_log": 0.0,
                "suspicious_window_count": 2,
                "max_window_gev_score": 4.0,
                "local_sequence_context_score": 0.5,
            },
            {
                "event_id": 2,
                "event_order": 2,
                "block_id": 1,
                "label": 0,
                "template": "normal core generated",
                "message": "RAS KERNEL INFO generating core",
                "template_burst_score": 1.0,
                "template_count_deviation_score": 5.0,
                "novelty_score": 0.1,
                "rarity_score": 0.2,
                "historical_count_log": 7.0,
                "suspicious_window_count": 1,
                "max_window_gev_score": 1.0,
                "local_sequence_context_score": 0.0,
            },
            {
                "event_id": 3,
                "event_order": 3,
                "block_id": 1,
                "label": 1,
                "template": "fatal socket disconnect",
                "message": "RAS APP FATAL socket disconnected again",
                "template_burst_score": 9.0,
                "template_count_deviation_score": 2.0,
                "novelty_score": 2.5,
                "rarity_score": 3.5,
                "historical_count_log": 0.0,
                "suspicious_window_count": 3,
                "max_window_gev_score": 5.0,
                "local_sequence_context_score": 0.7,
            },
        ]
    )

    packets = build_review_packets(
        df,
        dataset="unit",
        selection_method="template_burst",
        top_k=2,
        prompt_strategy="event_explanation_v1",
        same_template_examples=1,
        neighbor_events=1,
    )

    assert [packet["event_id"] for packet in packets] == [1, 3]
    assert packets[0]["packet_schema_version"] == "llm_event_packet_v1"
    assert packets[0]["scores"]["template_burst_score"] == 10.0
    assert packets[0]["window_context"]["max_window_gev_score"] == 4.0
    assert packets[1]["related_events"]["same_template_examples"][0]["event_id"] == 1
    assert "label" not in str(packets).lower()


def test_llm_review_packet_leakage_guard_rejects_label_keys():
    try:
        assert_no_label_leakage({"packet_id": "x", "label": 1})
    except ValueError as error:
        assert "Label-like key" in str(error)
    else:
        raise AssertionError("Expected label leakage guard to reject label keys")


def test_llm_cluster_packets_group_ranked_events_without_labels():
    df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "event_order": 1,
                "block_id": 1,
                "label": 1,
                "template": "fatal socket disconnect",
                "message": "RAS APP FATAL socket disconnected",
                "template_burst_score": 10.0,
                "template_count_deviation_score": 1.0,
                "novelty_score": 2.0,
                "rarity_score": 3.0,
            },
            {
                "event_id": 2,
                "event_order": 2,
                "block_id": 1,
                "label": 0,
                "template": "normal core generated",
                "message": "RAS KERNEL INFO generating core",
                "template_burst_score": 1.0,
                "template_count_deviation_score": 5.0,
                "novelty_score": 0.1,
                "rarity_score": 0.2,
            },
            {
                "event_id": 3,
                "event_order": 3,
                "block_id": 2,
                "label": 1,
                "template": "fatal socket disconnect",
                "message": "RAS APP FATAL socket disconnected again",
                "template_burst_score": 9.0,
                "template_count_deviation_score": 2.0,
                "novelty_score": 2.5,
                "rarity_score": 3.5,
            },
        ]
    )

    packets = build_template_cluster_packets(
        df,
        dataset="unit",
        selection_method="template_burst",
        top_k=3,
        max_events_per_cluster=4,
    )

    assert len(packets) == 2
    assert packets[0]["cluster"]["template"] == "fatal socket disconnect"
    assert packets[0]["cluster"]["event_count"] == 2
    assert packets[0]["cluster"]["unique_blocks"] == 2
    assert packets[0]["score_summary"]["template_burst_score"]["max"] == 10.0
    assert [event["event_id"] for event in packets[0]["representative_events"]] == [1, 3]
    assert "label" not in json.dumps(packets).lower()


def test_llm_triage_json_extraction_and_validation():
    packet = {"packet_id": "unit:1"}
    response = {
        "packet_id": "unit:1",
        "review_decision": "likely_anomaly",
        "severity": "high",
        "confidence": 0.8,
        "reason_codes": ["burst", "rare_template"],
        "rationale": "The event combines a burst with rare template evidence.",
        "suspicious_evidence": ["High burst score", "Rare template evidence"],
        "benign_evidence": [],
        "recommended_action": "inspect_event",
        "needs_more_context": False,
    }
    noisy = "prefix\n" + json.dumps(response) + "\ntrailing text"
    parsed = extract_json_object(noisy)
    parsed_response, errors = parse_and_validate_response(noisy, packet)

    assert parsed["packet_id"] == "unit:1"
    assert parsed_response == response
    assert errors == []


def test_llm_triage_normalizes_feature_reason_code_aliases():
    packet = {"packet_id": "unit:1"}
    response = {
        "packet_id": "unit:1",
        "review_decision": "likely_anomaly",
        "severity": "medium",
        "confidence": 0.8,
        "reason_codes": ["template_burst_score", "novelty_score", "rarity_score"],
        "rationale": "The event combines burst, novelty, and rarity evidence.",
        "suspicious_evidence": ["High burst score", "Rare template evidence"],
        "benign_evidence": [],
        "recommended_action": "inspect_template_cluster",
        "needs_more_context": False,
    }
    parsed_response, errors = parse_and_validate_response(json.dumps(response), packet)

    assert errors == []
    assert parsed_response["reason_codes"] == ["template_burst", "novel_template", "rare_template"]


def test_llm_triage_accepts_optional_rerank_fields():
    packet = {"packet_id": "unit:1"}
    response = {
        "packet_id": "unit:1",
        "review_decision": "uncertain",
        "severity": "medium",
        "confidence": 0.7,
        "triage_score": 66,
        "review_priority": "medium",
        "reason_codes": ["template_burst", "insufficient_context"],
        "rationale": "The cluster is bursty but needs more operational context.",
        "suspicious_evidence": ["High burst score"],
        "benign_evidence": ["No fatal wording"],
        "recommended_action": "inspect_template_cluster",
        "needs_more_context": True,
    }
    parsed_response, errors = parse_and_validate_response(json.dumps(response), packet)

    assert errors == []
    assert parsed_response["triage_score"] == 66
    assert parsed_response["review_priority"] == "medium"


def test_llm_triage_repairs_low_priority_normal_inspect_action():
    packet = {"packet_id": "unit:1"}
    response = {
        "packet_id": "unit:1",
        "review_decision": "likely_normal",
        "severity": "low",
        "confidence": 0.9,
        "triage_score": 25,
        "review_priority": "low",
        "reason_codes": ["benign_repetition"],
        "rationale": "The cluster appears low-priority and operational.",
        "suspicious_evidence": [],
        "benign_evidence": ["Low-priority operational wording"],
        "recommended_action": "inspect_template_cluster",
        "needs_more_context": False,
    }
    parsed_response, errors = parse_and_validate_response(json.dumps(response), packet)

    assert errors == []
    assert parsed_response["recommended_action"] == "ignore"


def test_llm_triage_validation_rejects_bad_schema_and_eval_terms():
    response = {
        "packet_id": "wrong",
        "review_decision": "yes",
        "severity": "urgent",
        "confidence": 2.0,
        "reason_codes": ["mystery"],
        "rationale": "This references labels and precision.",
        "suspicious_evidence": "not-list",
        "benign_evidence": [],
        "recommended_action": "panic",
        "needs_more_context": "no",
    }
    errors = validate_response(response, expected_packet_id="unit:1")

    assert any("packet_id" in error for error in errors)
    assert any("Invalid review_decision" in error for error in errors)
    assert any("Forbidden evaluation terms" in error for error in errors)


def test_llm_triage_validation_rejects_normal_inspect_inconsistency():
    response = {
        "packet_id": "unit:1",
        "review_decision": "likely_normal",
        "severity": "low",
        "confidence": 0.8,
        "reason_codes": ["benign_repetition"],
        "rationale": "The event looks repetitive and common.",
        "suspicious_evidence": [],
        "benign_evidence": ["Repeated template"],
        "recommended_action": "inspect_window",
        "needs_more_context": False,
    }
    errors = validate_response(response, expected_packet_id="unit:1")

    assert any("likely_normal should recommend ignore" in error for error in errors)
    assert any("inspect actions require" in error for error in errors)


def test_llm_triage_dry_run_renders_prompt_and_records_settings():
    packet = {
        "packet_id": "unit:1",
        "event_id": 1,
        "selection_rank": 1,
        "message": "RAS APP FATAL socket disconnected",
    }
    prompt = render_prompt("Review:\n{{PACKET_JSON}}", packet)
    results = run_triage_packets(
        [packet],
        prompt_template="Review:\n{{PACKET_JSON}}",
        prompt_sha256="abc",
        settings=LlmSettings(model="qwen3:8b", temperature=0.0),
        cache_dir=None,
        retries=0,
        dry_run=True,
    )

    assert '"packet_id": "unit:1"' in prompt
    assert results[0]["packet_id"] == "unit:1"
    assert results[0]["request_sha256"]
    assert results[0]["valid_json"] is False
    assert results[0]["validation_errors"] == ["dry_run: model was not called"]


def test_llm_triage_evaluation_merges_labels_and_reports_policies():
    llm_df = pd.DataFrame(
        [
            {
                "packet_id": "p1",
                "event_id": 1,
                "selection_rank": 1,
                "valid_json": True,
                "review_decision": "likely_anomaly",
                "recommended_action": "inspect_event",
                "confidence": 0.9,
                "latency_ms": 100,
                "total_tokens": 1000,
                "prompt_tokens": 900,
                "completion_tokens": 100,
                "cache_hit": False,
            },
            {
                "packet_id": "p2",
                "event_id": 2,
                "selection_rank": 2,
                "valid_json": True,
                "review_decision": "likely_normal",
                "recommended_action": "ignore",
                "confidence": 0.8,
                "latency_ms": 200,
                "total_tokens": 800,
                "prompt_tokens": 700,
                "completion_tokens": 100,
                "cache_hit": False,
            },
            {
                "packet_id": "p3",
                "event_id": 3,
                "selection_rank": 3,
                "valid_json": True,
                "review_decision": "uncertain",
                "recommended_action": "inspect_window",
                "confidence": 0.5,
                "latency_ms": 300,
                "total_tokens": 900,
                "prompt_tokens": 800,
                "completion_tokens": 100,
                "cache_hit": False,
            },
        ]
    )
    candidates = pd.DataFrame(
        [
            {"event_id": 1, "label": 1, "template": "fatal", "message": "fatal one"},
            {"event_id": 2, "label": 0, "template": "normal", "message": "normal two"},
            {"event_id": 3, "label": 1, "template": "rare", "message": "rare three"},
        ]
    )
    evaluated = merge_llm_with_labels(llm_df, candidates)
    summary = build_llm_summary(
        evaluated,
        top_k_values=[3],
        totals={"total_stream_events": 30, "total_anomalies": 4},
    )
    likely = summary[summary["policy"] == "likely_anomaly"].iloc[0]
    inspect = summary[summary["policy"] == "inspect_action"].iloc[0]
    uncertain = summary[summary["policy"] == "uncertain_or_anomaly"].iloc[0]

    assert likely["selected_events"] == 1
    assert likely["precision"] == 1.0
    assert likely["recall_against_all_anomalies"] == 0.25
    assert inspect["selected_events"] == 2
    assert inspect["positive_events"] == 2
    assert uncertain["selected_events"] == 2
    assert uncertain["total_tokens"] == 2700


def test_llm_triage_evaluation_reranks_by_triage_score():
    evaluated = pd.DataFrame(
        [
            {
                "packet_id": "p1",
                "event_id": 1,
                "selection_rank": 1,
                "valid_json": True,
                "review_decision": "likely_normal",
                "recommended_action": "ignore",
                "triage_score": 10,
                "confidence": 0.9,
                "latency_ms": 100,
                "total_tokens": 100,
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "cache_hit": False,
                "label": 0,
                "template": "normal",
            },
            {
                "packet_id": "p2",
                "event_id": 2,
                "selection_rank": 2,
                "valid_json": True,
                "review_decision": "uncertain",
                "recommended_action": "inspect_template_cluster",
                "triage_score": 90,
                "confidence": 0.8,
                "latency_ms": 100,
                "total_tokens": 100,
                "prompt_tokens": 80,
                "completion_tokens": 20,
                "cache_hit": False,
                "label": 1,
                "template": "fatal",
            },
        ]
    )
    evaluated["llm_likely_anomaly"] = evaluated["review_decision"] == "likely_anomaly"
    evaluated["llm_review_action"] = evaluated["recommended_action"].isin(["inspect_template_cluster"])
    evaluated["llm_uncertain_or_anomaly"] = evaluated["review_decision"].isin(["likely_anomaly", "uncertain"])

    summary = build_llm_summary(
        evaluated,
        top_k_values=[1],
        totals={"total_stream_events": 10, "total_anomalies": 1},
    )
    rerank = summary[summary["policy"] == "triage_score_rank"].iloc[0]

    assert rerank["selected_events"] == 1
    assert rerank["positive_events"] == 1
    assert rerank["precision"] == 1.0


def test_llm_triage_cluster_evaluation_expands_member_events():
    llm_path = ROOT / "outputs" / "reports" / "test_cluster_llm_eval_tmp.jsonl"
    llm_path.parent.mkdir(parents=True, exist_ok=True)
    packet_rows = [
        {
            "packet_id": "cluster:1",
            "packet_schema_version": "llm_template_cluster_packet_v1",
            "cluster": {
                "cluster_rank": 1,
                "event_count": 2,
                "member_event_ids": [10, 11],
                "member_selection_ranks": [1, 2],
            },
        }
    ]
    llm_row = {
        "packet_id": "cluster:1",
        "valid_json": True,
        "review_decision": "likely_anomaly",
        "recommended_action": "inspect_template_cluster",
        "confidence": 0.9,
        "latency_ms": 50,
        "total_tokens": 500,
        "prompt_tokens": 450,
        "completion_tokens": 50,
        "cache_hit": False,
        "parsed_response": {
            "severity": "medium",
            "reason_codes": ["template_burst"],
            "rationale": "Cluster is bursty.",
        },
        "validation_errors": [],
    }
    llm_path.write_text(json.dumps(llm_row) + "\n", encoding="utf-8")
    candidates = pd.DataFrame(
        [
            {"event_id": 10, "label": 1, "template": "fatal", "message": "fatal one"},
            {"event_id": 11, "label": 0, "template": "fatal", "message": "fatal two"},
        ]
    )

    llm_df = load_llm_outputs(llm_path, packet_rows=packet_rows, evaluation_mode="cluster")
    evaluated = merge_llm_with_labels(llm_df, candidates)
    summary = build_llm_summary(
        evaluated,
        top_k_values=[2],
        totals={"total_stream_events": 20, "total_anomalies": 2},
    )
    likely = summary[summary["policy"] == "likely_anomaly"].iloc[0]

    assert list(llm_df["event_id"]) == [10, 11]
    assert likely["selected_events"] == 2
    assert likely["positive_events"] == 1
    assert likely["llm_input_packets"] == 1
    assert likely["total_tokens"] == 500


def test_retrieval_corpus_excludes_labels_and_builds_text():
    df = pd.DataFrame(
        [
            {
                "event_id": 1,
                "event_order": 1,
                "block_id": 1,
                "label": 1,
                "template": "fatal socket",
                "message": "RAS APP FATAL socket disconnected",
                "template_burst_score": 10.0,
            }
        ]
    )
    corpus = build_retrieval_corpus(df, text_mode="template_message")

    assert "label" not in corpus.columns
    assert "Template: fatal socket" in corpus.iloc[0]["retrieval_text"]
    assert "Message: RAS APP FATAL" in corpus.iloc[0]["retrieval_text"]


def test_retrieval_context_excludes_cluster_members_and_returns_contrast():
    packet = {
        "packet_id": "cluster:1",
        "cluster": {
            "template": "fatal socket",
            "member_event_ids": [1],
        },
        "score_summary": {
            "template_burst_score": {"mean": 10.0},
        },
    }
    meta = pd.DataFrame(
        [
            {
                "event_id": 1,
                "event_order": 1,
                "block_id": 1,
                "template": "fatal socket",
                "message": "member",
                "template_burst_score": 10.0,
            },
            {
                "event_id": 2,
                "event_order": 2,
                "block_id": 1,
                "template": "fatal socket",
                "message": "same template lower score",
                "template_burst_score": 1.0,
            },
            {
                "event_id": 3,
                "event_order": 3,
                "block_id": 1,
                "template": "normal core",
                "message": "other template",
                "template_burst_score": 2.0,
            },
        ]
    )
    embeddings = pd.DataFrame([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]]).to_numpy()
    query = pd.Series([1.0, 0.0]).to_numpy()

    context = retrieve_context_for_packet(
        packet,
        meta,
        embeddings,
        query,
        nearest_k=2,
        same_template_k=2,
        low_score_k=2,
    )

    nearest_ids = [example["event_id"] for example in context["nearest_examples"]]
    same_template_ids = [example["event_id"] for example in context["same_template_examples"]]
    lower_score_ids = [example["event_id"] for example in context["lower_score_contrast_examples"]]

    assert 1 not in nearest_ids
    assert same_template_ids == [2]
    assert 2 in lower_score_ids
    assert '"label"' not in json.dumps(context).lower()
