from thesis_log_anomaly.datasets.bgl import load_bgl_records, to_rows as bgl_to_rows
from thesis_log_anomaly.datasets.hdfs100k import load_hdfs100k_records, to_rows as hdfs100k_to_rows
from thesis_log_anomaly.datasets.hdfs2k import load_hdfs2k_records, to_rows as hdfs2k_to_rows
from thesis_log_anomaly.datasets.linux2k import load_linux2k_records, to_rows as linux2k_to_rows
from thesis_log_anomaly.datasets.openstack2k import (
    load_openstack2k_records,
    to_rows as openstack2k_to_rows,
)
from thesis_log_anomaly.datasets.thunderbird import (
    load_thunderbird_records,
    to_rows as thunderbird_to_rows,
)


def load_logs(config: dict):
    """Load logs from the configured dataset source."""
    data_config = config.get("data", {})
    dataset = data_config.get("dataset", "bgl")
    raw_dir = data_config.get("raw_dir", "data/raw")

    if dataset == "bgl":
        dataset_subdir = data_config.get("dataset_subdir", "bgl")
        return bgl_to_rows(load_bgl_records(f"{raw_dir}/{dataset_subdir}"))
    if dataset == "linux2k":
        return linux2k_to_rows(load_linux2k_records(f"{raw_dir}/linux2k"))
    if dataset == "hdfs100k":
        return hdfs100k_to_rows(load_hdfs100k_records(f"{raw_dir}/hdfs100k"))
    if dataset == "hdfs2k":
        return hdfs2k_to_rows(load_hdfs2k_records(f"{raw_dir}/hdfs2k"))
    if dataset == "openstack2k":
        return openstack2k_to_rows(load_openstack2k_records(f"{raw_dir}/openstack2k"))
    if dataset == "thunderbird":
        return thunderbird_to_rows(load_thunderbird_records(f"{raw_dir}/thunderbird"))

    raise ValueError(f"Unsupported dataset: {dataset}")
