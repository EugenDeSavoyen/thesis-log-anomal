from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from thesis_log_anomaly.pipeline import run_pipeline


if __name__ == "__main__":
    config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/base.yaml"
    run_pipeline(config_path=config_path)
