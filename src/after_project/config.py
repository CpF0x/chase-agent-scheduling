"""Project paths and shared experiment constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

BORG_TRACE_PATH = DATA_DIR / "borg_traces_data.csv"
FL_DRL_MODEL_PATH = MODEL_DIR / "fl_drl_model.pt"

NUM_TRIALS = 50
AGENT_COUNT = 40
TASK_NUM_RANGE = [20, 40, 60, 80, 100, 120]

ALGORITHMS = ["MCT", "CHASE", "IRS", "DyLAN", "FL_DRL", "BRTOA"]
CHASE_ABLATIONS = ["CHASE", "CHASE_NO_CONG", "CHASE_NO_REPICK", "CHASE_NO_SMART"]
