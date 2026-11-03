"""Paths and radio thresholds. Override paths with env vars when moving machines."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HACKATHON = ROOT.parent

MAVERIC_ROOT = Path(os.environ.get("MAVERIC_ROOT", HACKATHON / "maveric-1.0.0 1" / "maveric-1.0.0"))
SIM_DATA = Path(os.environ.get("ATR_SIM_DATA", HACKATHON / "data" / "sim_data" / "sim_data" / "3cell"))

MODELS = ROOT / "models"
RUNS = Path(os.environ.get("ATR_RUNS_DIR", ROOT / "runs"))   # /tmp/runs on Lambda (read-only code dir)
HEATMAPS = RUNS / "heatmaps"
AUDIT_LOG = RUNS / "audit.jsonl"
DB_PATH = RUNS / "findings.sqlite"

for _p in (MODELS, RUNS, HEATMAPS):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass   # read-only filesystem (the model dir is baked into the image)

# MAVERIC's radp package is imported straight from its source tree.
if str(MAVERIC_ROOT) not in sys.path:
    sys.path.insert(0, str(MAVERIC_ROOT))

# Radio thresholds used by the KPI proxies.
NOISE_DBM = -125.0        # per-resource-element noise floor (thermal + noise figure)
COVERAGE_RSRP_DBM = -110.0
RLF_RSRP_DBM = -115.0     # radio-link-failure risk below this serving RSRP...
RLF_SINR_DB = 0.0         # ...or below this SINR
OFF_RSRP_DBM = -200.0     # RSRP assigned to a switched-off cell
DETECT_FLOOR_DBM = -125.0  # a UE's measurement report lists every cell it can hear above this
WEAK_COVERAGE_DBM = -110.0  # "weak coverage" symptom threshold (serving RSRP)

# Safety bounds enforced at the gateway (the twin is trained on 4-16 degrees).
TILT_RANGE = (4.0, 16.0)
# Calibration offset the Critic's null test may apply to one cell's predicted RSRP (twin mis-calibration, not a
# network setting).
OFFSET_RANGE = (-8.0, 8.0)
