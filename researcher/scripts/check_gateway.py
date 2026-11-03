"""Safety check: the gateway must block unsafe experiments and audit every call.

Run from the researcher folder:  ..\.venv\Scripts\python.exe scripts\check_gateway.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from atr import config  # noqa: E402
from atr.experiment import ExperimentRunner  # noqa: E402
from atr.gateway import ToolError, build_gateway  # noqa: E402
from atr.twin import RFTwin  # noqa: E402
from atr.world import World  # noqa: E402

twin = RFTwin.load_or_build()
inv = twin.inventory()
gw = build_gateway(ExperimentRunner(twin, inv, World(twin, inv).field_observation(200)))

cases = {
    "tilt 30 (out of safe range)": ({"cells": {"1": {"tilt": 30}}}, True),
    "two settings changed at once": ({"cells": {"1": {"tilt": 8}, "2": {"tilt": 8}}}, True),
    "unknown cell 99": ({"cells": {"99": {"tilt": 8}}}, True),
    "forbidden field 'txpower'": ({"cells": {"1": {"txpower": 30}}}, True),
    "valid: cell 1 tilt 8": ({"cells": {"1": {"tilt": 8}}}, False),
}
failed = 0
for name, (cfg, should_block) in cases.items():
    try:
        gw.invoke("run_experiment", {"config": cfg, "seeds": [1]}, agent="check")
        blocked = False
    except ToolError as e:
        blocked, why = True, str(e)
    ok = blocked == should_block
    failed += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {name:<32} -> {'blocked: ' + why if blocked else 'allowed'}")

lines = sum(1 for _ in open(config.AUDIT_LOG, encoding="utf-8"))
print(f"audit log: {lines} records in {config.AUDIT_LOG}")
sys.exit(1 if failed else 0)
