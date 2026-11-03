"""The 'field': the real network as the researcher can observe it, including a hidden fault.

Agents only ever receive `field_observation()` and the documented inventory. `truth()` exists solely for the
reveal at the end of a blind test and is never passed to an agent.

Local stand-in: the field is the twin run with the faulted settings plus measurement noise. When moving to a
live MAVERIC simulation, replace `field_observation` with a call that runs that simulation.
"""
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config, kpi
from .faults import Fault  # noqa: F401 (re-exported)
from .twin import RFTwin, Settings

SHADOWING_STD_DB = 2.0


class World:
    def __init__(self, twin: RFTwin, inventory: Settings, fault: Optional[Fault] = None,
                 twin_bias: Optional[Dict[int, float]] = None):
        """`twin_bias` models the twin being wrong about the real network: a constant dB offset per cell (clutter,
        cable loss, antenna gain the training data did not capture). It is what twin fine-tuning has to learn."""
        self.twin = twin
        self.inventory = inventory
        self._fault = fault
        self._actual = fault.apply(inventory) if fault else inventory
        self._bias = np.array([float((twin_bias or {}).get(c, 0.0)) for c in twin.cells])

    def _actual_mean(self) -> np.ndarray:
        mean = self.twin.rsrp_grid(self._actual)[0].copy()
        on = mean > config.OFF_RSRP_DBM + 1
        return np.where(on, mean + self._bias, mean)

    def field_observation(self, n_ue: int = 400, seed: int = 12345) -> pd.DataFrame:
        """A drive-test style log: where each UE was, which cell served it, serving RSRP and SINR."""
        rng = np.random.default_rng(seed)
        mean = self._actual_mean()
        idx = rng.integers(0, self.twin.n, n_ue)
        noisy = mean[idx] + rng.normal(0, SHADOWING_STD_DB, (n_ue, mean.shape[1]))
        serving, srv_rsrp, sinr = kpi.link_state(noisy)
        df = pd.DataFrame({
            "ue_id": np.arange(n_ue),
            "grid_idx": idx,
            "lon": self.twin.grid.loc_x.values[idx],
            "lat": self.twin.grid.loc_y.values[idx],
            "serving_cell": np.array(self.twin.cells)[serving],
            "serving_rsrp_dbm": srv_rsrp,
            "sinr_db": sinr,
        })
        # Measurement report: RSRP of every cell the UE can hear (NaN = not detected). This is what lets a
        # cell that is off (never heard) be told apart from one that is merely tilted/turned away (heard weakly).
        for j, cid in enumerate(self.twin.cells):
            df[f"rsrp_c{cid}"] = np.where(noisy[:, j] >= config.DETECT_FLOOR_DBM, noisy[:, j], np.nan)
        return df

    def reveal_rsrp(self):
        """Per-cell RSRP of the real (faulted) network, for the reveal slide only."""
        return self._actual_mean()

    def truth(self) -> Optional[dict]:
        return asdict(self._fault) if self._fault else None

