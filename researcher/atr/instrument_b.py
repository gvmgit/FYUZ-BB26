"""Second instrument: MAVERIC's simulator ground-truth RSRP tables, read directly.

The twin is a Gaussian-process fit to some of these tables. Replicating a finding against the tables themselves removes
the fitted model from the loop, so an effect that exists only because of how the twin interpolates will not replicate.

Limits, stated plainly: the simulator output exists for one tilt at a time (4, 7, 10, 13, 16 degrees, all cells together),
at the inventory azimuths, with every cell on. So it can check tilt claims whose value is within 1.5 degrees of a simulated
tilt. It cannot check azimuth or cell-off claims, and the twin was trained on tilts 4, 10 and 16 (7 and 13 were held out),
so for those three tilts the instrument is independent of the twin's fit but not of its training data.
"""
import numpy as np
import pandas as pd

from .topology import _sim_dirs

MAX_TILT_GAP = 1.5


class SimInstrument:
    def __init__(self, twin):
        self.twin = twin
        sims = _sim_dirs()
        self.tilts = sorted(sims)
        key = {(round(x, 7), round(y, 7)): i for i, (x, y) in enumerate(zip(twin.grid.loc_x, twin.grid.loc_y))}
        self.tables = {}
        for tilt, d in sims.items():
            f = pd.read_csv(d / "full_data.csv", usecols=["loc_x", "loc_y", "cell_id", "rsrp_dbm"])
            for cid in twin.cells:
                sub = f[f.cell_id == cid]
                idx = np.array([key.get((round(x, 7), round(y, 7)), -1) for x, y in zip(sub.loc_x, sub.loc_y)])
                ok = idx >= 0
                col = np.full(twin.n, np.nan)
                col[idx[ok]] = sub.rsrp_dbm.values[ok]
                self.tables[(tilt, cid)] = col

    def nearest_tilt(self, tilt: float) -> int:
        t = min(self.tilts, key=lambda s: abs(s - tilt))
        if abs(t - tilt) > MAX_TILT_GAP:
            raise ValueError(f"no simulator table within {MAX_TILT_GAP} deg of tilt {tilt:g} (have {self.tilts})")
        return t

    def matrix(self, settings: dict, inventory: dict) -> np.ndarray:
        """Per-grid-point, per-cell RSRP read from the simulator tables for these settings."""
        out = np.empty((self.twin.n, len(self.twin.cells)))
        for j, cid in enumerate(self.twin.cells):
            s, inv = settings[cid], inventory[cid]
            if not s.get("enabled", True) or abs(s["azimuth"] - inv["azimuth"]) > 1e-6 or s.get("offset_db", 0.0):
                raise ValueError("the second instrument (simulator tables) covers tilt changes only")
            tilt = self.nearest_tilt(s["tilt"])
            col = self.tables[(tilt, cid)].copy()
            gaps = np.isnan(col)
            if gaps.any():    # a few grid points are absent from the simulator output; use the twin there
                col[gaps] = self.twin._cell_grid(cid, tilt, inv["azimuth"])[0][gaps]
            out[:, j] = col
        return out
