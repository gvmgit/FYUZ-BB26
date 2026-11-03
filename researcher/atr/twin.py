"""RF digital twin: MAVERIC's per-cell Bayesian (Gaussian-process) models behind a settings -> RSRP API."""
import json
import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from . import config
from radp.digital_twin.rf.bayesian.bayesian_engine import BayesianDigitalTwin
from radp.digital_twin.utils.gis_tools import GISTools

log = logging.getLogger(__name__)

X_COLUMNS = ["cell_el_deg", "log_distance", "relative_bearing"]
# Same normalisation ranges MAVERIC's training service uses.
X_MIN = {"cell_el_deg": -10, "cell_lat": -90, "cell_lon": -180, "log_distance": 0, "relative_bearing": 0,
         "relative_tilt": -90, "relative_tilt_squared": 0}
X_MAX = {"cell_el_deg": 50, "cell_lat": 90, "cell_lon": 180, "log_distance": 5, "relative_bearing": 360,
         "relative_tilt": 180, "relative_tilt_squared": 32400}

TRAIN_TILTS = (4, 10, 16)
VALID_TILTS = (7, 13)
MODEL_FILE = config.MODELS / "twin_model_map.pickle"
REPORT_FILE = config.MODELS / "twin_report.json"

from .topology import Settings, _sim_dirs, default_inventory, load_grid, load_topology  # noqa: E402,F401


def build_model_map(per_cell_samples: int = 400, seed: int = 0, maxiter: int = 100) -> Dict[int, BayesianDigitalTwin]:
    topology = load_topology()
    sims = _sim_dirs()
    frames = [pd.read_csv(sims[t] / "full_data.csv",
                          usecols=["loc_x", "loc_y", "cell_id", "rsrp_dbm", "cell_el_deg"]) for t in TRAIN_TILTS]
    model_map = {}
    for cid in topology.cell_id:
        parts = [f[f.cell_id == cid].sample(per_cell_samples, random_state=seed) for f in frames]
        df = pd.concat(parts)
        train = pd.DataFrame({"cell_id": cid, "avg_rsrp": df.rsrp_dbm.values, "lon": df.loc_x.values,
                              "lat": df.loc_y.values, "cell_el_deg": df.cell_el_deg.values})
        data = next(iter(BayesianDigitalTwin.preprocess_ue_training_data(train, topology.copy()).values()))
        model = BayesianDigitalTwin(data_in=[data], x_columns=X_COLUMNS, y_columns=["avg_rsrp"],
                                    x_min=X_MIN, x_max=X_MAX)
        losses = model.train_distributed_gpmodel(maxiter=maxiter)
        model.predict_distributed_gpmodel([data.head(1)])  # prime the prediction cache, as MAVERIC does
        model_map[int(cid)] = model
        log.info("cell %s trained, final loss %.3f", cid, float(losses[losses != 0][-1]))
    return model_map


class RFTwin:
    def __init__(self, model_map: Dict[int, BayesianDigitalTwin]):
        self.models = model_map
        self.topology = load_topology().set_index("cell_id")
        self.cells = sorted(model_map)
        self.grid = load_grid()
        self.n = len(self.grid)
        self._geo = {}
        for cid in self.cells:
            t = self.topology.loc[cid]
            logd, brg = np.empty(self.n), np.empty(self.n)
            for i, (lon, lat) in enumerate(zip(self.grid.loc_x.values, self.grid.loc_y.values)):
                logd[i] = np.log(1 + 1000 * GISTools.dist((t.cell_lat, t.cell_lon), (lat, lon)))
                brg[i] = GISTools.convert_bearing_0_to_360(GISTools.get_bearing((lat, lon), (t.cell_lat, t.cell_lon)))
            self._geo[cid] = (logd, brg)
        # 3x3 zones by coordinate terciles, used to localise symptoms.
        xe = np.quantile(self.grid.loc_x, [1 / 3, 2 / 3])
        ye = np.quantile(self.grid.loc_y, [1 / 3, 2 / 3])
        self.zone = np.digitize(self.grid.loc_y, ye) * 3 + np.digitize(self.grid.loc_x, xe)
        self._cache: Dict[tuple, Tuple[np.ndarray, np.ndarray]] = {}

    @classmethod
    def load_or_build(cls, retrain: bool = False, model_file=None) -> "RFTwin":
        """Load the base twin (training it if missing), or a fine-tuned version via `model_file`."""
        if model_file:
            return cls(BayesianDigitalTwin.load_model_map_from_pickle(str(model_file)))
        if retrain or not MODEL_FILE.exists():
            log.info("training twin ...")
            model_map = build_model_map()
            BayesianDigitalTwin.save_model_map_to_pickle(model_file_path=str(MODEL_FILE), model_map=model_map)
        else:
            model_map = BayesianDigitalTwin.load_model_map_from_pickle(str(MODEL_FILE))
        return cls(model_map)

    # -- fine-tuning support -------------------------------------------------------------------------------------
    def save(self, path) -> None:
        BayesianDigitalTwin.save_model_map_to_pickle(model_file_path=str(path), model_map=self.models)

    def clone(self) -> "RFTwin":
        """Independent copy, so an update can be tried and thrown away."""
        import pickle
        return RFTwin(pickle.loads(pickle.dumps(self.models)))

    def features(self, cid: int, idx, tilt: float, az: float) -> pd.DataFrame:
        """Model inputs for grid points `idx` of one cell under the given tilt/azimuth."""
        logd, brg = self._geo[cid]
        idx = np.asarray(idx)
        return pd.DataFrame({"cell_el_deg": float(tilt), "log_distance": logd[idx],
                             "relative_bearing": (brg[idx] - az) % 360})

    def update_cell(self, cid: int, rows: pd.DataFrame) -> None:
        """Incrementally fit one cell's model on new measurements (columns: model inputs + avg_rsrp)."""
        self.models[cid].update_trained_gpmodel([rows.reset_index(drop=True)])
        self._cache = {k: v for k, v in self._cache.items() if k[0] != cid}

    def inventory(self, tilt: float = 10.0) -> Settings:
        """The documented network: every cell at `tilt`, azimuth as in the topology, all on."""
        return default_inventory(tilt)

    def _cell_grid(self, cid: int, tilt: float, az: float) -> Tuple[np.ndarray, np.ndarray]:
        key = (cid, round(float(tilt), 2), round(float(az) % 360, 1))
        if key not in self._cache:
            logd, brg = self._geo[cid]
            df = pd.DataFrame({"cell_el_deg": float(tilt), "log_distance": logd,
                               "relative_bearing": (brg - az) % 360})
            mean, std = self.models[cid].predict_distributed_gpmodel([df])
            self._cache[key] = (np.asarray(mean, dtype=float).reshape(-1), np.asarray(std, dtype=float).reshape(-1))
        return self._cache[key]

    def rsrp_grid(self, settings: Settings) -> Tuple[np.ndarray, np.ndarray]:
        """Predicted RSRP mean/std for every grid point and cell; shape (n_points, n_cells)."""
        mean = np.empty((self.n, len(self.cells)))
        std = np.zeros_like(mean)
        for j, cid in enumerate(self.cells):
            s = settings[cid]
            if not s.get("enabled", True):
                mean[:, j] = config.OFF_RSRP_DBM
                continue
            mean[:, j], std[:, j] = self._cell_grid(cid, s["tilt"], s["azimuth"])
            mean[:, j] = mean[:, j] + s.get("offset_db", 0.0)   # null-hypothesis test: "the twin is biased here"
        return mean, std

    def _point_features(self, cid: int, lon: np.ndarray, lat: np.ndarray):
        """log-distance and bearing (point -> cell, degrees 0-360) for arbitrary positions: the same formulas as MAVERIC's
        GISTools.dist / get_bearing, vectorised (checked against them in scripts/check_mobility.py)."""
        t = self.topology.loc[cid]
        p1, l1 = np.radians(lat), np.radians(lon)
        p2, l2 = np.radians(t.cell_lat), np.radians(t.cell_lon)
        a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin((l2 - l1) / 2) ** 2
        dist_km = 2 * GISTools.R * np.arcsin(np.sqrt(a))
        y = np.sin(l2 - l1) * np.cos(p2)
        x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(l2 - l1)
        return np.log(1 + 1000 * dist_km), (np.degrees(np.arctan2(y, x)) + 360) % 360

    def rsrp_cell_points(self, cid: int, tilt: float, az: float, lon, lat) -> np.ndarray:
        """Predicted mean RSRP of one cell at arbitrary positions (not just the grid), for moving UEs."""
        logd, brg = self._point_features(cid, np.asarray(lon, float), np.asarray(lat, float))
        df = pd.DataFrame({"cell_el_deg": float(tilt), "log_distance": logd, "relative_bearing": (brg - az) % 360})
        mean, _ = self.models[cid].predict_distributed_gpmodel([df])
        return np.asarray(mean, dtype=float).reshape(-1)

    def validate(self, write: bool = True) -> dict:
        """RMSE of the twin against simulator ground truth at tilts it was NOT trained on."""
        sims = _sim_dirs()
        report = {}
        for t in VALID_TILTS:
            f = pd.read_csv(sims[t] / "full_data.csv", usecols=["loc_x", "loc_y", "cell_id", "rsrp_dbm"])
            key = {(round(x, 7), round(y, 7)): i for i, (x, y) in enumerate(zip(self.grid.loc_x, self.grid.loc_y))}
            per_cell = {}
            for j, cid in enumerate(self.cells):
                sub = f[f.cell_id == cid]
                idx = np.array([key.get((round(x, 7), round(y, 7)), -1) for x, y in zip(sub.loc_x, sub.loc_y)])
                ok = idx >= 0
                pred, _ = self._cell_grid(cid, t, self.topology.loc[cid].cell_az_deg)
                per_cell[int(cid)] = round(float(np.sqrt(np.mean((pred[idx[ok]] - sub.rsrp_dbm.values[ok]) ** 2))), 2)
            report[f"tilt_{t}"] = per_cell
        if write:
            REPORT_FILE.write_text(json.dumps(report, indent=2))
        return report

