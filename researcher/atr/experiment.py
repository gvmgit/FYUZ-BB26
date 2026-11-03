"""run_experiment: change settings in the twin, compare with the field log, report KPIs.

The score is a discrepancy between what the twin predicts under the tested settings and what the field's
measurement reports contain:
    score = RMSE of RSRP over cells heard in both (dB)
          + 10 x serving-cell mismatch rate
          + 15 x detection mismatch rate (a cell heard in one but not the other)
A setting that reproduces the field better has a lower score. The twin's mean prediction is used (its predictive
std is 2-6 dB, which would drown the signal); seeds vary the bootstrap resample of field UEs.
"""
import itertools
import time
from typing import List

import numpy as np
import pandas as pd

from . import config, kpi
from .heatmap import render_best_server
from .twin import RFTwin, Settings

FIELDS = ("tilt", "azimuth", "enabled", "offset_db")
# KPI the what-if can optimise, and which direction is better (+1 higher, -1 lower). rlf_risk_pct is a PROXY (best
# RSRP below -115 dBm or SINR below 0 dB), and mean_sinr_db a full-load proxy computed from RSRP; neither is MAVERIC's
# own handover/RLF simulation.
KPI_METRICS = {"coverage_pct": 1, "mean_best_rsrp": 1, "mean_sinr_db": 1, "rlf_risk_pct": -1}
SERVING_WEIGHT, DETECT_WEIGHT = 10.0, 15.0


class ExperimentRunner:
    def __init__(self, twin: RFTwin, inventory: Settings, field_obs: pd.DataFrame):
        self.twin = twin
        self.inventory = inventory
        self.obs = field_obs.reset_index(drop=True)
        self.count = 0
        self.id_prefix = ""   # set per container on Lambda so run ids stay unique across containers
        self._cell_pos = {c: j for j, c in enumerate(twin.cells)}
        self._obs_srv = self.obs.serving_cell.map(self._cell_pos).values
        self._obs_rsrp = self.obs.serving_rsrp_dbm.values
        self._obs_mat = self.obs[[f"rsrp_c{c}" for c in twin.cells]].values   # NaN = not detected
        self._obs_zone = twin.zone[self.obs.grid_idx.values]

    # -- configuration -------------------------------------------------------------------------------------
    def settings_for(self, cfg: dict) -> Settings:
        out = {c: dict(s) for c, s in self.inventory.items()}
        for cid, change in (cfg.get("cells") or {}).items():
            out[int(cid)].update({k: v for k, v in change.items() if k in FIELDS})
        return out

    def _sim(self):
        if getattr(self, "_sim_instrument", None) is None:
            from .instrument_b import SimInstrument
            self._sim_instrument = SimInstrument(self.twin)
        return self._sim_instrument

    def validate(self, cfg: dict) -> None:
        """Safety boundary + methodology: valid cells, tilt in range, at most one setting changed."""
        if cfg.get("instrument", "twin") not in ("twin", "simulator"):
            raise ValueError(f"unknown instrument '{cfg['instrument']}' (allowed: twin, simulator)")
        if cfg.get("objective", "field") not in ("field", "kpi", "mobility"):
            raise ValueError(f"unknown objective '{cfg['objective']}' (allowed: field, kpi, mobility)")
        if not 0.0 <= float(cfg.get("twin_error_db", 0.0)) <= 4.0:
            raise ValueError("twin_error_db must be between 0 and 4")
        changed = 0
        if cfg.get("objective") == "mobility":
            from . import mobility as mob
            base_h, base_t = self._baseline_rule(cfg)
            hyst, ttt = float(cfg.get("hyst", base_h)), cfg.get("ttt", base_t)
            for name, v in (("hyst", hyst), ("baseline hyst", base_h)):
                if not 0.0 <= v <= 10.0:
                    raise ValueError(f"{name} {v} outside the safe range 0-10 dB")
            for name, v in (("ttt", ttt), ("baseline ttt", base_t)):
                if int(v) != v or not 2 <= int(v) <= 8:
                    raise ValueError(f"{name} {v} must be a whole number of ticks from 2 to 8 (MAVERIC's attachment needs at least 2)")
            if not 1 <= int(cfg.get("n_ue", 30)) <= 200 or not 5 <= int(cfg.get("n_ticks", 60)) <= 300:
                raise ValueError("n_ue must be 1-200 and n_ticks 5-300")
            if not 0.0 <= float(cfg.get("shadowing_db", mob.DEFAULT_SHADOWING_DB)) <= 8.0:
                raise ValueError("shadowing_db must be between 0 and 8")
            changed += (hyst != base_h) + (int(ttt) != int(base_t))   # each handover knob, against the documented rule
        for cid, change in (cfg.get("cells") or {}).items():
            if int(cid) not in self._cell_pos:
                raise ValueError(f"unknown cell {cid}")
            for k, v in change.items():
                if k not in FIELDS:
                    raise ValueError(f"cannot change '{k}' (allowed: {FIELDS})")
                if k == "tilt" and not config.TILT_RANGE[0] <= float(v) <= config.TILT_RANGE[1]:
                    raise ValueError(f"tilt {v} outside safe range {config.TILT_RANGE}")
                if k == "azimuth" and not 0 <= float(v) < 360:
                    raise ValueError(f"azimuth {v} outside [0, 360)")
                if k == "offset_db" and not config.OFFSET_RANGE[0] <= float(v) <= config.OFFSET_RANGE[1]:
                    raise ValueError(f"offset_db {v} outside {config.OFFSET_RANGE}")
                if v != self.inventory[int(cid)].get(k, 0.0):
                    changed += 1
        if changed > 1:
            raise ValueError(f"{changed} settings changed; experiments must change one setting at a time")

    # -- the tool ------------------------------------------------------------------------------------------
    def run(self, config_: dict, seeds: List[int]) -> dict:
        t0 = time.time()
        settings = self.settings_for(config_)
        instrument = config_.get("instrument", "twin")
        if config_.get("objective") == "kpi":
            return self._run_kpi(config_, seeds, settings, instrument, t0)
        if config_.get("objective") == "mobility":
            return self._run_mobility(config_, seeds, settings, instrument, t0)
        if instrument == "simulator":
            mean = self._sim().matrix(settings, self.inventory)
        else:
            mean, _ = self.twin.rsrp_grid(settings)
        n_obs = len(self.obs)
        per_seed, zone_rmse = [], {z: [] for z in range(9)}
        for seed in seeds:
            rng = np.random.default_rng(seed)
            boot = rng.integers(0, n_obs, n_obs)
            pred = mean[self.obs.grid_idx.values[boot]]
            obs = self._obs_mat[boot]
            heard_obs, heard_pred = ~np.isnan(obs), pred >= config.DETECT_FLOOR_DBM
            both = heard_obs & heard_pred
            sq = np.where(both, (pred - np.nan_to_num(obs)) ** 2, np.nan)
            rmse = float(np.sqrt(np.nanmean(sq))) if both.any() else 30.0
            detect_mismatch = float(np.mean(heard_obs != heard_pred))
            serving, srv_rsrp, sinr = kpi.link_state(pred)
            mismatch = float(np.mean(serving != self._obs_srv[boot]))
            per_seed.append({"seed": seed,
                             "score": rmse + SERVING_WEIGHT * mismatch + DETECT_WEIGHT * detect_mismatch,
                             "rmse_db": rmse, "serving_mismatch": mismatch, "detect_mismatch": detect_mismatch,
                             "coverage_pct": kpi.coverage_pct(srv_rsrp),
                             "rlf_risk_pct": kpi.rlf_risk_pct(srv_rsrp, sinr)})
            zb = self._obs_zone[boot]
            for z in range(9):
                m = (zb == z)[:, None] & both
                if m.any():
                    zone_rmse[z].append(float(np.sqrt(np.mean(sq[m]))))
        self.count += 1
        run_id = f"{self.id_prefix}exp{self.count:03d}"
        result = {
            "run_id": run_id,
            "name": config_.get("name", run_id),
            "instrument": instrument,
            "config": config_.get("cells") or {},
            "seeds": list(seeds),
            "per_seed": per_seed,
            "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0] if k != "seed"},
            "zones": {f"Z{z}": round(float(np.mean(v)), 2) for z, v in zone_rmse.items() if v},
            "seconds": round(time.time() - t0, 2),
        }
        if config_.get("diagnostics"):
            result["diagnostics"] = self._diagnostics(mean)
        if config_.get("render"):
            path = config.HEATMAPS / f"{run_id}.png"
            render_best_server(self.twin, mean, path, f"{result['name']} (twin, best-server RSRP)")
            result["heatmap"] = str(path)
        return result

    @staticmethod
    def _baseline_rule(cfg: dict):
        """The documented handover rule experiments are measured against (hysteresis dB, time-to-trigger ticks). The documented
        network has no handover-rule inventory, so the default is an assumption; pass `baseline_rule` to state the real one."""
        from . import mobility as mob
        b = cfg.get("baseline_rule") or {}
        return float(b.get("hyst", mob.HYST_DB)), b.get("ttt", mob.TTT_TICKS)

    def _mobility_lab(self, n_ue: int, n_ticks: int):
        labs = getattr(self, "_mob_labs", None)
        if labs is None:
            labs = self._mob_labs = {}
        if (n_ue, n_ticks) not in labs:
            from .mobility import MobilityLab
            labs[(n_ue, n_ticks)] = MobilityLab(self.twin, self.inventory, n_ue, n_ticks)
        return labs[(n_ue, n_ticks)]

    def _run_mobility(self, config_: dict, seeds: List[int], settings: Settings, instrument: str, t0: float) -> dict:
        """Handover / radio-link-failure what-if for moving UEs (MAVERIC's mobility generator, attachment rule and MRO
        counters; see atr/mobility.py). Seeds vary the UE tracks and the shadowing; the same seed gives the same UEs and the
        same signal fluctuation, so a baseline and a treatment are paired. `hyst`/`ttt` are the handover rule's own knobs.
        Score (lower is better) = interruption seconds per UE-minute with MAVERIC's MRO weights (50 ms per handover, 1 s per
        failure tick)."""
        if instrument != "twin":
            raise ValueError("the second instrument (simulator tables) has no mobility data")
        from . import mobility as mob
        n_ue, n_ticks = int(config_.get("n_ue", 30)), int(config_.get("n_ticks", 60))
        base_h, base_t = self._baseline_rule(config_)
        hyst, ttt = float(config_.get("hyst", base_h)), int(config_.get("ttt", base_t))
        shadow = float(config_.get("shadowing_db", mob.DEFAULT_SHADOWING_DB))
        twin_err = float(config_.get("twin_error_db", 0.0))
        lab = self._mobility_lab(n_ue, n_ticks)
        minutes = n_ue * n_ticks / 60.0
        per_seed, per_cell = [], {"handovers_out": {}, "rlf": {}, "time_share": {}}
        for seed in seeds:
            k = lab.kpis(lab.attach(seed, settings, hyst=hyst, ttt=ttt, shadowing_db=shadow, twin_error_db=twin_err))
            cost = (k["handovers"] * 0.05 + k["rlf"] * 1.0) / minutes
            per_seed.append({"seed": seed, "handovers_per_ue_min": k["handovers_per_ue_min"],
                             "ping_pongs_per_ue_min": k["ping_pongs"] / minutes, "rlf_per_ue_min": k["rlf_per_ue_min"],
                             "rlf_ue_share_pct": k["rlf_ue_share_pct"], "mro_metric": k["mro_metric"],
                             "interruption_s_per_ue_min": cost, "score": cost})
            for name, key in (("handovers_out", "handovers_out_by_cell"), ("rlf", "rlf_by_serving_cell"),
                              ("time_share", "time_share_by_cell")):
                for c, v in k[key].items():
                    per_cell[name].setdefault(c, []).append(v)
        self.count += 1
        run_id = f"{self.id_prefix}exp{self.count:03d}"
        return {"run_id": run_id, "name": config_.get("name", run_id), "instrument": instrument, "objective": "mobility",
                "handover_rule": {"hyst_db": hyst, "ttt_ticks": ttt}, "shadowing_db": shadow, "twin_error_db": twin_err,
                "n_ue": n_ue,
                "n_ticks": n_ticks, "config": config_.get("cells") or {}, "seeds": list(seeds), "per_seed": per_seed,
                "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0] if k != "seed"},
                "per_cell_mean": {n: {int(c): float(np.mean(v)) for c, v in d.items()} for n, d in per_cell.items()},
                "seconds": round(time.time() - t0, 2)}

    def _run_kpi(self, config_: dict, seeds: List[int], settings: Settings, instrument: str, t0: float) -> dict:
        """What-if on network KPIs, no field log: how do coverage and signal quality change under these settings?

        Seeds vary the sampled UE locations AND, when `twin_error_db` > 0, a random per-cell dB offset standing in for
        the twin's own error. The draws depend only on the seed, so a baseline and a treatment run with the same seeds
        see identical UEs and identical twin error: the comparison is paired, and its spread includes model uncertainty.
        """
        sigma = float(config_.get("twin_error_db", 0.0))
        n_ue = int(config_.get("n_ue", 1200))
        metric = config_.get("metric", "coverage_pct")
        if metric not in KPI_METRICS:
            raise ValueError(f"metric must be one of {KPI_METRICS}")
        base = self._sim().matrix(settings, self.inventory) if instrument == "simulator" \
            else self.twin.rsrp_grid(settings)[0]
        on = base.max(axis=0) > config.OFF_RSRP_DBM + 1       # per cell: radiating or not
        per_seed = []
        for seed in seeds:
            rng = np.random.default_rng(seed)
            idx = rng.integers(0, self.twin.n, n_ue)
            offset = rng.normal(0.0, sigma, base.shape[1]) if sigma > 0 else np.zeros(base.shape[1])
            m = base[idx] + np.where(on, offset, 0.0)
            serving, srv_rsrp, sinr = kpi.link_state(m)
            row = {"seed": seed, "coverage_pct": kpi.coverage_pct(srv_rsrp), "mean_best_rsrp": float(srv_rsrp.mean()),
                   "mean_sinr_db": float(sinr.mean()), "rlf_risk_pct": kpi.rlf_risk_pct(srv_rsrp, sinr)}
            row["score"] = -KPI_METRICS[metric] * row[metric]     # lower is better, as everywhere else
            per_seed.append(row)
        self.count += 1
        run_id = f"{self.id_prefix}exp{self.count:03d}"
        result = {"run_id": run_id, "name": config_.get("name", run_id), "instrument": instrument,
                  "objective": "kpi", "metric": metric, "twin_error_db": sigma, "config": config_.get("cells") or {},
                  "seeds": list(seeds), "per_seed": per_seed,
                  "mean": {k: float(np.mean([p[k] for p in per_seed])) for k in per_seed[0] if k != "seed"},
                  "seconds": round(time.time() - t0, 2)}
        if config_.get("render"):
            path = config.HEATMAPS / f"{run_id}.png"
            render_best_server(self.twin, base, path, f"{result['name']} ({instrument}, best-server RSRP)")
            result["heatmap"] = str(path)
        return result

    def _diagnostics(self, mean: np.ndarray) -> dict:
        """Seed-free comparison of field vs twin used by the Scientist to localise the symptom."""
        cells = self.twin.cells
        pm = mean[self.obs.grid_idx.values]
        pred_srv, pred_best = pm.argmax(1), pm.max(1)
        heard_obs, heard_pred = ~np.isnan(self._obs_mat), pm >= config.DETECT_FLOOR_DBM
        both = heard_obs & heard_pred
        diff = np.where(both, np.nan_to_num(self._obs_mat) - pm, np.nan)
        diag = {"cells": {}, "confusion": {}}
        for j, cid in enumerate(cells):
            col = diff[:, j]
            diag["cells"][int(cid)] = {
                "obs_share": round(float(np.mean(self._obs_srv == j)), 3),
                "pred_share": round(float(np.mean(pred_srv == j)), 3),
                "obs_heard": round(float(heard_obs[:, j].mean()), 3),
                "pred_heard": round(float(heard_pred[:, j].mean()), 3),
                "mean_abs_resid_db": round(float(np.nanmean(np.abs(col))), 2) if both[:, j].any() else None,
                "mean_resid_db": round(float(np.nanmean(col)), 2) if both[:, j].any() else None,
            }
        for a, b in itertools.product(range(len(cells)), repeat=2):
            n = int(np.sum((self._obs_srv == a) & (pred_srv == b)))
            if a != b and n:
                diag["confusion"][f"obs{cells[a]}->pred{cells[b]}"] = n
        weak = {}
        for z in range(9):
            m = self._obs_zone == z
            if m.sum() >= 10:
                weak[f"Z{z}"] = {"n_ue": int(m.sum()),
                                 "field_weak_pct": round(float(np.mean(self._obs_rsrp[m] < config.WEAK_COVERAGE_DBM) * 100), 1),
                                 "expected_weak_pct": round(float(np.mean(pred_best[m] < config.WEAK_COVERAGE_DBM) * 100), 1),
                                 "field_mean_rsrp": round(float(self._obs_rsrp[m].mean()), 1),
                                 "expected_mean_rsrp": round(float(pred_best[m].mean()), 1)}
        diag["weak_coverage_by_zone"] = weak
        return diag
