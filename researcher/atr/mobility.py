"""Mobility and handover: moving UEs, the twin's signal along their tracks, and handover / radio-link-failure KPIs.

Built on MAVERIC's own pieces, not re-invented:
  tracks      radp UETracksGenerator (Gauss-Markov, stationary / pedestrian / cyclist / car classes)
  attachment  radp.digital_twin.utils.cell_selection.perform_attachment_hyst_ttt (hysteresis, time-to-trigger, SINR failure
              threshold) - the same function MAVERIC's MRO notebook uses
  counters    the MRO app's _count_handovers, _count_rlf and calculate_mro_metric

What is ours: choosing speeds and noise for THIS area (MAVERIC's examples are for a different scale, see SPEED_MS), asking the
twin for signal at arbitrary positions along each track, and computing per-cell KPIs the counters do not give (where failures
happen, ping-pongs). Units: one tick is one second (the assumption inside calculate_mro_metric); the Gauss-Markov grid is 100 x 100
units over the twin's bounding box, so one grid unit is derived from that box, about 16 m here.

KPIs are SIMULATED handovers and failures on the twin's predicted signal, with MAVERIC's default failure rule (SINR below
RLF_THRESHOLD) and noise constant. They are not measurements from a real network.
"""
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config

CLASS_MIX = {"stationary": 0.1, "pedestrian": 0.4, "cyclist": 0.2, "car": 0.3}
SPEED_MS = {"stationary": 0.0, "pedestrian": 1.4, "cyclist": 5.0, "car": 15.0}
SPEED_SPREAD_MS = {"stationary": 0.0, "pedestrian": 0.4, "cyclist": 1.5, "car": 4.0}
ALPHA, VARIANCE = 0.8, 0.08          # Gauss-Markov memory, and noise in grid units per tick
HYST_DB, TTT_TICKS = 3.0, 3          # handover rule defaults (MAVERIC's MRO tunes these two)
CARRIER_MHZ = 1800.0
PING_PONG_TICKS = 5
SHADOW_RHO = 0.7                     # tick-to-tick correlation of the optional shadowing (assumption)
DEFAULT_SHADOWING_DB = 2.0           # used by run_experiment: matches the 2 dB measurement noise elsewhere (assumption)


def _constants():
    """MAVERIC's noise and failure-threshold constants: light, no app imports (safe inside the Lambda image)."""
    from radp.digital_twin.utils.constants import LATENT_BACKGROUND_NOISE_DB, RLF_THRESHOLD
    return LATENT_BACKGROUND_NOISE_DB, RLF_THRESHOLD


def _maveric():
    """MAVERIC's attachment function and MRO counters, imported lazily (they pull in notebooks.radp_library). Only needed
    to cross-check the fast path, never for experiments."""
    from apps.mobility_robustness_optimization.mobility_robustness_optimization import (
        _count_handovers, _count_rlf, calculate_mro_metric)
    from radp.digital_twin.utils.cell_selection import perform_attachment_hyst_ttt
    from radp.digital_twin.utils.constants import LATENT_BACKGROUND_NOISE_DB, RLF_THRESHOLD
    return perform_attachment_hyst_ttt, _count_handovers, _count_rlf, calculate_mro_metric, \
        LATENT_BACKGROUND_NOISE_DB, RLF_THRESHOLD


def attach_arrays(P: np.ndarray, S: np.ndarray, cell_ids: np.ndarray, hyst: float, ttt: int, rlf: float) -> pd.DataFrame:
    """Vectorised equivalent of MAVERIC's perform_attachment_hyst_ttt, including its quirks.

    P, S: (ticks, ues, cells) RSRP (dBm) and SINR (dB). Per tick, as MAVERIC does it:
      - for the first ttt-1 ticks a UE simply takes the strongest cell;
      - afterwards: (1) _check_hyst: the strongest cell that beats the previous cell's CURRENT power by `hyst`, else keep the
        previous cell (a UE coming off RLF takes the strongest); (2) _check_ttt: switch only if the last `ttt` such choices
        agree, else keep the previous cell (strongest after RLF); (3) _check_hyst_in_current_tick: revert a switch that does not
        beat the previous cell by `hyst` - MAVERIC reverts the cell and power but leaves the SINR of the cell it reverted from;
      - always last, _check_rlf_threshold on that SINR: below the threshold, move to the best-SINR cell if any reaches it,
        else the UE is in RLF (cell 'RLF', power and SINR -inf).
    """
    n_t, n_u, _ = P.shape
    u = np.arange(n_u)
    cell = np.empty((n_t, n_u), dtype=object)
    power, sinr = np.empty((n_t, n_u)), np.empty((n_t, n_u))
    history, past = [], np.full(n_u, -1)          # past: previous attached cell index, -1 = RLF / none
    for t in range(n_t):
        Pt, St = P[t], S[t]
        if ttt - 1 > len(history):
            att = Pt.argmax(1)
            history.append(att)
            att_power, att_sinr = Pt[u, att], St[u, att]
        else:
            p_past = np.where(past >= 0, Pt[u, np.maximum(past, 0)], -999.0)
            beats = (Pt - p_past[:, None]) >= hyst
            strong = np.where(beats.any(1), np.where(beats, Pt, -np.inf).argmax(1), past)
            history.append(strong)
            same = np.all([h == history[0] for h in history], axis=0)
            att = np.where(same, history[0], np.where(past >= 0, past, Pt.argmax(1)))
            att_power, att_sinr = Pt[u, att], St[u, att]
            switched = att != past
            prev_power = np.where(past >= 0, Pt[u, np.maximum(past, 0)], -np.inf)
            revert = switched & (att_power < prev_power + hyst)
            att_power = np.where(revert, prev_power, att_power)      # sinr deliberately left as the switched-to cell's
            att = np.where(revert, past, att)
            if len(history) == ttt:
                history.pop(0)
        low = att_sinr < rlf
        best = St.argmax(1)
        can_recover = low & (St[u, best] >= rlf)
        fail = low & ~can_recover
        att = np.where(can_recover, best, att)
        att_power = np.where(can_recover, Pt[u, best], att_power)
        att_sinr = np.where(can_recover, St[u, best], att_sinr)
        cell[t] = np.where(fail, "RLF", cell_ids[np.maximum(att, 0)])
        power[t] = np.where(fail, -np.inf, att_power)
        sinr[t] = np.where(fail, -np.inf, att_sinr)
        past = np.where(fail, -1, att)
    return pd.DataFrame({"ue_id": np.tile(u, n_t), "cell_id": cell.reshape(-1), "cell_rxpower_dbm": power.reshape(-1),
                         "tick": np.repeat(np.arange(n_t), n_u), "sinr_db": sinr.reshape(-1)})


def fast_kpis(attached: pd.DataFrame, n_ue: int, n_ticks: int) -> dict:
    """The MRO counters, vectorised: handovers (cell changes between consecutive ticks, not into RLF; leaving RLF counts),
    RLF ticks, and calculate_mro_metric. Equivalent to MAVERIC's (checked)."""
    a = attached.sort_values(["ue_id", "tick"])
    cells = a.cell_id.values.reshape(n_ue, n_ticks)
    changed = cells[:, 1:] != cells[:, :-1]
    ho = int(np.sum(changed & (cells[:, 1:] != "RLF")))
    rlf = int(np.sum(cells == "RLF"))
    return {"handovers": ho, "rlf": rlf, "mro_metric": float(n_ticks * 1.0 - (ho * 0.05 + rlf * 1.0))}


class MobilityLab:
    def __init__(self, twin, inventory: dict, n_ue: int = 30, n_ticks: int = 60, mix: Optional[Dict[str, float]] = None):
        self.twin, self.inventory = twin, inventory
        self.n_ue, self.n_ticks = n_ue, n_ticks
        self.mix = mix or CLASS_MIX
        g = twin.grid
        self.lon_min, self.lon_max, self.lat_min, self.lat_max = g.loc_x.min(), g.loc_x.max(), g.loc_y.min(), g.loc_y.max()
        mid_lat = np.radians((self.lat_min + self.lat_max) / 2)
        width_m = (self.lon_max - self.lon_min) * 111_320 * np.cos(mid_lat)
        height_m = (self.lat_max - self.lat_min) * 110_540
        self.unit_m = float(np.mean([width_m, height_m]) / 100)
        self._tracks, self._rsrp = {}, {}

    # -- tracks -------------------------------------------------------------------------------------------------------
    def tracks(self, seed: int) -> pd.DataFrame:
        """ue_id, tick, lon, lat, mobility_class for `n_ue` UEs over `n_ticks`, from MAVERIC's generator."""
        if seed in self._tracks:
            return self._tracks[seed]
        from radp.digital_twin.mobility.ue_tracks import MobilityClass, UETracksGenerator
        from radp.digital_twin.utils.gis_tools import GISTools
        classes = [MobilityClass(k) for k in self.mix]
        units = lambda ms: ms / self.unit_m          # noqa: E731  (metres per second -> grid units per 1 s tick)
        gen = UETracksGenerator(
            rng=np.random.default_rng(seed), num_ticks=self.n_ticks, num_UEs=self.n_ue, alpha=ALPHA, variance=VARIANCE,
            lon_x_dims=100, lon_y_dims=100, min_lat=self.lat_min, max_lat=self.lat_max, min_lon=self.lon_min,
            max_lon=self.lon_max,
            mobility_class_distribution={c: self.mix[c.value] for c in classes},
            mobility_class_velocities={c: units(SPEED_MS[c.value]) for c in classes},
            mobility_class_velocity_variances={c: units(SPEED_SPREAD_MS[c.value]) for c in classes})
        batch = next(gen.generate())
        label = np.concatenate([[c.value] * int(gen.num_users_per_mobility_class[c]) for c in classes])
        rows = []
        for tick, xy in enumerate(batch):
            ll = GISTools.converting_xy_points_into_lonlat_pairs(
                xy_points=xy, x_dim=100, y_dim=100, min_longitude=self.lon_min, max_longitude=self.lon_max,
                min_latitude=self.lat_min, max_latitude=self.lat_max)
            rows.append(pd.DataFrame({"ue_id": np.arange(self.n_ue), "tick": tick, "lon": [p[0] for p in ll],
                                      "lat": [p[1] for p in ll], "mobility_class": label}))
        gen.close()
        self._tracks[seed] = pd.concat(rows, ignore_index=True)
        return self._tracks[seed]

    # -- signal along the tracks --------------------------------------------------------------------------------------
    def shadowing(self, seed: int, sigma_db: float, rho: float = SHADOW_RHO) -> np.ndarray:
        """Slow signal fluctuation per (ue, tick, cell): AR(1) in time, std `sigma_db`, rows follow tracks(seed). OUR
        assumption, not MAVERIC's: its MRO flow uses the twin's smooth mean prediction, which produces almost no ping-pong or
        failure. Zero by default."""
        rng = np.random.default_rng(10_000 + seed)
        eps = rng.standard_normal((self.n_ticks, self.n_ue, len(self.twin.cells)))
        s = np.empty_like(eps)
        s[0] = eps[0]
        for t in range(1, self.n_ticks):
            s[t] = rho * s[t - 1] + np.sqrt(1 - rho ** 2) * eps[t]
        return (sigma_db * s).reshape(-1, len(self.twin.cells))

    def twin_error(self, seed: int, sigma_db: float) -> np.ndarray:
        """One random dB offset per cell, fixed for a seed: stands in for the twin's own error (same idea as the KPI objective).
        Depends only on the seed, so a baseline and a treatment run with the same seed see the same twin error."""
        return np.random.default_rng(20_000 + seed).normal(0.0, sigma_db, len(self.twin.cells))

    def rsrp(self, seed: int, settings: dict, shadowing_db: float = 0.0, twin_error_db: float = 0.0) -> np.ndarray:
        """Predicted RSRP of every cell at every (ue, tick) point; rows follow tracks(seed). Per-cell results are cached by
        (track, cell, tilt, azimuth), so changing one cell's setting recomputes one column."""
        tr = self.tracks(seed)
        out = np.empty((len(tr), len(self.twin.cells)))
        for j, cid in enumerate(self.twin.cells):
            s = settings[cid]
            if not s.get("enabled", True):
                out[:, j] = config.OFF_RSRP_DBM
                continue
            key = (seed, cid, round(float(s["tilt"]), 2), round(float(s["azimuth"]) % 360, 1))
            if key not in self._rsrp:
                self._rsrp[key] = self.twin.rsrp_cell_points(cid, s["tilt"], s["azimuth"], tr.lon.values, tr.lat.values)
            out[:, j] = self._rsrp[key] + s.get("offset_db", 0.0)
        on = out.max(axis=0) > config.OFF_RSRP_DBM + 1
        if twin_error_db > 0:
            out = out + np.where(on, self.twin_error(seed, twin_error_db), 0.0)
        if shadowing_db > 0:
            out = out + np.where(on, self.shadowing(seed, shadowing_db), 0.0)
        return out

    def ue_data(self, seed: int, settings: dict, noise_db: Optional[float] = None, shadowing_db: float = 0.0,
                twin_error_db: float = 0.0) -> pd.DataFrame:
        """The table MAVERIC's attachment function takes: one row per (ue, tick, cell) with RSRP and SINR."""
        noise_db = _constants()[0] if noise_db is None else noise_db
        r = self.rsrp(seed, settings, shadowing_db, twin_error_db)
        lin = 10 ** (r / 10)
        sinr = r - 10 * np.log10(lin.sum(1, keepdims=True) - lin + 10 ** (noise_db / 10))
        tr, cells = self.tracks(seed), self.twin.cells
        df = pd.DataFrame({"ue_id": np.repeat(tr.ue_id.values, len(cells)), "tick": np.repeat(tr.tick.values, len(cells)),
                           "cell_id": np.tile(cells, len(tr)), "cell_rxpower_dbm": r.reshape(-1),
                           "cell_carrier_freq_mhz": CARRIER_MHZ, "sinr_db": sinr.reshape(-1)})
        return df.sort_values(["tick", "ue_id", "cell_id"]).reset_index(drop=True)

    # -- attachment and KPIs ------------------------------------------------------------------------------------------
    def attach(self, seed: int, settings: dict, hyst: float = HYST_DB, ttt: int = TTT_TICKS,
               rlf_threshold: Optional[float] = None, engine: str = "fast", shadowing_db: float = 0.0,
               twin_error_db: float = 0.0) -> pd.DataFrame:
        """Cell attachment over the whole simulation. engine='maveric' runs MAVERIC's own function (about 10 s for 30 UEs x
        60 ticks); 'fast' is a vectorised version proven identical to it (scripts/check_mobility.py)."""
        noise_db, rlf_default = _constants()
        thr = rlf_default if rlf_threshold is None else rlf_threshold
        if engine == "maveric":
            return _maveric()[0](self.ue_data(seed, settings, shadowing_db=shadowing_db, twin_error_db=twin_error_db),
                                 hyst, ttt, thr)
        r = self.rsrp(seed, settings, shadowing_db, twin_error_db)
        n_cells, n_pts = len(self.twin.cells), len(r)
        ticks = self.tracks(seed).tick.values
        order = np.argsort(ticks, kind="stable")
        P = r[order].reshape(self.n_ticks, self.n_ue, n_cells)
        lin = 10 ** (P / 10)
        noise = 10 ** (noise_db / 10)
        S = P - 10 * np.log10(lin.sum(2, keepdims=True) - lin + noise)
        return attach_arrays(P, S, np.array(self.twin.cells), hyst, ttt, thr)

    def kpis(self, attached: pd.DataFrame, engine: str = "fast") -> dict:
        """Handover / failure KPIs. Totals come from MAVERIC's MRO counters (engine='maveric') or a vectorised equivalent
        (checked identical); the per-cell view and ping-pongs are ours, the counters do not give them."""
        a = attached.sort_values(["ue_id", "tick"]).reset_index(drop=True)
        n_ue, n_t = self.n_ue, self.n_ticks
        if engine == "maveric":
            _, count_ho, count_rlf, mro, *_ = _maveric()
            ho, rlf, d = int(count_ho(attached)), int(count_rlf(attached)), float(mro(attached))
        else:
            t = fast_kpis(a, n_ue, n_t)
            ho, rlf, d = t["handovers"], t["rlf"], t["mro_metric"]
        cells = a.cell_id.map(lambda x: -1 if str(x) == "RLF" else int(float(x))).values.reshape(n_ue, n_t)
        pp = 0
        for u in range(n_ue):
            # a switch is a change between two consecutive NON-failed ticks; a failure breaks the sequence (it is counted
            # as a failure, and the re-attachment after it is not a ping-pong)
            sw = [(t, cells[u][t - 1], cells[u][t]) for t in range(1, n_t)
                  if cells[u][t] != -1 and cells[u][t - 1] != -1 and cells[u][t] != cells[u][t - 1]]
            pp += sum(1 for k in range(1, len(sw)) if sw[k][2] == sw[k - 1][1] and sw[k][0] - sw[k - 1][0] <= PING_PONG_TICKS)
        prev, cur = cells[:, :-1], cells[:, 1:]
        ho_out, rlf_by = {}, {}
        for c in self.twin.cells:
            ho_out[int(c)] = int(np.sum((prev == c) & (cur != c) & (cur != -1)))
            rlf_by[int(c)] = int(np.sum((prev == c) & (cur == -1)))
        minutes = n_ue * n_t / 60.0
        return {"handovers": ho, "rlf": rlf, "mro_metric": d, "handovers_per_ue_min": ho / minutes,
                "rlf_per_ue_min": rlf / minutes, "rlf_ue_share_pct": float(100 * np.mean((cells == -1).any(1))),
                "ping_pongs": int(pp), "rlf_by_serving_cell": rlf_by, "handovers_out_by_cell": ho_out,
                "time_share_by_cell": {int(c): float(np.mean(cells == c)) for c in self.twin.cells}}

    def run(self, seed: int, settings: dict, **kw) -> dict:
        return self.kpis(self.attach(seed, settings, **kw))
