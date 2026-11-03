"""Checks for the mobility and handover work: that it really is MAVERIC's logic, and what it can and cannot do.

Run from the researcher folder:  ..\.venv\Scripts\python.exe scripts\check_mobility.py
"""
import subprocess
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from atr import mobility as M, runtime  # noqa: E402
from atr.gateway import ToolError  # noqa: E402
from atr.topology import default_inventory  # noqa: E402
from atr.twin import RFTwin  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


twin, inv = RFTwin.load_or_build(), default_inventory()
attach_fn, count_ho, count_rlf, mro_metric, _, _ = M._maveric()


def norm(df):
    d = df.sort_values(["tick", "ue_id"]).reset_index(drop=True)
    cells = d.cell_id.map(lambda x: "RLF" if str(x) == "RLF" else str(int(float(x)))).values
    return cells, d.cell_rxpower_dbm.astype(float).values, d.sinr_db.astype(float).values


def identical(a, b):
    (ca, pa, sa), (cb, pb, sb) = norm(a), norm(b)
    return bool((ca == cb).all() and np.allclose(pa, pb, equal_nan=True) and np.allclose(sa, sb, equal_nan=True))


# ---- the twin at arbitrary positions ---------------------------------------------------------------------------------------
from radp.digital_twin.utils.gis_tools import GISTools  # noqa: E402

rng = np.random.default_rng(0)
lon = rng.uniform(twin.grid.loc_x.min(), twin.grid.loc_x.max(), 150)
lat = rng.uniform(twin.grid.loc_y.min(), twin.grid.loc_y.max(), 150)
cid = twin.cells[2]
t = twin.topology.loc[cid]
ld, br = twin._point_features(cid, lon, lat)
ld0 = np.array([np.log(1 + 1000 * GISTools.dist((t.cell_lat, t.cell_lon), (la, lo))) for lo, la in zip(lon, lat)])
br0 = np.array([GISTools.convert_bearing_0_to_360(GISTools.get_bearing((la, lo), (t.cell_lat, t.cell_lon))) for lo, la in zip(lon, lat)])
check("distance and bearing at arbitrary points equal MAVERIC's GISTools", np.abs(ld - ld0).max() < 1e-9 and np.abs(br - br0).max() < 1e-9)
idx = np.arange(0, twin.n, 50)
pt = twin.rsrp_cell_points(cid, 10, inv[cid]["azimuth"], twin.grid.loc_x.values[idx], twin.grid.loc_y.values[idx])
check("the twin's prediction at a point equals its grid prediction there",
      np.abs(pt - twin._cell_grid(cid, 10, inv[cid]["azimuth"])[0][idx]).max() < 0.01)

# ---- tracks ------------------------------------------------------------------------------------------------------------------
lab = M.MobilityLab(twin, inv, n_ue=30, n_ticks=60)
tr = lab.tracks(1)
check("tracks are deterministic per seed and differ across seeds",
      tr.equals(M.MobilityLab(twin, inv, 30, 60).tracks(1)) and not tr[["lon", "lat"]].equals(lab.tracks(2)[["lon", "lat"]]))
check("every UE has every tick, inside the twin's area",
      len(tr) == 30 * 60 and tr.groupby("ue_id").size().eq(60).all()
      and tr.lon.between(lab.lon_min - 1e-9, lab.lon_max + 1e-9).all() and tr.lat.between(lab.lat_min - 1e-9, lab.lat_max + 1e-9).all())
sp = {}
for u, g in tr.sort_values(["ue_id", "tick"]).groupby("ue_id"):
    dx = np.diff(g.lon.values) * 111320 * np.cos(np.radians(g.lat.values[:-1]))
    dy = np.diff(g.lat.values) * 110540
    sp.setdefault(g.mobility_class.iloc[0], []).append(np.hypot(dx, dy).mean())
check("realised speeds match the targets (measured in plain metres)",
      all(abs(np.mean(sp[c]) - M.SPEED_MS[c]) < max(0.8, 0.12 * M.SPEED_MS[c]) for c in sp if c != "stationary")
      and np.mean(sp.get("stationary", [0])) < 1.5, str({c: round(float(np.mean(v)), 1) for c, v in sp.items()}))

# ---- SINR, attachment and counters are MAVERIC's logic -----------------------------------------------------------------------
from apps.mobility_robustness_optimization.mobility_robustness_optimization import MobilityRobustnessOptimization  # noqa: E402

small = M.MobilityLab(twin, inv, n_ue=4, n_ticks=3)
ud = small.ue_data(1, inv)
ref = MobilityRobustnessOptimization._add_sinr_column(None, ud.drop(columns=["sinr_db"]))
check("the vectorised SINR equals MAVERIC's _add_sinr_column",
      np.allclose(ref.sort_values(["tick", "ue_id", "cell_id"]).sinr_db.values, ud.sinr_db.values, atol=1e-6))

lab10 = M.MobilityLab(twin, inv, n_ue=10, n_ticks=30)
ok = True
for hy, tt in ((3.0, 3), (0.5, 2), (6.0, 5)):
    for sh in (0.0, 4.0):
        ok &= identical(lab10.attach(1, inv, hyst=hy, ttt=tt, shadowing_db=sh, engine="maveric"),
                        lab10.attach(1, inv, hyst=hy, ttt=tt, shadowing_db=sh))
check("fast attachment is identical to MAVERIC's perform_attachment_hyst_ttt on real tracks (3 rules x 2 shadowing levels)", ok)

ok, fails, rlf_total = True, 0, 0
rng = np.random.default_rng(7)
cells = np.array([1, 2, 3, 4, 5])
for trial in range(8):
    P = np.clip(-100 + rng.normal(0, 6, (40, 8, 5)).cumsum(0) * 0.15 + rng.normal(0, 4, (40, 8, 5)), -140, -60)
    lin = 10 ** (P / 10)
    S = P - 10 * np.log10(lin.sum(2, keepdims=True) - lin + 1e-15) + rng.normal(0, 3, P.shape)
    hy, tt, thr = [(3.0, 3, -4), (0.5, 2, -4), (6.0, 4, -2), (1.0, 3, 0.0)][trial % 4]
    rows = [(u, t, int(cells[j]), P[t, u, j], 1800.0, S[t, u, j]) for t in range(40) for u in range(8) for j in range(5)]
    df = pd.DataFrame(rows, columns=["ue_id", "tick", "cell_id", "cell_rxpower_dbm", "cell_carrier_freq_mhz", "sinr_db"])
    a_ref, a_fast = attach_fn(df, hy, tt, thr), M.attach_arrays(P, S, cells, hy, tt, thr)
    ok &= identical(a_ref, a_fast)
    rlf_total += int((a_fast.cell_id.astype(str) == "RLF").sum())
    k_ref = (int(count_ho(a_ref)), int(count_rlf(a_ref)), float(mro_metric(a_ref)))
    k_fast = M.fast_kpis(a_fast, 8, 40)
    fails += k_ref != (k_fast["handovers"], k_fast["rlf"], k_fast["mro_metric"])
check("fast attachment is identical to MAVERIC's on 8 random signal sets that force handovers, reverts, failures and recovery",
      ok, f"{rlf_total} failure ticks exercised")
check("fast counters equal MAVERIC's _count_handovers, _count_rlf and calculate_mro_metric on those", fails == 0)

pp_lab = M.MobilityLab(twin, inv, n_ue=2, n_ticks=12)


def pp(seqs):
    rows = [(u, t, c, 0, 0) for u, s in enumerate(seqs) for t, c in enumerate(s)]
    return pp_lab.kpis(pd.DataFrame(rows, columns=["ue_id", "tick", "cell_id", "cell_rxpower_dbm", "sinr_db"]))["ping_pongs"]


A, B = 1, 2
check("a ping-pong is A to B and back to A within 5 ticks, not a slower return, and not through a failure",
      pp([[A, A, B, B, A, A, A, A, A, A, A, A], [A, A, B, B, B, B, B, B, B, B, B, B]]) == 1
      and pp([[A, B, B, B, B, B, B, B, B, B, A, A], [A, A, A, A, A, A, A, A, A, A, A, A]]) == 0
      and pp([[A, B, "RLF", "RLF", A, A, A, A, A, A, A, A], [A, A, A, A, A, A, A, A, A, A, A, A]]) == 0)

# ---- shadowing (our addition, off by default) ----------------------------------------------------------------------------------
sh = lab.shadowing(1, 2.0).reshape(60, 30, 5)
lag1 = np.mean([np.corrcoef(sh[:-1, u, c], sh[1:, u, c])[0, 1] for u in range(30) for c in range(5)])
check("shadowing is zero-mean noise with the stated spread and tick-to-tick correlation",
      abs(sh.std() - 2.0) < 0.2 and abs(sh.mean()) < 0.2 and abs(lag1 - M.SHADOW_RHO) < 0.1,
      f"std {sh.std():.2f}, lag-1 corr {lag1:.2f}")
check("it is deterministic per seed, and off by default (MAVERIC's smooth behaviour)",
      np.array_equal(lab.shadowing(1, 2.0), lab.shadowing(1, 2.0)) and np.array_equal(lab.rsrp(1, inv), lab.rsrp(1, inv, 0.0)))
dead = dict(inv)
dead = {c: dict(s) for c, s in inv.items()}
dead[3]["enabled"] = False
check("a switched-off cell stays off under shadowing", (lab.rsrp(1, dead, 4.0)[:, twin.cells.index(3)] <= -199).all())

# ---- through run_experiment ------------------------------------------------------------------------------------------------------
rt = runtime.build("local", None, 1200, lambda m: None, twin=twin)
g = rt.researcher.gw


def mob(cells=None, **kw):
    return g.invoke("run_experiment", {"config": {"objective": "mobility", "cells": cells or {}, **kw},
                                       "seeds": list(range(1, 7))}, agent="experimenter")


b1, b2 = mob(), mob()
check("the mobility objective reports the KPIs, per cell, and is deterministic (paired baselines)",
      {"handovers_per_ue_min", "ping_pongs_per_ue_min", "rlf_per_ue_min", "interruption_s_per_ue_min"} <= set(b1["mean"])
      and set(b1["per_cell_mean"]["time_share"]) == set(twin.cells) and b1["per_seed"] == b2["per_seed"])
check("the score is MAVERIC's interruption cost (50 ms per handover, 1 s per failure tick) per UE-minute",
      abs(b1["mean"]["score"] - np.mean([(p["handovers_per_ue_min"] * 0.05 + p["rlf_per_ue_min"]) for p in b1["per_seed"]])) < 1e-9)
no_hyst = mob(hyst=0.0)
check("with shadowing the hysteresis matters: the default 3 dB beats 0 dB on handovers and ping-pongs (one knob changed)",
      b1["mean"]["handovers_per_ue_min"] < no_hyst["mean"]["handovers_per_ue_min"]
      and b1["mean"]["ping_pongs_per_ue_min"] < no_hyst["mean"]["ping_pongs_per_ue_min"],
      f"handovers/UE-min {b1['mean']['handovers_per_ue_min']:.2f} vs {no_hyst['mean']['handovers_per_ue_min']:.2f}")
check("switching a cell off clearly cuts its share of attached time",
      mob({"2": {"enabled": False}})["per_cell_mean"]["time_share"][2] < 0.01 < b1["per_cell_mean"]["time_share"][2])
for cfg, kw, why in [({}, {"hyst": 20.0}, "hysteresis out of range"), ({}, {"ttt": 1}, "time-to-trigger below MAVERIC's minimum"),
                     ({}, {"ttt": 2.5}, "a fractional time-to-trigger"), ({}, {"hyst": 6.0, "ttt": 5}, "two handover knobs at once"),
                     ({"1": {"tilt": 8.0}}, {"hyst": 6.0}, "a cell setting plus a handover knob"),
                     ({}, {"n_ue": 1000}, "too many UEs"), ({}, {"instrument": "simulator"}, "the simulator tables for mobility")]:
    try:
        mob(cfg, **kw)
        check(f"blocked: {why}", False)
    except ToolError:
        check(f"blocked: {why}", True)
check("a cell change with the default handover rule is allowed", mob({"1": {"tilt": 8.0}})["objective"] == "mobility")

# ---- the fast path does not need the heavy MAVERIC app import (the Lambda image would not have it) ---------------------------------
code = ("import sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, r'%s');"
        "from atr import mobility as M; from atr.twin import RFTwin; from atr.topology import default_inventory;"
        "tw=RFTwin.load_or_build(); lab=M.MobilityLab(tw, default_inventory(), 6, 10);"
        "k=lab.kpis(lab.attach(1, default_inventory(), shadowing_db=2.0));"
        "print('notebooks' in sys.modules or any(m.startswith('apps') for m in sys.modules), k['handovers'] >= 0)" % ROOT)
out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT)).stdout.split()[-2:]
check("experiments run without importing MAVERIC's MRO app or notebooks (only radp)", out == ["False", "True"], str(out))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
