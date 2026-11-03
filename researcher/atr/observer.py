"""Observer: notices patterns ACROSS modalities, not within one.

Inputs come from the feature tables (SQL that runs on SQLite locally and on Athena on AWS) and from log search:
  kpi              per-cell KPI windows (share of UEs served, handovers, ping-pongs, failures, SINR, load)
  interface_logs   counts of decoded O-RAN interface events per cell (handover preparation, path switches, RLF releases, ...)
  alarms           vendor-style alarms per cell
  rf_survey        the Scientist's per-cell suspicion from the drive-test survey (added after the investigation observes)
plus an anomaly model (an isolation forest over the per-cell KPI windows; the SageMaker anomaly model's stand-in), which is
reported but does not count as a separate modality because it reads the same KPIs.

Each cell is compared with a healthy baseline campaign under the same traffic profile and handover rule: a Welch-style z
per feature. A modality is flagged for a cell when any of its features reaches |z| >= Z_FLAG (alarms: a Poisson excess). A
cell flagged by two or more modalities is a cross-modal pattern: that, not a single noisy counter, is what earns a hypothesis.
"""
from typing import Dict, List, Optional

import numpy as np

from .lab import KPI_FEATURES

Z_FLAG = 4.0
EVENT_CATS = ("handover_intra", "handover_inter", "path_switch", "context_release", "rlf_release", "reestablishment",
              "measurement_report", "attach")
FLOOR = {"ue_share": 0.01, "mean_rsrp_dbm": 0.5, "mean_sinr_db": 0.5, "prb_util_pct": 1.0}   # minimum sd per feature
COUNT_FLOOR = 0.5
MODALITIES = ("kpi", "interface_logs", "alarms", "rf_survey")


def window_tables(kpi_rows: List[dict], event_counts: List[dict], cells: List[int]) -> Dict[str, np.ndarray]:
    """feature -> array (windows, cells). event_counts rows: window, cell, category, n."""
    wins = sorted({r["window"] for r in kpi_rows})
    wi, ci = {w: i for i, w in enumerate(wins)}, {c: j for j, c in enumerate(cells)}
    out = {f: np.full((len(wins), len(cells)), np.nan) for f in KPI_FEATURES}
    for r in kpi_rows:
        for f in KPI_FEATURES:
            if r.get(f) is not None:
                out[f][wi[r["window"]], ci[int(r["cell"])]] = float(r[f])
    for cat in EVENT_CATS:
        out["ev_" + cat] = np.zeros((len(wins), len(cells)))
    for r in event_counts:
        if r["category"] in EVENT_CATS and r["cell"] is not None and int(r["cell"]) in ci and r["window"] in wi:
            out["ev_" + r["category"]][wi[r["window"]], ci[int(r["cell"])]] += r["n"]
    return out


def _z(cur: np.ndarray, base: np.ndarray, floor: float) -> float:
    cur, base = cur[~np.isnan(cur)], base[~np.isnan(base)]
    if len(base) == 0:
        return 0.0
    if len(cur) == 0:        # the cell served nobody in every window: as extreme as it gets for a level feature
        return -10.0 if base.mean() != 0 else 0.0
    vc = max(cur.var(ddof=1) if len(cur) > 1 else 0.0, floor ** 2)
    vb = max(base.var(ddof=1) if len(base) > 1 else 0.0, floor ** 2)
    return float(np.clip((cur.mean() - base.mean()) / np.sqrt(vc / len(cur) + vb / len(base)), -10, 10))


def z_matrix(cur: Dict[str, np.ndarray], base: Dict[str, np.ndarray], cells: List[int]) -> Dict[int, Dict[str, float]]:
    out = {}
    for j, c in enumerate(cells):
        out[c] = {f: round(_z(cur[f][:, j], base[f][:, j], FLOOR.get(f, COUNT_FLOOR)), 2) for f in cur}
    return out


def signature(z: Dict[int, Dict[str, float]], cells: List[int]) -> np.ndarray:
    feats = list(KPI_FEATURES) + ["ev_" + c for c in EVENT_CATS]
    return np.array([z[c].get(f, 0.0) for c in cells for f in feats])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


class Observer:
    name = "observer"

    def __init__(self, features, search=None, cells: Optional[List[int]] = None):
        self.features, self.search = features, search
        self.cells = cells or [1, 2, 3, 4, 5]

    # -- reading the feature tables (ANSI SQL: SQLite locally, Athena on AWS) ------------------------------------------
    def tables(self, campaign: str) -> Dict[str, np.ndarray]:
        # "window" is reserved in Athena (Presto/Trino); double quotes work in both engines
        kpi = self.features.query('SELECT * FROM kpi_window WHERE campaign = ? ORDER BY "window", cell', (campaign,))
        ev = self.features.query('SELECT "window", cell, category, COUNT(*) AS n FROM iface_event WHERE campaign = ? '
                                 'GROUP BY "window", cell, category', (campaign,))
        return window_tables(kpi, ev, self.cells)

    def alarm_counts(self, campaign: str) -> Dict[int, Dict[str, int]]:
        rows = self.features.query("SELECT cell, alarm, COUNT(*) AS n FROM alarm WHERE campaign = ? GROUP BY cell, alarm",
                                   (campaign,))
        out: Dict[int, Dict[str, int]] = {c: {} for c in self.cells}
        for r in rows:
            out.setdefault(int(r["cell"]), {})[r["alarm"]] = int(r["n"])
        return out

    def n_windows(self, campaign: str) -> int:
        r = self.features.query('SELECT COUNT(DISTINCT "window") AS n FROM kpi_window WHERE campaign = ?', (campaign,))
        return int(r[0]["n"]) if r else 0

    # -- the anomaly model ------------------------------------------------------------------------------------------------
    def anomaly_share(self, cur: Dict[str, np.ndarray], base: Dict[str, np.ndarray]) -> Dict[int, float]:
        """Isolation forest trained on healthy windows (features standardised per cell), applied to the current windows.
        Returns, per cell, the share of current windows the model calls anomalous."""
        from sklearn.ensemble import IsolationForest
        feats = [f for f in KPI_FEATURES if f != "mean_rsrp_dbm"]

        def standardise(t):
            X = []
            for j in range(len(self.cells)):
                cols = []
                for f in feats:
                    b = base[f][:, j]
                    mu = np.nanmean(b) if np.any(~np.isnan(b)) else 0.0
                    sd = max(np.nanstd(b) if np.any(~np.isnan(b)) else 0.0, FLOOR.get(f, COUNT_FLOOR))
                    cols.append((np.nan_to_num(t[f][:, j], nan=mu - 5 * sd) - mu) / sd)
                X.append(np.stack(cols, 1))
            return X   # per cell: (windows, features)

        Xb, Xc = standardise(base), standardise(cur)
        model = IsolationForest(n_estimators=200, contamination=0.05, random_state=0).fit(np.concatenate(Xb))
        return {c: round(float(np.mean(model.predict(Xc[j]) == -1)), 2) for j, c in enumerate(self.cells)}

    # -- patterns -----------------------------------------------------------------------------------------------------
    def observe(self, campaign: str, baseline: str, rf_suspicion: Optional[Dict[int, float]] = None) -> dict:
        cur, base = self.tables(campaign), self.tables(baseline)
        z = z_matrix(cur, base, self.cells)
        nc, nb = self.n_windows(campaign), self.n_windows(baseline)
        ac, ab = self.alarm_counts(campaign), self.alarm_counts(baseline)
        anomaly = self.anomaly_share(cur, base)
        patterns = []
        for c in self.cells:
            kz = {f: z[c][f] for f in KPI_FEATURES}
            ez = {f: v for f, v in z[c].items() if f.startswith("ev_")}
            mods: Dict[str, dict] = {}
            top_k = sorted(kz.items(), key=lambda kv: -abs(kv[1]))[:3]
            mods["kpi"] = {"flag": abs(top_k[0][1]) >= Z_FLAG, "top": dict(top_k)}
            top_e = sorted(ez.items(), key=lambda kv: -abs(kv[1]))[:3]
            mods["interface_logs"] = {"flag": abs(top_e[0][1]) >= Z_FLAG, "top": dict(top_e)}
            excess = {}
            for name in set(ac.get(c, {})) | set(ab.get(c, {})):
                expect = ab.get(c, {}).get(name, 0) * nc / max(nb, 1)
                seen = ac.get(c, {}).get(name, 0)
                if seen >= expect + 3 * np.sqrt(max(expect, 1.0)) and seen >= 2:
                    excess[name] = {"seen": seen, "expected": round(expect, 1)}
                elif expect >= 2 and seen <= max(0.0, expect - 3 * np.sqrt(expect)):
                    excess[name] = {"seen": seen, "expected": round(expect, 1)}      # an alarm that went quiet
            mods["alarms"] = {"flag": bool(excess), "top": excess}
            if rf_suspicion:
                med = float(np.median(list(rf_suspicion.values()))) or 1e-9
                s = rf_suspicion.get(c, 0.0)
                mods["rf_survey"] = {"flag": s >= 2.5 * med and s == max(rf_suspicion.values()), "top": {"suspicion": s,
                                                                                                           "median": round(med, 3)}}
            flagged = [m for m in MODALITIES if mods.get(m, {}).get("flag")]
            score = sum(min(10.0, abs(next(iter(mods[m]["top"].values()), 0) if m in ("kpi", "interface_logs") else 5.0))
                        for m in flagged)
            patterns.append({"cell": c, "modalities": mods, "flagged": flagged, "n_modalities": len(flagged),
                             "cross_modal": len(flagged) >= 2, "anomaly_model_share": anomaly[c], "score": round(score, 2)})
        patterns.sort(key=lambda p: (-p["n_modalities"], -p["score"]))
        for p in patterns:
            if p["cross_modal"] and self.search is not None:
                q = f"cell={p['cell']} radio-link-failure handover " + " ".join(p["modalities"]["alarms"]["top"])
                seen, ev = set(), []
                for d in self.search.search(q, k=20, where={"campaign": campaign, "cell": p["cell"]}):
                    key = d["text"].split(" ", 1)[-1]          # same message at another time is not new evidence
                    if key not in seen:
                        seen.add(key)
                        ev.append({"text": d["text"], "score": d["score"]})
                    if len(ev) == 3:
                        break
                p["log_evidence"] = ev
        return {"campaign": campaign, "baseline": baseline, "windows": nc, "baseline_windows": nb, "z": z,
                "patterns": patterns, "cross_modal": [p for p in patterns if p["cross_modal"]],
                "signature": signature(z, self.cells).tolist()}

    @staticmethod
    def describe(obs: dict) -> str:
        cm = obs["cross_modal"]
        if not cm:
            return "no cross-modal pattern: no cell is flagged by two or more modalities"
        p = cm[0]
        parts = []
        for m in p["flagged"]:
            top = p["modalities"][m]["top"]
            if m in ("kpi", "interface_logs"):
                parts.append(f"{m}: " + ", ".join(f"{k} z={v:+.1f}" for k, v in top.items() if abs(v) >= 2))
            elif m == "alarms":
                parts.append("alarms: " + ", ".join(f"{k} {v['seen']} vs {v['expected']} expected" for k, v in top.items()))
            else:
                parts.append(f"rf_survey: suspicion {top['suspicion']} vs median {top['median']}")
        return f"cell {p['cell']} flagged by {p['n_modalities']} modalities ({'; '.join(parts)})"
