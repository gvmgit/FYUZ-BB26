"""Emulated lab: the instrument the researcher intervenes on, and the telemetry it produces.

Stands in for three HCLTech/organizer assets that are not available here, each behind the interface the real one would
plug into:
  VIAVI RAN simulator        fault injection, parameter changes, mobility and load control -> `EmulatedLab`
  E2E 5G network generator   traffic profiles (UE count, mobility mix, per-UE demand)       -> `TRAFFIC_PROFILES`
  HCLTech test automation API  the lab executor the agents call through the gateway         -> `LabExecutor`

Physics is MAVERIC's: signal from the RF digital twin, UE tracks from its Gauss-Markov generator, attachment with its
hysteresis / time-to-trigger / radio-link-failure rule (see mobility.py). What is ours, and EMULATED rather than measured:
  - per-cell KPI windows (load from per-UE demand over an assumed cell capacity),
  - alarms raised by fixed vendor-style thresholds,
  - decoded O-RAN interface logs (RRC, F1AP, E1AP, XnAP, NGAP) synthesised from the attachment timeline, in a syslog text
    format our parsers read (ingest.py). Real PCAP from the O-RAN test twin, parsed by HCLTech's F1/NGAP/E1 parsers, would
    replace `_interface_logs` and the parsers without touching anything downstream.
"""
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .faults import Fault
from .mobility import CLASS_MIX, MobilityLab

CELL_CAPACITY_MBPS = 150.0          # assumption: one 20 MHz NR carrier at moderate spectral efficiency
WINDOW_TICKS = 60                   # one KPI window = 60 one-second ticks
CAMPAIGN_EPOCH = 1793235600         # 2026-10-29T01:00:00Z: campaigns start "overnight"

# E2E 5G generator stand-in: who is on the network and how much they ask for.
TRAFFIC_PROFILES: Dict[str, dict] = {
    "busy_hour": {"n_ue": 60, "mix": dict(CLASS_MIX), "demand_mbps": 4.0, "slices": {"eMBB": 0.8, "URLLC": 0.1, "mMTC": 0.1}},
    "commuter": {"n_ue": 50, "mix": {"stationary": 0.05, "pedestrian": 0.2, "cyclist": 0.15, "car": 0.6}, "demand_mbps": 2.5,
                 "slices": {"eMBB": 0.9, "URLLC": 0.05, "mMTC": 0.05}},
    "factory_urllc": {"n_ue": 40, "mix": {"stationary": 0.5, "pedestrian": 0.4, "cyclist": 0.1, "car": 0.0}, "demand_mbps": 2.0,
                      "slices": {"eMBB": 0.3, "URLLC": 0.6, "mMTC": 0.1}},
    "night": {"n_ue": 20, "mix": {"stationary": 0.6, "pedestrian": 0.3, "cyclist": 0.05, "car": 0.05}, "demand_mbps": 1.0,
              "slices": {"eMBB": 0.6, "URLLC": 0.0, "mMTC": 0.4}},
}

# Vendor-style alarm thresholds per cell per window (fixed, documented; not tuned to the faults).
ALARM_RULES = [
    ("RadioLinkFailureSpike", "major", lambda k: k["rlf"] >= 3, "radio link failures on cell >= 3 in window"),
    ("PingPongRateHigh", "minor", lambda k: k["ping_pongs"] >= 4, "ping-pong handovers from cell >= 4 in window"),
    ("PRBCongestion", "minor", lambda k: k["prb_util_pct"] >= 85.0, "PRB utilisation >= 85%"),
    ("LowSINR", "minor", lambda k: k["mean_sinr_db"] is not None and k["mean_sinr_db"] < 3.0, "mean SINR of served UEs < 3 dB"),
    ("CellTrafficZero", "minor", lambda k: k["ue_share"] == 0.0, "no UE served in the whole window (sleeping-cell counter)"),
]
KPI_FEATURES = ("ue_share", "handovers_out", "handovers_in", "ping_pongs", "rlf", "mean_rsrp_dbm", "mean_sinr_db", "prb_util_pct")


def _iso(ts: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(ts)) + ".000Z"


def gnb_of(topology_rows) -> Dict[int, str]:
    """Co-sited cells share a gNB (one CU); cells on different sites hand over across Xn."""
    sites, out = {}, {}
    for _, r in topology_rows.iterrows():
        key = (round(float(r.cell_lat), 4), round(float(r.cell_lon), 4))
        sites.setdefault(key, f"gnb{len(sites) + 1}")
        out[int(r.cell_id)] = sites[key]
    return out


@dataclass
class Campaign:
    id: str
    profile: str
    windows: int
    seed0: int
    settings_note: str
    kpi: List[dict] = field(default_factory=list)       # one row per (window, cell)
    alarms: List[dict] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)       # decoded interface logs, syslog text
    meta: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"id": self.id, "profile": self.profile, "windows": self.windows, "kpi_rows": len(self.kpi),
                "alarms": len(self.alarms), "log_lines": len(self.logs), **self.meta}


class EmulatedLab:
    """The lab instrument. State: documented settings, an optional injected fault, a traffic profile and a handover rule.
    The injected fault is the lab's secret: `run_campaign` uses it, nothing returned to an agent reveals it."""

    def __init__(self, twin, inventory: dict, shadowing_db: float = 2.0, topology=None):
        from .topology import load_topology
        self.twin, self.inventory = twin, {c: dict(s) for c, s in inventory.items()}
        self.shadowing_db = shadowing_db
        self.gnb = gnb_of(topology if topology is not None else load_topology())
        self._labs: Dict[Tuple[str, int], MobilityLab] = {}
        self.reset()

    # -- the test automation API surface ------------------------------------------------------------------------------
    def reset(self):
        self.settings = {c: dict(s) for c, s in self.inventory.items()}
        self._fault: Optional[Fault] = None
        self.profile = "busy_hour"
        self.rule = (3.0, 3)

    def inject_fault(self, fault: Optional[Fault]):
        self._fault = fault

    def set_parameter(self, cell: int, name: str, value):
        if name not in ("tilt", "azimuth", "enabled"):
            raise ValueError(f"parameter {name} is not controllable in this lab")
        self.settings[int(cell)][name] = value

    def set_traffic(self, profile: str):
        if profile not in TRAFFIC_PROFILES:
            raise ValueError(f"unknown traffic profile {profile}; have {sorted(TRAFFIC_PROFILES)}")
        self.profile = profile

    def set_handover_rule(self, hyst: float, ttt: int):
        self.rule = (float(hyst), int(ttt))

    def truth(self) -> Optional[dict]:
        """For the reveal only; never passed to an agent."""
        return None if self._fault is None else {"kind": self._fault.kind, "cell": self._fault.cell, "value": self._fault.value}

    def actual_settings(self) -> dict:
        return self._fault.apply(self.settings) if self._fault else {c: dict(s) for c, s in self.settings.items()}

    # -- running -------------------------------------------------------------------------------------------------------
    def _mobility(self, profile: str) -> MobilityLab:
        p = TRAFFIC_PROFILES[profile]
        key = (profile, p["n_ue"])
        if key not in self._labs:
            self._labs[key] = MobilityLab(self.twin, self.inventory, n_ue=p["n_ue"], n_ticks=WINDOW_TICKS, mix=p["mix"])
        return self._labs[key]

    def run_campaign(self, windows: int = 6, seed0: int = 1, campaign_id: Optional[str] = None,
                     settings: Optional[dict] = None, profile: Optional[str] = None, with_logs: bool = True) -> Campaign:
        """Run `windows` one-minute windows, each a fresh set of UE tracks (seed0, seed0+1, ...). `settings` overrides the
        lab state for a replication run (the fault being tested injected into a clean lab)."""
        profile = profile or self.profile
        net = settings if settings is not None else self.actual_settings()
        lab, p = self._mobility(profile), TRAFFIC_PROFILES[profile]
        cid = campaign_id or f"camp-{profile}-{seed0}"
        camp = Campaign(cid, profile, windows, seed0, "lab state" if settings is None else "replication settings",
                        meta={"rule": list(self.rule), "shadowing_db": self.shadowing_db, "n_ue": p["n_ue"]})
        for w in range(windows):
            seed = seed0 + w
            att = lab.attach(seed, net, hyst=self.rule[0], ttt=self.rule[1], shadowing_db=self.shadowing_db)
            a = att.sort_values(["ue_id", "tick"])
            cells = a.cell_id.map(lambda x: -1 if str(x) == "RLF" else int(float(x))).values.reshape(lab.n_ue, lab.n_ticks)
            rsrp = a.cell_rxpower_dbm.values.reshape(lab.n_ue, lab.n_ticks)
            sinr = a.sinr_db.values.reshape(lab.n_ue, lab.n_ticks)
            t0 = CAMPAIGN_EPOCH + (seed0 % 1000) * 3600 + w * WINDOW_TICKS
            rows = self._kpi_rows(cid, w, t0, cells, rsrp, sinr, p["demand_mbps"])
            camp.kpi.extend(rows)
            for r in rows:
                for name, sev, rule, text in ALARM_RULES:
                    if rule(r):
                        camp.alarms.append({"campaign": cid, "window": w, "ts": _iso(t0 + WINDOW_TICKS - 1), "cell": r["cell"],
                                            "gnb": self.gnb.get(r["cell"]), "alarm": name, "severity": sev,
                                            "text": f"{name} on cell {r['cell']}: {text}"})
            if with_logs:
                camp.logs.extend(self._interface_logs(t0, cells))
        return camp

    def _kpi_rows(self, cid, w, t0, cells, rsrp, sinr, demand) -> List[dict]:
        n_ue, n_t = cells.shape
        prev, cur = cells[:, :-1], cells[:, 1:]
        pp = self._ping_pongs(cells)
        out = []
        for c in self.twin.cells:
            c = int(c)
            served = cells == c
            ue_ticks = int(served.sum())
            out.append({
                "campaign": cid, "window": w, "ts": _iso(t0), "cell": c, "gnb": self.gnb.get(c),
                "ue_share": round(ue_ticks / float(n_ue * n_t), 4),
                "handovers_out": int(np.sum((prev == c) & (cur != c) & (cur != -1))),
                "handovers_in": int(np.sum((prev != c) & (prev != -1) & (cur == c))),
                "ping_pongs": int(pp.get(c, 0)),
                "rlf": int(np.sum((prev == c) & (cur == -1))),
                "mean_rsrp_dbm": round(float(rsrp[served].mean()), 2) if ue_ticks else None,
                "mean_sinr_db": round(float(sinr[served].mean()), 2) if ue_ticks else None,
                "prb_util_pct": round(min(100.0, 100.0 * demand * ue_ticks / n_t / CELL_CAPACITY_MBPS), 1),
            })
        return out

    @staticmethod
    def _ping_pongs(cells, window: int = 5) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for row in cells:
            sw = [(t, row[t - 1], row[t]) for t in range(1, len(row)) if row[t] != -1 and row[t - 1] != -1 and row[t] != row[t - 1]]
            for k in range(1, len(sw)):
                if sw[k][2] == sw[k - 1][1] and sw[k][0] - sw[k - 1][0] <= window:
                    out[int(sw[k - 1][1])] = out.get(int(sw[k - 1][1]), 0) + 1
        return out

    def _interface_logs(self, t0: float, cells) -> List[str]:
        """Decoded O-RAN interface messages implied by the attachment timeline (EMULATED, see module docstring)."""
        lines = []

        def emit(t, node, proto, msg, **kv):
            lines.append(f"{_iso(t0 + t)} {node} {proto}: {msg} " + " ".join(f"{k}={v}" for k, v in kv.items()))

        for ue, row in enumerate(cells):
            prev = None
            for t, c in enumerate(row):
                c = int(c)
                if prev is None or (prev == -1 and c != -1):
                    if c == -1:
                        prev = c
                        continue
                    g = self.gnb[c]
                    reason = "re-establishment" if prev == -1 else "initial"
                    emit(t, f"{g}-du{c}", "RRC", "RRCSetupComplete", ue=ue, cell=c, reason=reason)
                    emit(t, f"{g}-cu", "F1AP", "InitialULRRCMessageTransfer", ue=ue, cell=c)
                    emit(t, f"{g}-cu", "NGAP", "InitialUEMessage", ue=ue, cell=c)
                    emit(t, f"{g}-cu", "E1AP", "BearerContextSetupRequest", ue=ue, cell=c)
                elif c == -1 and prev != -1:
                    g = self.gnb[prev]
                    emit(t, f"{g}-du{prev}", "RRC", "RRCReestablishmentRequest", ue=ue, cell=prev, cause="otherFailure")
                    emit(t, f"{g}-cu", "F1AP", "UEContextReleaseRequest", ue=ue, cell=prev, cause="radio-link-failure")
                elif c != prev and c != -1:
                    gs, gt = self.gnb[prev], self.gnb[c]
                    emit(t, f"{gs}-du{prev}", "RRC", "MeasurementReport", ue=ue, cell=prev, event="A3", neighbour=c)
                    if gs == gt:
                        emit(t, f"{gt}-cu", "F1AP", "UEContextSetupRequest", ue=ue, cell=c, source_cell=prev, cause="handover-desirable")
                        emit(t, f"{gt}-cu", "E1AP", "BearerContextModificationRequest", ue=ue, cell=c)
                    else:
                        emit(t, f"{gs}-cu", "XnAP", "HandoverRequest", ue=ue, cell=c, source_cell=prev, target_gnb=gt)
                        emit(t, f"{gt}-cu", "NGAP", "PathSwitchRequest", ue=ue, cell=c, source_cell=prev)
                    emit(t, f"{gs}-cu", "F1AP", "UEContextReleaseCommand", ue=ue, cell=prev, cause="handover-complete")
                prev = c
        return lines


class LabExecutor:
    """The test automation API the agents call through the gateway (tool `lab_execute`). Safety: it only drives the lab,
    never the live network; parameter changes are bounded like run_experiment's; every call is audited by the gateway."""

    ACTIONS = ("reset", "set_traffic", "set_handover_rule", "set_parameter", "run_campaign", "replicate")

    def __init__(self, lab: EmulatedLab, sink=None):
        self.lab, self.sink = lab, sink      # sink(campaign) ships telemetry into ingestion (streams, log lake)

    def validate(self, args: dict):
        from . import config
        act = args.get("action")
        if act not in self.ACTIONS:
            raise ValueError(f"unknown lab action {act}; allowed {self.ACTIONS}")
        if act == "set_parameter" and args.get("name") == "tilt" and not (config.TILT_RANGE[0] <= float(args["value"]) <= config.TILT_RANGE[1]):
            raise ValueError("tilt outside the lab's safe range")
        if act in ("run_campaign", "replicate") and int(args.get("windows", 6)) > 24:
            raise ValueError("at most 24 windows per campaign")

    def execute(self, action: str, **kw) -> dict:
        lab = self.lab
        if action == "reset":
            fault = lab._fault
            lab.reset()
            lab.inject_fault(fault)      # reset clears the agents' changes, not the lab's hidden fault
            return {"ok": True}
        if action == "set_traffic":
            lab.set_traffic(kw["profile"])
            return {"ok": True, "profile": kw["profile"]}
        if action == "set_handover_rule":
            lab.set_handover_rule(kw["hyst"], kw["ttt"])
            return {"ok": True, "rule": list(lab.rule)}
        if action == "set_parameter":
            lab.set_parameter(kw["cell"], kw["name"], kw["value"])
            return {"ok": True}
        if action == "run_campaign":
            camp = lab.run_campaign(int(kw.get("windows", 6)), int(kw.get("seed0", 1)), kw.get("campaign_id"))
        else:   # replicate: a CLEAN lab (documented settings) with one hypothesised fault injected
            f = kw.get("fault")
            settings = Fault(f["kind"], int(f["cell"]), float(f.get("value", 0.0))).apply(lab.settings) if f else dict(lab.settings)
            camp = lab.run_campaign(int(kw.get("windows", 3)), int(kw.get("seed0", 5000)), kw.get("campaign_id"),
                                    settings=settings, profile=kw.get("profile"), with_logs=bool(kw.get("with_logs", False)))
        if self.sink is not None:
            self.sink(camp)
        return {"campaign": camp.summary(), "kpi": camp.kpi, "alarms": camp.alarms}
