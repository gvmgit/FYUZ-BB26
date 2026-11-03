"""Lab replication and the Analyst.

After the agents confirm a claim on the RF survey, the protocol INTERVENES on the lab: it injects the claimed fault into a
clean lab, and each rival explanation into another, re-runs the network on fresh UE tracks under more than one traffic
profile (a load and mobility intervention), and asks which injected fault reproduces the cross-modal signature that was
observed (KPIs and interface-log counts across all cells). This is a second check that uses different evidence from the
RF fit: mobility telemetry and decoded interface events instead of the drive-test survey.

It runs on the same MAVERIC physics as the observed lab, so it is a replication on a different instrument VIEW, not on
independent physics; the simulator tables remain the independent second instrument for tilt (agents.Critic).

The Analyst is its own agent (mesh role 'analyst'): it may compute statistics and nothing else. Its effect sizes run in
AgentCore Code Interpreter on AWS (stats.set_executor), the same code as locally.
"""
from typing import Dict, List, Optional

import numpy as np

from . import prereg, stats
from .faults import Fault
from .ingest import parse_log
from .observer import cosine, signature, window_tables, z_matrix

PROFILES = ("busy_hour", "commuter")
MIN_SIMILARITY = 0.5          # the injected claim must reproduce the observed signature at least this well (cosine)
MIN_MARGIN = 0.10             # ...and beat the best rival by at least this much, CI above zero, in every profile


class Analyst:
    name = "analyst"

    def __init__(self, mesh=None):
        self.mesh = mesh

    def effect(self, claim_sims: List[float], rival_sims: List[float]) -> dict:
        """Paired effect of 'the claim reproduces the observation better than its best rival'. Positive = claim wins."""
        if self.mesh is not None:
            self.mesh.authorize(self.name, "effect_size")
        return stats.paired_effect([1 - s for s in rival_sims], [1 - s for s in claim_sims])


def rivals_for(claim: dict, inventory: dict, second_cell: Optional[int]) -> List[dict]:
    k, c = claim["kind"], int(claim["cell"])
    az = inventory[c]["azimuth"]
    same_cell = {"tilt": [{"kind": "azimuth", "cell": c, "value": (az + 60) % 360}, {"kind": "cell_off", "cell": c}],
                 "azimuth": [{"kind": "tilt", "cell": c, "value": 16.0}, {"kind": "cell_off", "cell": c}],
                 "cell_off": [{"kind": "tilt", "cell": c, "value": 16.0}, {"kind": "azimuth", "cell": c, "value": (az + 120) % 360}]}[k]
    out = list(same_cell)
    if second_cell is not None and second_cell != c:
        v = claim.get("value", 0.0)
        if k == "azimuth":
            v = (inventory[second_cell]["azimuth"] + (float(claim["value"]) - az)) % 360
        out.append({"kind": k, "cell": int(second_cell), "value": float(v or 0.0)})
    return out


class LabReplicator:
    def __init__(self, lab_exec, inventory: dict, cells: List[int], audit=None, analyst: Optional[Analyst] = None,
                 reps: int = 4, windows: int = 3):
        self.lab_exec, self.inventory, self.cells = lab_exec, inventory, cells
        self.audit, self.analyst = audit, analyst or Analyst()
        self.reps, self.windows = reps, windows
        self._baselines: Dict[str, dict] = {}
        self.campaigns = 0

    def _tables(self, camp) -> Dict[str, np.ndarray]:
        events = parse_log("\n".join(camp.logs))
        start = camp.kpi[0]["ts"]
        from .ingest import _window_of
        counts: Dict[tuple, int] = {}
        for e in events:
            key = (_window_of(e["ts"], start), e["cell"], e["category"])
            counts[key] = counts.get(key, 0) + 1
        rows = [{"window": w, "cell": c, "category": cat, "n": n} for (w, c, cat), n in counts.items()]
        return window_tables(camp.kpi, rows, self.cells)

    def baseline(self, profile: str) -> dict:
        if profile not in self._baselines:
            camp = self.lab_exec.lab.run_campaign(12, 9001, f"baseline-{profile}", settings=self.inventory, profile=profile)
            self._baselines[profile] = self._tables(camp)
        return self._baselines[profile]

    def _sim(self, fault: Optional[dict], profile: str, rep: int, observed: np.ndarray) -> float:
        settings = (Fault(fault["kind"], int(fault["cell"]), float(fault.get("value", 0.0))).apply(self.inventory)
                    if fault else self.inventory)
        camp = self.lab_exec.lab.run_campaign(self.windows, 5000 + 100 * rep, f"repl-{profile}-{rep}", settings=settings,
                                              profile=profile)
        self.campaigns += 1
        z = z_matrix(self._tables(camp), self.baseline(profile), self.cells)
        return cosine(signature(z, self.cells), observed)

    def replicate(self, claim: dict, observed_signature: List[float], second_cell: Optional[int] = None,
                  profiles=PROFILES, excluded_elsewhere: Optional[Dict[str, str]] = None) -> dict:
        """`excluded_elsewhere`: rival id ('kind-cell') -> why the RF investigation already ruled it out. A rival the lab
        view cannot separate still leaves the claim standing if other evidence excluded it; that is reported, not hidden."""
        target, rivals, registration = self.preregister(claim, second_cell, profiles)
        per_profile = {p: self.replicate_profile(target, observed_signature, rivals, p) for p in profiles}
        out = combine(per_profile, rivals, excluded_elsewhere or {})
        out.update({"preregistration": registration, "campaigns": self.campaigns})
        return out

    def preregister(self, claim: dict, second_cell: Optional[int], profiles=PROFILES):
        """Seal the replication design, its prediction and its null outcome before any replication campaign runs."""
        rivals = rivals_for(claim, self.inventory, second_cell)
        target = {"kind": claim["kind"], "cell": int(claim["cell"]), "value": float(claim.get("value") or 0.0)}
        registration = prereg.register({
            "id": f"lab-replication-{claim['kind']}-{claim['cell']}", "claim": target, "rivals": rivals,
            "design": {"intervention": "inject the claim and each rival into a clean lab", "profiles": list(profiles),
                       "replicates": self.reps, "windows_each": self.windows, "fresh_seeds": "5000 + 100*rep"},
            "predicted_effect": {"metric": "cosine similarity to the observed cross-modal signature",
                                 "claim_reproduces_at_least": MIN_SIMILARITY,
                                 "rival_separated_when": f"claim beats it by >= {MIN_MARGIN} with 95% CI above 0",
                                 "in_every_profile": True,
                                 "pass": "claim reproduces in every profile and every rival is separated here or was "
                                         "already excluded by the RF investigation"},
            "null_outcome": "the claim does not reproduce the signature, or a rival that nothing else excluded reproduces "
                            "it as well"}, self.audit)
        return target, rivals, registration

    def replicate_profile(self, target: dict, observed, rivals: List[dict], profile: str) -> dict:
        """One traffic profile (one Map iteration of the Step Functions protocol)."""
        observed = np.asarray(observed, float)
        cs = [self._sim(target, profile, r, observed) for r in range(self.reps)]
        rs = {f"{f['kind']}-{f['cell']}": [self._sim(f, profile, r, observed) for r in range(self.reps)] for f in rivals}
        return {"profile": profile, "claim_similarity": round(float(np.mean(cs)), 3), "claim_sims": cs,
                "rival_similarity": {k: round(float(np.mean(v)), 3) for k, v in rs.items()},
                "effects": {k: self.analyst.effect(cs, v) for k, v in rs.items()},
                "reproduces": bool(np.mean(cs) >= MIN_SIMILARITY)}


def combine(per_profile: Dict[str, dict], rivals: List[dict], excluded_elsewhere: Dict[str, str]) -> dict:
    """The replication Critic's rule, applied to all profiles together (pre-registered in LabReplicator.replicate)."""
    ids = [f"{f['kind']}-{f['cell']}" for f in rivals]
    separated = {k: all(v["effects"][k]["mean_improvement"] >= MIN_MARGIN and v["effects"][k]["ci95"][0] > 0
                        for v in per_profile.values()) for k in ids}
    reproduces = all(v["reproduces"] for v in per_profile.values())
    rival_status = {k: ("separated in the lab" if separated[k] else
                        f"not separable in the lab view; excluded by the RF investigation ({excluded_elsewhere[k]})"
                        if k in excluded_elsewhere else "NOT excluded") for k in ids}
    passed = bool(reproduces and all(v != "NOT excluded" for v in rival_status.values()))
    return {"passed": passed, "reproduces": bool(reproduces), "rival_status": rival_status, "profiles": per_profile,
            "rivals": rivals, "prediction_met": passed,
            "note": "replication on the lab's mobility and interface-log view (same MAVERIC physics, fresh tracks, "
                    f"{len(per_profile)} traffic profiles)"}
