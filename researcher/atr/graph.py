"""The finding graph: patterns, causes, tests and fault classes, linked across findings.

Local: a networkx multigraph persisted as node-link JSON. AWS: Amazon Neptune over openCypher (aws_adapters.NeptuneGraph),
written with the same MERGE statements `cypher_for` produces here, so both hold the same graph.

Nodes   Finding{id,title,status}  Cell{id}  FaultClass{kind}  Pattern{id,modalities}  Test{name}  Artefact{id,type}
Edges   ABOUT (Finding->Cell)  CLASSIFIED_AS (Finding->FaultClass)  EXPLAINS (Finding->Pattern)  OBSERVED_IN (Pattern->Cell)
        TESTED_BY {passed} (Finding->Test)  RULED_OUT {how} (Finding->FaultClass of a rival)  YIELDED (Finding->Artefact)

The reusable question it answers: for a fault class, which tests separated it from its rivals, which modalities showed it,
and how many experiments it took - a recommended test strategy (`test_strategy`).
"""
import json
from pathlib import Path
from typing import Dict, List

from . import config

GRAPH_FILE = config.RUNS / "finding_graph.json"


def _checks(card: dict) -> List[dict]:
    out = [{"name": k["name"], "passed": k["passed"]} for k in card.get("checks", [])]
    lr = card.get("lab_replication")
    if lr:
        out.append({"name": "lab replication across traffic profiles", "passed": lr["passed"]})
    return out


def finding_facts(card: dict) -> dict:
    """Everything the graph stores about one finding card, in one plain structure (shared by both backends)."""
    c = card["claim"]
    status = ("accepted" if any(v["verdict"] == "accept" for v in card.get("engineer_verdicts", [])) else
              "rejected" if any(v["verdict"] == "reject" for v in card.get("engineer_verdicts", [])) else card["verdict"])
    obs = card.get("observation") or {}
    pat = next((p for p in obs.get("patterns", []) if p["cell"] == c["cell"]), None)
    ruled = [{"rival": k, "how": v} for k, v in (card.get("lab_replication") or {}).get("rival_status", {}).items()
             if v != "NOT excluded"]
    return {"id": card["id"], "title": card["title"], "status": status, "kind": c["kind"], "cell": int(c["cell"]),
            "experiments": (card.get("run_summary") or {}).get("experiments_run"),
            "pattern": ({"id": f"{card['id']}/pattern", "modalities": pat["flagged"]} if pat else None),
            "tests": _checks(card), "ruled_out": ruled,
            "artefacts": [{"id": a["id"], "type": a["type"]} for a in card.get("artefacts", [])]}


def cypher_for(f: dict) -> List[tuple]:
    """openCypher statements (with parameters) that upsert one finding. Used by the Neptune backend."""
    q = [("MERGE (x:Finding {id:$id}) SET x.title=$title, x.status=$status, x.experiments=$experiments",
          {"id": f["id"], "title": f["title"], "status": f["status"], "experiments": f["experiments"]}),
         ("MERGE (c:Cell {id:$cell}) WITH c MATCH (x:Finding {id:$id}) MERGE (x)-[:ABOUT]->(c)", {"cell": f["cell"], "id": f["id"]}),
         ("MERGE (k:FaultClass {kind:$kind}) WITH k MATCH (x:Finding {id:$id}) MERGE (x)-[:CLASSIFIED_AS]->(k)",
          {"kind": f["kind"], "id": f["id"]})]
    if f["pattern"]:
        q.append(("MERGE (p:Pattern {id:$pid}) SET p.modalities=$mods WITH p MATCH (x:Finding {id:$id}), (c:Cell {id:$cell}) "
                  "MERGE (x)-[:EXPLAINS]->(p) MERGE (p)-[:OBSERVED_IN]->(c)",
                  {"pid": f["pattern"]["id"], "mods": f["pattern"]["modalities"], "id": f["id"], "cell": f["cell"]}))
    for t in f["tests"]:
        q.append(("MERGE (t:Test {name:$name}) WITH t MATCH (x:Finding {id:$id}) MERGE (x)-[r:TESTED_BY]->(t) SET r.passed=$passed",
                  {"name": t["name"], "id": f["id"], "passed": t["passed"]}))
    for r in f["ruled_out"]:
        q.append(("MERGE (k:FaultClass {kind:$kind}) WITH k MATCH (x:Finding {id:$id}) MERGE (x)-[r:RULED_OUT {rival:$rival}]->(k) "
                  "SET r.how=$how", {"kind": r["rival"].split("-")[0], "id": f["id"], "rival": r["rival"], "how": r["how"]}))
    for a in f["artefacts"]:
        q.append(("MERGE (a:Artefact {id:$aid}) SET a.type=$type WITH a MATCH (x:Finding {id:$id}) MERGE (x)-[:YIELDED]->(a)",
                  {"aid": a["id"], "type": a["type"], "id": f["id"]}))
    return q


STRATEGY_QUERY = ("MATCH (x:Finding)-[:CLASSIFIED_AS]->(:FaultClass {kind:$kind}) WHERE x.status IN ['confirmed','accepted'] "
                  "OPTIONAL MATCH (x)-[t:TESTED_BY]->(test:Test) OPTIONAL MATCH (x)-[r:RULED_OUT]->(rk:FaultClass) "
                  "OPTIONAL MATCH (x)-[:EXPLAINS]->(p:Pattern) RETURN x.id AS finding, x.experiments AS experiments, "
                  "collect(DISTINCT [test.name, t.passed]) AS tests, collect(DISTINCT [r.rival, r.how]) AS ruled_out, "
                  "collect(DISTINCT p.modalities) AS modalities")


def strategy_from_rows(kind: str, rows: List[dict]) -> dict:
    """Turn per-finding rows (from either backend) into a recommended test strategy for the fault class."""
    if not rows:
        return {"fault_class": kind, "findings": 0, "strategy": [], "note": "no confirmed finding of this class yet"}
    tests, ruled, mods, exps = {}, {}, {}, []
    for r in rows:
        if r.get("experiments"):
            exps.append(r["experiments"])
        for name, passed in r["tests"]:
            if name is None or passed is None:      # not applicable for this kind (e.g. no second instrument)
                continue
            t = tests.setdefault(name, {"ran": 0, "passed": 0})
            t["ran"] += 1
            t["passed"] += 1 if passed else 0
        for rival, how in r["ruled_out"]:
            if rival is None:
                continue
            ruled.setdefault(rival.split("-")[0], set()).add("lab" if how.startswith("separated") else "rf_investigation")
        for m in r["modalities"]:
            for x in (m or []):
                mods[x] = mods.get(x, 0) + 1
    order = sorted(tests.items(), key=lambda kv: (-kv[1]["passed"] / kv[1]["ran"], kv[0]))
    steps = [f"watch {', '.join(sorted(mods, key=mods.get, reverse=True)) or 'the RF survey'} for the signature",
             f"sweep the {kind.replace('_', ' ')} on the top suspect cell, one setting at a time"]
    steps += [f"{n} (passed {v['passed']}/{v['ran']})" for n, v in order]
    for rk, how in sorted(ruled.items()):
        steps.append(f"rule out {rk.replace('_', ' ')} via {' and '.join(sorted(how))}")
    return {"fault_class": kind, "findings": len(rows),
            "median_experiments": sorted(exps)[len(exps) // 2] if exps else None,
            "modalities_seen": mods, "tests": tests, "rivals_ruled_out": {k: sorted(v) for k, v in ruled.items()},
            "strategy": steps}


class FindingGraph:
    def __init__(self, path: Path = GRAPH_FILE):
        import networkx as nx
        self.path = Path(path)
        self.g = nx.MultiDiGraph()
        if self.path.exists():
            self.g = nx.node_link_graph(json.loads(self.path.read_text()), directed=True, multigraph=True)

    def save(self):
        import networkx as nx
        self.path.write_text(json.dumps(nx.node_link_data(self.g)), encoding="utf-8")

    def _node(self, key: str, label: str, **props):
        if key in self.g:
            self.g.nodes[key].update(props)
        else:
            self.g.add_node(key, label=label, **props)
        return key

    def _edge(self, a, b, rel, **props):
        for _, v, d in self.g.out_edges(a, data=True):
            if v == b and d.get("rel") == rel and all(d.get(k) == x for k, x in props.items() if k == "rival"):
                d.update(props)
                return
        self.g.add_edge(a, b, rel=rel, **props)

    def add_finding(self, card: dict) -> dict:
        f = finding_facts(card)
        x = self._node(f"finding:{f['id']}", "Finding", title=f["title"], status=f["status"], experiments=f["experiments"])
        cell = self._node(f"cell:{f['cell']}", "Cell")
        self._edge(x, cell, "ABOUT")
        self._edge(x, self._node(f"class:{f['kind']}", "FaultClass"), "CLASSIFIED_AS")
        if f["pattern"]:
            p = self._node(f"pattern:{f['pattern']['id']}", "Pattern", modalities=f["pattern"]["modalities"])
            self._edge(x, p, "EXPLAINS")
            self._edge(p, cell, "OBSERVED_IN")
        for t in f["tests"]:
            self._edge(x, self._node(f"test:{t['name']}", "Test"), "TESTED_BY", passed=t["passed"])
        for r in f["ruled_out"]:
            self._edge(x, self._node(f"class:{r['rival'].split('-')[0]}", "FaultClass"), "RULED_OUT", rival=r["rival"], how=r["how"])
        for a in f["artefacts"]:
            self._edge(x, self._node(f"artefact:{a['id']}", "Artefact", type=a["type"]), "YIELDED")
        self.save()
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges()}

    def rows_for(self, kind: str) -> List[dict]:
        rows = []
        for x, _, d in self.g.in_edges(f"class:{kind}", data=True):
            if d.get("rel") != "CLASSIFIED_AS" or self.g.nodes[x].get("status") not in ("confirmed", "accepted"):
                continue
            out = list(self.g.out_edges(x, data=True))
            rows.append({"finding": x, "experiments": self.g.nodes[x].get("experiments"),
                         "tests": [[v[5:], e.get("passed")] for _, v, e in out if e["rel"] == "TESTED_BY"],
                         "ruled_out": [[e.get("rival"), e.get("how")] for _, v, e in out if e["rel"] == "RULED_OUT"],
                         "modalities": [self.g.nodes[v].get("modalities") for _, v, e in out if e["rel"] == "EXPLAINS"]})
        return rows

    def test_strategy(self, kind: str) -> dict:
        return strategy_from_rows(kind, self.rows_for(kind))

    def summary(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for _, d in self.g.nodes(data=True):
            out[d.get("label", "?")] = out.get(d.get("label", "?"), 0) + 1
        return out

    def neighbours(self, key: str) -> List[str]:
        return sorted(set(self.g.successors(key)) | set(self.g.predecessors(key))) if key in self.g else []


def make_graph(svc=None):
    from . import modes
    svc = svc or modes.resolve()
    if svc.aws("graph"):
        from .aws_adapters import NeptuneGraph
        return NeptuneGraph(region=svc.region)
    return FindingGraph()
