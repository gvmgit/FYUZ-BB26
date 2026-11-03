"""Checks for the full research pipeline: emulated lab, multimodal ingestion, cross-modal Observer, lab replication and the
Analyst, the Step Functions protocol and its local interpreter, the finding graph, artefacts, the review console API, the
language-model dataset, the AWS adapters (against fakes) and the CDK template.

Run from the researcher folder:  ..\.venv\Scripts\python.exe scripts\check_research_pipeline.py
"""
import json
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402

from atr import artefacts as A, aws_adapters as aws, modes, nlm, prereg, protocol as P, research_cycle as RC  # noqa: E402
from atr import serve  # noqa: E402
from atr.anf import AgentMesh, PolicyViolation  # noqa: E402
from atr.faults import Fault  # noqa: E402
from atr.graph import FindingGraph, cypher_for, finding_facts, strategy_from_rows  # noqa: E402
from atr.ingest import FeatureStore, Ingestion, LogLake, LogSearch, StreamBus, parse_line  # noqa: E402
from atr.lab import TRAFFIC_PROFILES, EmulatedLab, LabExecutor  # noqa: E402
from atr.observer import Observer  # noqa: E402
from atr.replicate import Analyst, LabReplicator, combine  # noqa: E402
from atr.scheduler import LocalScheduler  # noqa: E402
from atr.store import SQLiteStore  # noqa: E402
from atr.topology import default_inventory  # noqa: E402
from atr.twin import RFTwin  # noqa: E402

results = []


def check(name, cond, detail=""):
    results.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


TMP = Path(tempfile.mkdtemp())
twin, inv = RFTwin.load_or_build(), default_inventory()
cells = list(twin.cells)


def fresh_ingestion(tag):
    return Ingestion(StreamBus(TMP / f"s-{tag}"), LogLake(TMP / f"l-{tag}"), FeatureStore(TMP / f"f-{tag}.sqlite"))


# ---- emulated lab ------------------------------------------------------------------------------------------------------
lab = EmulatedLab(twin, inv)
a = lab.run_campaign(3, 11, "a", settings=inv)
b = lab.run_campaign(3, 11, "b", settings=inv)
check("lab: one KPI row per (window, cell)", len(a.kpi) == 3 * len(cells), len(a.kpi))
check("lab: same seeds give the same telemetry", [r["ue_share"] for r in a.kpi] == [r["ue_share"] for r in b.kpi])
off = lab.run_campaign(3, 11, "off", settings=Fault("cell_off", 2).apply(inv))
check("lab: a cell that is off serves nobody and raises the sleeping-cell counter",
      all(r["ue_share"] == 0 for r in off.kpi if r["cell"] == 2) and any(x["alarm"] == "CellTrafficZero" and x["cell"] == 2 for x in off.alarms))
check("lab: interface logs cover RRC, F1AP, E1AP, NGAP and XnAP",
      {"RRC", "F1AP", "E1AP", "NGAP", "XnAP"} <= {l.split()[2].rstrip(":") for l in a.logs})
check("lab: co-sited cells share a gNB", lab.gnb[1] == lab.gnb[2] and lab.gnb[3] != lab.gnb[4], lab.gnb)
check("lab: four traffic profiles (E2E generator stand-in)", set(TRAFFIC_PROFILES) == {"busy_hour", "commuter", "factory_urllc", "night"})
night = lab.run_campaign(2, 11, "n", settings=inv, profile="night")
check("lab: load intervention changes PRB utilisation", np.mean([r["prb_util_pct"] for r in night.kpi]) <
      np.mean([r["prb_util_pct"] for r in a.kpi]))

ex = LabExecutor(lab)
lab.inject_fault(Fault("tilt", 3, 16))
for bad, why in (({"action": "format_disk"}, "unknown action"), ({"action": "set_parameter", "cell": 3, "name": "tilt", "value": 30}, "tilt bound"),
                 ({"action": "run_campaign", "windows": 99}, "campaign size")):
    try:
        ex.validate(bad)
        ok = False
    except ValueError:
        ok = True
    check(f"lab executor refuses: {why}", ok)
ex.execute("reset")
check("lab executor: reset keeps the lab's hidden fault", lab.truth() == {"kind": "tilt", "cell": 3, "value": 16})
out = ex.execute("run_campaign", windows=2, seed0=5)
check("lab executor: results never reveal the hidden fault", "tilt" not in json.dumps(out["campaign"]) and "truth" not in json.dumps(out))

# ---- ingestion -------------------------------------------------------------------------------------------------------------
ev = parse_line("2026-10-29T01:00:05.000Z gnb2-cu F1AP: UEContextReleaseRequest ue=7 cell=3 cause=radio-link-failure")
check("parser: F1AP radio-link-failure release", ev and ev["category"] == "rlf_release" and ev["cell"] == 3 and ev["ue"] == 7)
ev = parse_line("2026-10-29T01:00:05.000Z gnb1-cu XnAP: HandoverRequest ue=1 cell=4 source_cell=2 target_gnb=gnb3")
check("parser: XnAP handover keeps its fields", ev["category"] == "handover_inter" and ev["fields"]["source_cell"] == "2")
check("parser: garbage line is skipped", parse_line("not a log line") is None)
ing = fresh_ingestion("main")
base = lab.run_campaign(12, 9001, RC.BASELINE_ID, settings=inv)
ing.ingest(base)
lab.inject_fault(Fault("tilt", 3, 16))
camp = lab.run_campaign(6, 101, "camp-tilt")
counts = ing.ingest(camp)
check("ingest: every log line becomes a parsed event", counts["events"] == counts["log_lines"], counts)
check("ingest: KPI and alarms went through the stream", len(ing.bus.read("kpi")) == len(base.kpi) + len(camp.kpi))
check("ingest: logs landed in the lake", "F1AP" in ing.lake.get("campaigns/camp-tilt/oran-interfaces.log"))
q = ing.features.query('SELECT COUNT(DISTINCT "window") AS n FROM iface_event WHERE campaign = ?', ("camp-tilt",))
check("ingest: interface events carry their window (quoted for Athena too)", q[0]["n"] == 6, q)
hits = ing.search.search("radio-link-failure handover cell=3", k=3, where={"campaign": "camp-tilt"})
check("search: finds failure / handover lines for the cell", hits and all(h["campaign"] == "camp-tilt" for h in hits))

# ---- observer ------------------------------------------------------------------------------------------------------------
obs = Observer(ing.features, ing.search, cells)
o = obs.observe("camp-tilt", RC.BASELINE_ID)
check("observer: tilt fault on cell 3 is a cross-modal pattern on cell 3", o["cross_modal"] and o["cross_modal"][0]["cell"] == 3,
      Observer.describe(o))
check("observer: z-scores are finite", all(np.isfinite(v) for c in o["z"].values() for v in c.values()))
o2 = obs.observe("camp-tilt", RC.BASELINE_ID, rf_suspicion={1: 2.0, 2: 1.8, 3: 11.0, 4: 2.5, 5: 1.9})
check("observer: the survey adds a fourth modality", "rf_survey" in o2["cross_modal"][0]["flagged"])
lab.inject_fault(None)
healthy = lab.run_campaign(6, 101, "camp-healthy")
ing.ingest(healthy)
oh = obs.observe("camp-healthy", RC.BASELINE_ID)
check("observer: healthy network shows no cross-modal pattern", not oh["cross_modal"], Observer.describe(oh))
lab.inject_fault(Fault("cell_off", 2))
ing.ingest(lab.run_campaign(6, 101, "camp-off"))
oo = obs.observe("camp-off", RC.BASELINE_ID)
check("observer: a dead cell is flagged by KPIs, interface logs and alarms", set(oo["cross_modal"][0]["flagged"]) >= {"kpi", "interface_logs", "alarms"}
      and oo["cross_modal"][0]["cell"] == 2)

# ---- replication and the Analyst --------------------------------------------------------------------------------------------
mesh = AgentMesh()
rep = LabReplicator(LabExecutor(lab), inv, cells, analyst=Analyst(mesh), reps=3)
r_ok = rep.replicate({"kind": "cell_off", "cell": 2}, oo["signature"], second_cell=4)
check("replication: the true fault reproduces the signature and separates every rival", r_ok["passed"], r_ok["rival_status"])
check("replication: pre-registered and verifiable", prereg.verify(r_ok["preregistration"]))
r_bad = rep.replicate({"kind": "azimuth", "cell": 3, "value": 110}, o["signature"], second_cell=5)
check("replication: a wrong claim does not reproduce the signature", not r_bad["passed"] and not r_bad["reproduces"])
r_tilt = rep.replicate({"kind": "tilt", "cell": 3, "value": 16}, o["signature"], second_cell=4)
check("replication: tilt-16 vs cell-off is not separable from telemetry alone (reported, not hidden)",
      "not separable" in r_tilt["rival_status"].get("cell_off-3", "") or r_tilt["rival_status"].get("cell_off-3") == "NOT excluded",
      r_tilt["rival_status"].get("cell_off-3"))
per = r_tilt["profiles"]
combined = combine(per, r_tilt["rivals"], {"cell_off-3": "twin offset explains it"})
check("replication: a rival excluded by the RF investigation leaves the claim standing", combined["passed"] == r_tilt["reproduces"]
      and all(v != "NOT excluded" for v in combined["rival_status"].values()) == combined["passed"])
try:
    mesh.authorize("analyst", "run_experiment")
    ok = False
except PolicyViolation:
    ok = True
check("mesh: the Analyst may compute statistics but not run experiments", ok)

# ---- protocol --------------------------------------------------------------------------------------------------------------
doc = P.definition()
names = set(doc["States"])
nexts = {s.get("Next") for s in doc["States"].values()} | {r["Next"] for s in doc["States"].values() for r in s.get("Choices", [])} | \
        {s.get("Default") for s in doc["States"].values()} | {c["Next"] for s in doc["States"].values() for c in s.get("Catch", [])}
check("protocol: every transition points at a state", (nexts - {None}) <= names, sorted((nexts - {None}) - names))
tasks = {s["Parameters"]["Payload"]["task"] for s in doc["States"].values() if s["Type"] == "Task"}
tasks |= {s["Parameters"]["Payload"]["task"] for s in doc["States"]["ReplicateAcrossProfiles"]["ItemProcessor"]["States"].values() if s["Type"] == "Task"}
implemented = {n[2:] for n in dir(RC.ResearchCycle) if n.startswith("t_")}
check("protocol: every task has an implementation", tasks <= implemented, sorted(tasks - implemented))
check("protocol: Lambda ARN is a CDK substitution", P.FN in json.dumps(doc))
toy = {"StartAt": "A", "States": {
    "A": {"Type": "Task", "Parameters": {"Payload": {"task": "a"}}, "ResultPath": "$.a", "Next": "C",
          "Catch": [{"ErrorEquals": ["States.ALL"], "ResultPath": "$.error", "Next": "F"}]},
    "C": {"Type": "Choice", "Choices": [{"Variable": "$.a.result.go", "BooleanEquals": True, "Next": "M"}], "Default": "D"},
    "M": {"Type": "Map", "ItemsPath": "$.items", "ItemSelector": {"v.$": "$$.Map.Item.Value"}, "ResultPath": "$.m",
          "ItemProcessor": {"StartAt": "T", "States": {"T": {"Type": "Task", "Parameters": {"Payload": {"task": "t"}},
                                                             "ResultPath": "$.r", "End": True}}}, "Next": "D"},
    "D": {"Type": "Succeed"}, "F": {"Type": "Fail", "Error": "Boom"}}}
sm = P.LocalStateMachine(toy, {"a": lambda s: {"go": True}, "t": lambda s: s["v"] * 2}, emit=None)
res = sm.run({"items": [1, 2, 3]})
check("interpreter: Task, Choice and Map run in order", [x["r"]["result"] for x in res["m"]] == [2, 4, 6])
sm2 = P.LocalStateMachine(toy, {"a": lambda s: 1 / 0, "t": lambda s: 0}, emit=None)
try:
    sm2.run({"items": []})
    ok = False
except P.ExecutionFailed:
    ok = True
check("interpreter: a failing task is caught and ends in Fail", ok and sm2.history[-1]["state"] == "F")

# ---- graph ------------------------------------------------------------------------------------------------------------------
card = {"id": "c1", "title": "Cell 3 tilt", "verdict": "confirmed", "claim": {"kind": "tilt", "cell": 3},
        "run_summary": {"experiments_run": 90},
        "checks": [{"name": "control cell shows no comparable gain", "passed": True}, {"name": "holds on fresh seeds", "passed": True}],
        "observation": {"patterns": [{"cell": 3, "flagged": ["kpi", "interface_logs"]}]},
        "lab_replication": {"passed": True, "rival_status": {"cell_off-3": "not separable in the lab view; excluded by the RF investigation (x)",
                                                              "azimuth-3": "separated in the lab"}},
        "artefacts": [{"id": "c1/ewi", "type": "early_warning_indicator"}]}
g = FindingGraph(TMP / "graph.json")
g.add_finding(card)
g.add_finding(card)                                   # idempotent
g2 = FindingGraph(TMP / "graph.json")
check("graph: persisted and idempotent", g2.summary() == g.summary() and g.summary().get("Finding") == 1, g.summary())
st = g2.test_strategy("tilt")
check("graph: test strategy for the class", st["findings"] == 1 and any("rule out cell off" in s for s in st["strategy"]), st["strategy"])
check("graph: unknown class has no strategy yet", g2.test_strategy("azimuth")["findings"] == 0)
qs = cypher_for(finding_facts(card))
check("graph: openCypher upserts cover finding, cell, class, pattern, tests, rivals, artefacts", len(qs) == 3 + 1 + 3 + 2 + 1, len(qs))
fake_rows = g2.rows_for("tilt")
check("graph: Neptune rows and local rows give the same strategy", strategy_from_rows("tilt", fake_rows) == st)
calls = []
ng = aws.NeptuneGraph(endpoint="db.cluster.example", request=lambda body: (calls.append(body), {"results": []})[1])
ng.add_finding(card)
check("Neptune adapter: one signed openCypher POST per statement (+ summary)", len(calls) == len(qs) + 1 and b"MERGE" in calls[0])

# ---- artefacts ---------------------------------------------------------------------------------------------------------------
ewi = A.early_warning_indicator({"id": "x", "claim": {"kind": "tilt", "cell": 4, "value": 6.0}}, None, rep)
check("early warning: tilt 4->6 indicator is deployable on fresh windows", ewi["status"] == "deployable", ewi["rule"] + " " + json.dumps(ewi["validation"]))
check("early warning: validated on seeds it was not calibrated on", ewi["validation"]["false_alarm"]["of"] == 24)
sample = A.fine_tuning_sample({"id": "c1", "claim": {"kind": "tilt", "cell": 3, "estimate": "16 deg"}, "checks": card["checks"]},
                              o2, {1: 2.0, 3: 11.0}, "zone Z7 weaker")
check("fine-tuning sample: chat format with a JSON target", [m["role"] for m in sample["messages"]] == ["system", "user", "assistant"]
      and json.loads(sample["messages"][2]["content"])["cell"] == 3)
cp = TMP / "corpus.jsonl"
A.append_to_corpus(sample, cp)
check("corpus: re-adding a finding replaces it", A.append_to_corpus(sample, cp) == 1)

# ---- language-model dataset uses accepted findings only ---------------------------------------------------------------------
store = SQLiteStore(TMP / "cards.sqlite")
store.put({"id": "c1", "created": 1, "verdict": "confirmed", "engineer_verdicts": [{"verdict": "accept", "by": "x"}]})
A.CORPUS = cp
ds = nlm.build_dataset(include_unreviewed=False, store=store)
check("NLM dataset: accepted finding is used", ds["findings_used"] == 1, ds["findings_used"])
store.put({"id": "c1", "created": 1, "verdict": "confirmed", "engineer_verdicts": []})
check("NLM dataset: unreviewed finding is not used unless asked", nlm.build_dataset(store=store)["findings_used"] == 0
      and nlm.build_dataset(include_unreviewed=True, store=store)["findings_used"] == 1)
check("NLM dataset: held-out test split is separate from training", not {r["meta"]["case_id"] for r in ds["test"]} &
      {r["meta"]["case_id"] for r in ds["train"]})

# ---- review console API -------------------------------------------------------------------------------------------------------
cards = SQLiteStore(TMP / "console.sqlite")
cards.put({"id": "k1", "created": 1, "verdict": "confirmed", "headline": True, "claim": {"kind": "tilt", "cell": 3}, "checks": [],
           "escalation": {"status": "awaiting_human_approval"}})
v = serve.api_verdict(cards, {"card_id": "k1", "verdict": "accept", "comment": "ok"}, "reviewer@example.com",
                      scheduler=LocalScheduler(), audit=AgentMesh().audit)
check("console API: verdict recorded under the reviewer's identity", v["verdicts"][-1]["by"] == "reviewer@example.com")
check("console API: graph route", set(serve.api_graph(g2, "tilt")) == {"summary", "strategy"})
js = (ROOT / "console" / "app.js").read_text(encoding="utf-8")
check("console: SPA calls the same routes and supports Cognito", all(x in js for x in ("/api/findings", "/api/graph", "/api/verdict", "oauth2/authorize")))

# ---- AWS adapters against fakes ------------------------------------------------------------------------------------------------
class FakeKinesis:
    def __init__(self):
        self.sent = []

    def put_records(self, StreamName, Records):
        self.sent.append((StreamName, len(Records)))
        return {"FailedRecordCount": 0}


fk = FakeKinesis()
n = aws.KinesisStreams("atr", client=fk).put_records("kpi", [{"cell": i % 5} for i in range(1203)])
check("Kinesis: batches of at most 500 to <prefix>-<stream>", n == 1203 and fk.sent == [("atr-kpi", 500), ("atr-kpi", 500), ("atr-kpi", 203)])


class FakeAthena:
    def __init__(self, state="SUCCEEDED"):
        self.state, self.kw = state, None

    def start_query_execution(self, **kw):
        self.kw = kw
        return {"QueryExecutionId": "q1"}

    def get_query_execution(self, QueryExecutionId):
        return {"QueryExecution": {"Status": {"State": self.state, "StateChangeReason": "bad"}}}

    def get_query_results(self, QueryExecutionId, **kw):
        return {"ResultSet": {"Rows": [{"Data": [{"VarCharValue": "cell"}, {"VarCharValue": "n"}]},
                                       {"Data": [{"VarCharValue": "3"}, {"VarCharValue": "1.5"}]}]}}


fa = FakeAthena()
af = aws.AthenaFeatureStore("db", "s3://o/", "lake", athena=fa, s3=object(), poll=0)
rows = af.query('SELECT cell, n FROM t WHERE campaign = ?', ("it's",))
check("Athena: parameters quoted as SQL literals, numbers typed back", fa.kw["ExecutionParameters"] == ["'it''s'"] and rows == [{"cell": 3, "n": 1.5}])
try:
    aws.AthenaFeatureStore("db", "s3://o/", "lake", athena=FakeAthena("FAILED"), s3=object(), poll=0).query("x")
    ok = False
except RuntimeError:
    ok = True
check("Athena: a failed query raises", ok)
reqs = []
os_ = aws.OpenSearchLogSearch("https://x.aoss.amazonaws.com", request=lambda m, u, b=None: (reqs.append((m, u, b)),
                              {"hits": {"hits": [{"_source": {"text": "t", "doc_id": "d1"}, "_score": 1.2}]}})[1],
                              embed=lambda t: [0.1, 0.2])
os_.index([{"id": "d1", "text": "rlf cell=3", "campaign": "c", "cell": 3}])
hit = os_.search("rlf", k=2, where={"cell": 3})
bulk = [r for r in reqs if r[1].endswith("/_bulk")]          # the index mapping is created first, on first use
query = [r for r in reqs if r[1].endswith("/_search")]
check("OpenSearch: bulk index with embeddings, hybrid k-NN + text search with filters",
      bulk and b'"vec"' in bulk[0][2] and query and b'"knn"' in query[0][2] and b'"term"' in query[0][2] and hit[0]["id"] == "d1")


class FakeSfn:
    def start_execution(self, **kw):
        self.kw = kw
        return {"executionArn": "arn:exec"}

    def describe_execution(self, executionArn):
        return {"status": "SUCCEEDED", "output": "{\"ok\": 1}"}

    def get_execution_history(self, executionArn, maxResults):
        return {"events": [{"type": "TaskStateEntered", "stateEnteredEventDetails": {"name": "Observe"}, "timestamp": 1}]}


fs = FakeSfn()
sp = aws.StepFunctionsProtocol("arn:sm", client=fs, poll=0)
arn = sp.start({"run_id": "r1"})
check("Step Functions: execution named by run id, waited, history read", fs.kw["name"] == "r1" and sp.wait(arn)["output"] == {"ok": 1}
      and sp.history(arn)[0]["state"] == "Observe")


class FakeBedrock:
    def create_model_import_job(self, **kw):
        self.kw = kw
        return {"jobArn": "arn:job"}


fb = FakeBedrock()
aws.BedrockModelImport(bedrock=fb, runtime=object()).start_import("j", "atr-nlm", "s3://m/merged/", "arn:role")
check("Bedrock Custom Model Import: job reads the merged model from S3", fb.kw["modelDataSource"]["s3DataSource"]["s3Uri"] == "s3://m/merged/")
check("lab on AgentCore Gateway: lab_execute tool", aws.RemoteLabGateway.tool == "lab_execute")
check("toggle: every new service has its AWS settings", all(s in modes.SERVICES for s in ("stream", "loglake", "features", "search", "graph",
                                                                                          "protocol", "lab", "model_import")))

# ---- the whole cycle, end to end (classic engine for speed; strands is exercised by the CLI) ---------------------------------
RC.NLM_STATE = TMP / "nlm_state.json"
A.CORPUS = TMP / "corpus-cycle.jsonl"
import atr.serve as S  # noqa: E402
S.make_scheduler = lambda: LocalScheduler()           # keep the real queue untouched
rc = RC.ResearchCycle(Fault("azimuth", 1, 225), engine="classic", ingestion=fresh_ingestion("cycle"), store=SQLiteStore(TMP / "cycle.sqlite"),
                      graph=FindingGraph(TMP / "cycle-graph.json"), emit=None, twin=twin, reps=3)
out = rc.run()
h = out["headline"]
check("cycle: the lab's secret fault is found", h and h["claim"]["kind"] == "azimuth" and h["claim"]["cell"] == 1, h and h["title"])
check("cycle: every protocol state ran", {x["state"] for x in out["history"]} >= {"RunLabCampaign", "Ingest", "Observe", "Investigate",
      "CrossCheckModalities", "PreRegisterReplication", "LabReplicate", "Analyse", "Falsify", "Publish", "UpdateFindingGraph",
      "DeriveArtefacts", "QueueRetraining", "Done"})
check("cycle: card carries observation, lab replication, artefacts and protocol history",
      all(k in h for k in ("observation", "lab_replication", "artefacts", "protocol")) and h["protocol"]["history"])
check("cycle: lab replication pre-registered", prereg.verify(h["lab_replication"]["preregistration"]))
check("cycle: nothing reaches the network without a person", h["escalation"]["status"] in ("awaiting_human_approval", "needs_human_review"))
rc2 = RC.ResearchCycle(None, engine="classic", ingestion=fresh_ingestion("cycle2"), store=SQLiteStore(TMP / "cycle2.sqlite"),
                       graph=FindingGraph(TMP / "cycle2-graph.json"), emit=None, twin=twin, reps=2)
out2 = rc2.run()
check("cycle: a healthy lab publishes no fault and skips replication", out2["headline"] is None and
      "PublishWithoutReplication" in {x["state"] for x in out2["history"]})

# ---- CDK template --------------------------------------------------------------------------------------------------------------
tpl = ROOT / "deploy" / "cdk" / "cdk.out" / "TelecomResearcher.template.json"
if tpl.exists():
    types = {r["Type"] for r in json.loads(tpl.read_text())["Resources"].values()}
    need = {"AWS::Kinesis::Stream", "AWS::Glue::Table", "AWS::Athena::WorkGroup", "AWS::OpenSearchServerless::Collection",
            "AWS::Neptune::DBCluster", "AWS::StepFunctions::StateMachine", "AWS::Scheduler::Schedule", "AWS::SQS::Queue",
            "AWS::Amplify::App", "AWS::Cognito::UserPool", "AWS::ApiGateway::RestApi", "AWS::Grafana::Workspace",
            "AWS::DynamoDB::Table", "AWS::Lambda::Function"}
    check("CDK: synthesised template has every proposal service", need <= types, sorted(need - types))
else:
    check("CDK: synthesised template present (run `python app.py` in deploy/cdk)", False)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
