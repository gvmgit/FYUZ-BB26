"""Multimodal ingestion: streams, log lake, interface-log parsers, feature tables and log search.

Local implementation          AWS implementation (aws_adapters.py)       carries
  StreamBus (JSONL files)       Kinesis Data Streams                       KPI windows, alarms
  LogLake (folder)              S3 + parser Lambda                         decoded O-RAN interface logs
  parse_line / PARSERS          (HCLTech F1/NGAP/E1 parsers on Fargate)    log line -> structured event
  FeatureStore (SQLite)         Glue tables + Athena                       kpi_window, alarm, iface_event
  LogSearch (TF-IDF vectors)    OpenSearch Serverless (text + vectors)     log and alarm text

The SQL the Observer runs is plain ANSI that both SQLite and Athena accept. `Ingestion` wires one implementation of each.
"""
import json
import re
import sqlite3
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from . import config

LAKE_DIR = config.RUNS / "lake"
STREAM_DIR = config.RUNS / "streams"
FEATURES_DB = config.RUNS / "features.sqlite"


# ---------------------------------------------------------------- streams
class StreamBus:
    """Append-only streams, one JSONL file each. `read` returns records after a sequence number, like a shard iterator."""

    def __init__(self, root: Path = STREAM_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put_records(self, stream: str, records: Iterable[dict]) -> int:
        n = 0
        with open(self.root / f"{stream}.jsonl", "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
                n += 1
        return n

    def read(self, stream: str, after: int = 0) -> List[dict]:
        p = self.root / f"{stream}.jsonl"
        if not p.exists():
            return []
        with open(p, encoding="utf-8") as f:
            return [json.loads(l) for i, l in enumerate(f) if i >= after and l.strip()]


# ---------------------------------------------------------------- log lake
class LogLake:
    def __init__(self, root: Path = LAKE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def put(self, key: str, text: str) -> str:
        p = self.root / key
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return str(p)

    def get(self, key: str) -> str:
        return (self.root / key).read_text(encoding="utf-8")


# ---------------------------------------------------------------- parsers
LINE = re.compile(r"^(?P<ts>\S+) (?P<node>\S+) (?P<proto>[A-Za-z0-9]+): (?P<msg>\S+)(?: (?P<kv>.*))?$")
KV = re.compile(r"(\w+)=(\S+)")

# message -> normalised event category (what the Observer counts)
CATEGORY = {
    ("RRC", "RRCSetupComplete"): "attach", ("F1AP", "InitialULRRCMessageTransfer"): "attach",
    ("NGAP", "InitialUEMessage"): "attach", ("E1AP", "BearerContextSetupRequest"): "bearer_setup",
    ("RRC", "MeasurementReport"): "measurement_report",
    ("F1AP", "UEContextSetupRequest"): "handover_intra", ("E1AP", "BearerContextModificationRequest"): "bearer_modify",
    ("XnAP", "HandoverRequest"): "handover_inter", ("NGAP", "PathSwitchRequest"): "path_switch",
    ("F1AP", "UEContextReleaseCommand"): "context_release",
    ("RRC", "RRCReestablishmentRequest"): "reestablishment", ("F1AP", "UEContextReleaseRequest"): "rlf_release",
}


def _base(m) -> dict:
    kv = dict(KV.findall(m.group("kv") or ""))
    ev = {"ts": m.group("ts"), "node": m.group("node"), "proto": m.group("proto"), "msg": m.group("msg"),
          "ue": int(kv.pop("ue")) if "ue" in kv else None, "cell": int(kv.pop("cell")) if "cell" in kv else None}
    ev["category"] = CATEGORY.get((ev["proto"], ev["msg"]), "other")
    ev["fields"] = kv
    return ev


def parse_f1ap(m) -> dict:
    ev = _base(m)
    if ev["fields"].get("cause") == "radio-link-failure":
        ev["category"] = "rlf_release"
    return ev


PARSERS: Dict[str, Callable] = {"F1AP": parse_f1ap, "NGAP": _base, "E1AP": _base, "XnAP": _base, "RRC": _base}


def parse_line(line: str) -> Optional[dict]:
    m = LINE.match(line.strip())
    if not m:
        return None
    return PARSERS.get(m.group("proto"), _base)(m)


def parse_log(text: str) -> List[dict]:
    return [e for e in (parse_line(l) for l in text.splitlines() if l.strip()) if e]


# ---------------------------------------------------------------- feature tables
SCHEMA = """
CREATE TABLE IF NOT EXISTS kpi_window (campaign TEXT, window INTEGER, ts TEXT, cell INTEGER, gnb TEXT, ue_share REAL,
  handovers_out INTEGER, handovers_in INTEGER, ping_pongs INTEGER, rlf INTEGER, mean_rsrp_dbm REAL, mean_sinr_db REAL,
  prb_util_pct REAL);
CREATE TABLE IF NOT EXISTS alarm (campaign TEXT, window INTEGER, ts TEXT, cell INTEGER, gnb TEXT, alarm TEXT, severity TEXT,
  text TEXT);
CREATE TABLE IF NOT EXISTS iface_event (campaign TEXT, window INTEGER, ts TEXT, node TEXT, proto TEXT, msg TEXT,
  category TEXT, ue INTEGER, cell INTEGER, cause TEXT);
"""
KPI_COLS = ["campaign", "window", "ts", "cell", "gnb", "ue_share", "handovers_out", "handovers_in", "ping_pongs", "rlf",
            "mean_rsrp_dbm", "mean_sinr_db", "prb_util_pct"]
ALARM_COLS = ["campaign", "window", "ts", "cell", "gnb", "alarm", "severity", "text"]
EVENT_COLS = ["campaign", "window", "ts", "node", "proto", "msg", "category", "ue", "cell", "cause"]


class FeatureStore:
    def __init__(self, path: Path = FEATURES_DB):
        self.path = str(path)
        with sqlite3.connect(self.path) as db:
            db.executescript(SCHEMA)

    def _insert(self, table: str, cols: List[str], rows: Iterable[dict]):
        with sqlite3.connect(self.path) as db:
            db.executemany(f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                           [tuple(r.get(c) for c in cols) for r in rows])

    def drop_campaign(self, campaign: str):
        with sqlite3.connect(self.path) as db:
            for t in ("kpi_window", "alarm", "iface_event"):
                db.execute(f"DELETE FROM {t} WHERE campaign=?", (campaign,))

    def load_kpi(self, rows): self._insert("kpi_window", KPI_COLS, rows)
    def load_alarms(self, rows): self._insert("alarm", ALARM_COLS, rows)
    def load_events(self, rows): self._insert("iface_event", EVENT_COLS, rows)

    def query(self, sql: str, params: tuple = ()) -> List[dict]:
        with sqlite3.connect(self.path) as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute(sql, params).fetchall()]


# ---------------------------------------------------------------- log search
class LogSearch:
    """Text + vector search over log lines and alarm texts. Locally TF-IDF vectors with cosine similarity; on AWS an
    OpenSearch Serverless collection with embeddings (aws_adapters.OpenSearchLogSearch)."""

    def __init__(self):
        self.docs: List[dict] = []
        self._vec = self._mat = None

    def index(self, docs: Iterable[dict]):
        self.docs.extend(docs)
        self._vec = None

    def _fit(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(token_pattern=r"[A-Za-z0-9_\-=]+", sublinear_tf=True)
        self._mat = self._vec.fit_transform([d["text"] for d in self.docs])

    def search(self, query: str, k: int = 5, where: Optional[Dict[str, object]] = None) -> List[dict]:
        if not self.docs:
            return []
        if self._vec is None:
            self._fit()
        import numpy as np
        sims = (self._mat @ self._vec.transform([query]).T).toarray().ravel()
        order = np.argsort(-sims)
        out = []
        for i in order:
            d = self.docs[i]
            if sims[i] <= 0:
                break
            if where and any(d.get(k2) != v for k2, v in where.items()):
                continue
            out.append({**d, "score": round(float(sims[i]), 3)})
            if len(out) == k:
                break
        return out


# ---------------------------------------------------------------- pipeline
def _window_of(ts: str, campaign_start: str) -> int:
    from datetime import datetime
    f = "%Y-%m-%dT%H:%M:%S.000Z"
    return int((datetime.strptime(ts, f) - datetime.strptime(campaign_start, f)).total_seconds() // 60)


class Ingestion:
    """One implementation per modality. `ingest(campaign)` is what the lab's telemetry sink calls: KPI and alarm records
    go onto streams and a consumer lands them in the feature tables; logs go to the lake, are parsed, land as events and
    are indexed for search."""

    def __init__(self, bus=None, lake=None, features=None, search=None):
        self.bus = bus or StreamBus()
        self.lake = lake or LogLake()
        self.features = features or FeatureStore()
        self.search = search or LogSearch()
        self.counts = {"kpi": 0, "alarms": 0, "log_lines": 0, "events": 0, "indexed": 0}

    def ingest(self, camp) -> dict:
        self.features.drop_campaign(camp.id)
        self.counts["kpi"] += self.bus.put_records("kpi", camp.kpi)
        self.counts["alarms"] += self.bus.put_records("alarms", camp.alarms)
        # stream consumer (Kinesis -> Lambda/Firehose -> Glue tables on AWS)
        self.features.load_kpi(camp.kpi)
        self.features.load_alarms(camp.alarms)
        docs = [{"id": f"{camp.id}/alarm/{i}", "text": a["text"], "campaign": camp.id, "cell": a["cell"], "kind": "alarm"}
                for i, a in enumerate(camp.alarms)]
        if camp.logs:
            self.lake.put(f"campaigns/{camp.id}/oran-interfaces.log", "\n".join(camp.logs))
            events = parse_log("\n".join(camp.logs))
            start = camp.kpi[0]["ts"] if camp.kpi else events[0]["ts"]
            for e in events:
                e["campaign"], e["window"], e["cause"] = camp.id, _window_of(e["ts"], start), e["fields"].get("cause")
            self.features.load_events(events)
            self.counts["log_lines"] += len(camp.logs)
            self.counts["events"] += len(events)
            # index the lines that carry operational meaning (failures, handover preparation), not every attach
            docs += [{"id": f"{camp.id}/log/{i}", "text": l, "campaign": camp.id, "cell": e["cell"], "kind": "log"}
                     for i, (l, e) in enumerate(zip(camp.logs, events))
                     if e["category"] in ("rlf_release", "reestablishment", "handover_inter", "handover_intra")]
        self.search.index(docs)
        self.counts["indexed"] += len(docs)
        return dict(self.counts)
