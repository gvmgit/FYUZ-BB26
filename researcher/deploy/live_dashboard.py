"""Live dashboard of the Telecom Researcher running on AWS: agents, messages, experiments through AgentCore Gateway,
the Critic's verdicts and the published findings, read from DynamoDB every few seconds.

    python deploy/live_dashboard.py [--port 8770]        then open http://localhost:8770
Bound to 127.0.0.1 only. Credentials come from the `atr` login and refresh themselves (deploy/aws_session.py).
"""
import json
import sys
import time
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import aws_session  # noqa: E402
from boto3.dynamodb.conditions import Attr  # noqa: E402

OUT = json.load(open(Path(__file__).parent / "cdk" / "outputs.json"))["TelecomResearcher"]
GAP = 120          # seconds of silence that separate two runs
CACHE = {"t": 0, "items": [], "seen": set(), "since": time.time() - 6 * 3600}


def _num(v):
    return float(v) if isinstance(v, Decimal) else v


def _scan(table, attr, after):
    t = aws_session.resource("dynamodb").Table(table)
    items, kw = [], {"FilterExpression": Attr(attr).gt(Decimal(str(after)))}
    while True:
        r = t.scan(**kw)
        items += r["Items"]
        if "LastEvaluatedKey" not in r:
            return items
        kw["ExclusiveStartKey"] = r["LastEvaluatedKey"]


def audit_items():
    """Incremental: keep every audit record of the last 6 hours, fetch only new ones."""
    try:
        new = [i for i in _scan(OUT["AuditTable"], "epoch", CACHE["since"] - 5) if i["id"] not in CACHE["seen"]]
    except Exception as e:  # noqa: BLE001
        if "Expired" in str(e):
            aws_session.expire_now()
        raise
    for i in new:
        CACHE["seen"].add(i["id"])
        i["epoch"] = _num(i["epoch"])
    CACHE["items"] = sorted(CACHE["items"] + new, key=lambda i: i["epoch"])
    if CACHE["items"]:
        CACHE["since"] = max(CACHE["since"], CACHE["items"][-1]["epoch"] - 10)
    return CACHE["items"]


def state():
    items = audit_items()
    if not items:
        return {"status": "idle"}
    start = 0
    for k in range(1, len(items)):                     # the latest run starts after the last long silence
        if items[k]["epoch"] - items[k - 1]["epoch"] > GAP:
            start = k
    run = items[start:]
    now = time.time()
    t0, t_last = run[0]["epoch"], run[-1]["epoch"]
    published = next((i for i in reversed(run) if i.get("kind") == "published"), None)
    status = "done" if published else ("running" if now - t_last < 30 else "stopped")
    calls = [i for i in run if i.get("kind") == "tool_call"]
    msgs = [i for i in run if i.get("kind") == "agent_message"]
    claims_ev = next((i for i in reversed(run) if i.get("kind") == "claims"), None)
    claim_ids = list(claims_ev["claims"]) if claims_ev else sorted({i["claim"] for i in run if i.get("kind") == "preregistration"})
    verdicts = {}
    for i in run:
        if i.get("kind") == "verdict":
            verdicts[i["claim"]] = {"status": i["status"], "reason": i.get("reason", "")}
    asked = {}
    for m in msgs:
        parts = str(m.get("summary", "")).split()
        if len(parts) >= 2 and parts[-1] in claim_ids:
            asked[parts[-1]] = f"{parts[0]} requested"
    claims = [{"id": c, "status": verdicts.get(c, {}).get("status") or asked.get(c, "proposed"),
               "reason": verdicts.get(c, {}).get("reason", "")} for c in claim_ids]
    active = None
    if status == "running":
        recent = [i for i in run if now - i["epoch"] < 8 and i.get("kind") in ("tool_call", "agent_message", "verdict", "claims")]
        if recent:
            r = recent[-1]
            active = r.get("to") if r.get("kind") == "agent_message" else r.get("agent")
    buckets = {}
    for cl in calls:
        b = int((cl["epoch"] - t0) // 10)
        buckets[b] = buckets.get(b, 0) + 1
    timeline = [buckets.get(b, 0) for b in range(0, int((t_last - t0) // 10) + 1)]
    feed = []
    for i in [i for i in run if not (i.get("kind") == "decision" and i.get("decision") == "ALLOW")][-60:]:
        k = i.get("kind")
        if k == "agent_message":
            feed.append({"t": i["ts"][11:19], "who": i["from"], "to": i["to"], "text": str(i.get("summary"))[:140], "type": "msg"})
        elif k == "tool_call":
            label = (i.get("args") or {}).get("config", {}).get("name", "") if isinstance(i.get("args"), dict) else ""
            feed.append({"t": i["ts"][11:19], "who": i.get("agent"), "text": f"{i.get('tool')}  {label}", "type": "call",
                         "ok": bool(i.get("ok")), "err": str(i.get("error") or "")[:120], "secs": _num(i.get("seconds"))})
        elif k == "verdict":
            feed.append({"t": i["ts"][11:19], "who": "critic", "text": f"{i['claim']}: {i['status'].upper()}", "type": "verdict",
                         "status": i["status"]})
        elif k == "published":
            feed.append({"t": i["ts"][11:19], "who": "orchestrator", "text": f"PUBLISHED: {i.get('headline')}", "type": "pub"})
        elif k == "decision" and i.get("decision") == "DENY":
            feed.append({"t": i["ts"][11:19], "who": str(i.get("principal", "")).split("::")[-1],
                         "text": f"DENIED by policy: {i.get('action')} on {i.get('resource')} "
                                 f"({', '.join(i.get('policies') or []) or 'no permit'})", "type": "deny"})
    cards = []
    if published or status != "running":
        for c in _scan(OUT["FindingsTable"], "created", t0 - 5):
            cards.append({"title": c.get("title"), "verdict": c.get("verdict"), "headline": bool(c.get("headline")),
                          "claim": "{kind}-{cell}".format(**{k: (c.get("claim") or {}).get(k) for k in ("kind", "cell")})})
        cards.sort(key=lambda c: (not c["headline"], c["verdict"] != "confirmed"))
    if cards and not published:          # runs from before the 'published'/'verdict' audit events: use the cards
        status = "done"
        published = {"headline": next((c["title"] for c in cards if c["headline"]), None)}
        final = {c["claim"]: ("accepted" if c["verdict"] == "confirmed" else "rejected") for c in cards if c["claim"]}
        for cl in claims:
            if cl["id"] in final and cl["status"] not in ("accepted", "rejected"):
                cl["status"] = final[cl["id"]]
    tok = (published or {}).get("tokens") or {}
    return {
        "status": status, "started": run[0]["ts"][11:19], "elapsed": round((published and _num(published.get("seconds"))) or (t_last - t0)),
        "experiments": sum(1 for c in calls if c.get("ok")), "failed": sum(1 for c in calls if not c.get("ok")),
        "messages": len(msgs), "denied": sum(1 for i in run if i.get("kind") == "decision" and i.get("decision") == "DENY"),
        "decisions": sum(1 for i in run if i.get("kind") == "decision"), "trace_id": run[-1].get("trace_id"),
        "claims": claims, "active": active, "timeline": timeline[-60:], "feed": feed[-26:][::-1],
        "symptom": (claims_ev or {}).get("symptom"), "suspicion": {str(k): _num(v) for k, v in ((claims_ev or {}).get("suspicion") or {}).items()},
        "headline": (published or {}).get("headline"), "tokens": {k: _num(v) for k, v in tok.items()}, "cards": cards,
        "model": "Claude Sonnet 5 on Amazon Bedrock", "region": aws_session.REGION,
    }


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Researcher Live Dashboard</title><style>
:root{--bg:#ffffff;--panel:#f4f6fb;--panel2:#ffffff;--line:#dce6f0;--fg:#1e1e23;--mut:#5a5c64;--teal:#0f5fdc;--amber:#d27800;--red:#c8283c;--blue:#0f5fdc;--violet:#5f1ebe;--green:#1a8c50}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.4 Arial,Helvetica,sans-serif}
header{display:flex;align-items:center;gap:14px;padding:14px 22px 12px;border-bottom:1px solid var(--line);border-top:6px solid;border-image:linear-gradient(90deg,#5f1ebe,#0f5fdc,#3c91ff) 1}
.logo{color:#0f5fdc;font-weight:800;font-size:20px;letter-spacing:-.02em}.logo:after{content:"";display:inline-block;width:1px;height:20px;background:#0f5fdc;margin-left:12px;vertical-align:-4px}
h1{font-size:18px;margin:0;font-weight:700;color:#1e1e23}
.pill{margin-left:auto;padding:5px 14px;border-radius:14px;font-weight:700;font-size:12px}.running{background:#e6f6ed;color:#1a8c50}
.done{background:#e8efff;color:#0f5fdc}.stopped,.idle{background:#fff3e0;color:#d27800}.meta{color:var(--mut);font-size:12px}
main{display:grid;grid-template-columns:1.25fr 1fr;gap:14px;padding:14px 22px}
.kpis{grid-column:1/3;display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
.kpi{background:var(--panel);border:1px solid var(--line);border-left:5px solid #5f1ebe;border-radius:10px;padding:12px 16px}.kpi b{display:block;font-size:30px;color:#5f1ebe}.kpi span{color:var(--mut);font-size:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;min-height:120px}.panel h2{margin:0 0 10px;font-size:13px;letter-spacing:.05em;color:#5f1ebe;text-transform:uppercase}
svg text{font-family:Arial}.claim{display:flex;justify-content:space-between;align-items:center;background:#fff;border:1px solid var(--line);border-radius:8px;padding:7px 10px;margin:6px 0}
.chip{font-size:11px;font-weight:700;padding:3px 10px;border-radius:10px;background:#eef1f7;color:var(--mut)}.accepted{background:#e6f6ed;color:#1a8c50}
.rejected{background:#fdecee;color:#c8283c}.needs{background:#fff3e0;color:#d27800}
.feed{font:12.5px/1.5 Consolas,monospace;max-height:360px;overflow:auto;background:#fff;border:1px solid var(--line);border-radius:8px;padding:6px 10px}.feed div{padding:2px 0;border-bottom:1px solid #eef1f7}
.t{color:#8a8f9c}.ok{color:#1a8c50}.err{color:#c8283c}.headline{grid-column:1/3;background:#e6f6ed;border:1px solid #9fd6b6;border-left:6px solid #1a8c50;border-radius:10px;padding:14px 18px;font-size:18px;font-weight:700;color:#11603a}
.card{display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #eef1f7}.bars{display:flex;align-items:flex-end;gap:3px;height:90px}
.bars i{flex:1;background:linear-gradient(#3c91ff,#5f1ebe);border-radius:2px 2px 0 0;min-height:2px}footer{color:#8a8f9c;font-size:11px;padding:4px 22px 16px}
</style></head><body>
<header><span class="logo">HCLTech</span><h1>Autonomous Telecom Researcher &middot; live on AWS</h1>
<span class="meta" id="meta"></span><span class="pill idle" id="status">CONNECTING</span></header>
<main>
<div class="kpis"><div class="kpi"><b id="k_exp">0</b><span>experiments via AgentCore Gateway</span></div>
<div class="kpi"><b id="k_msg">0</b><span>agent-to-agent messages</span></div><div class="kpi"><b id="k_claims">0</b><span>claims under test</span></div>
<div class="kpi"><b id="k_time">0s</b><span>elapsed</span></div><div class="kpi"><b id="k_fail">0</b><span>failed calls / policy denials</span></div></div>
<div class="headline" id="headline" style="display:none"></div>
<section class="panel"><h2>Agents</h2><svg viewBox="0 0 640 300" width="100%" id="flow"></svg></section>
<section class="panel"><h2>Claims and the Critic's verdicts</h2><div id="symptom" class="meta"></div><div id="claims"></div></section>
<section class="panel"><h2>Live activity</h2><div class="feed" id="feed"></div></section>
<section class="panel"><h2>Experiments per 10 s</h2><div class="bars" id="bars"></div><h2 style="margin-top:16px">Finding cards (DynamoDB)</h2><div id="cards" class="meta">published when the investigation ends</div></section>
</main><footer>Reads the DynamoDB audit trail and finding cards every 3 s &middot; Strands agents in Amazon Bedrock AgentCore Runtime &middot; experiments: AgentCore Gateway &rarr; Lambda (MAVERIC twin)</footer>
<script>
const C={orchestrator:"#0f5fdc",scientist:"#5f1ebe",experimenter:"#d27800",critic:"#c8283c"};
const N={orchestrator:[320,150,"Orchestrator"],scientist:[110,60,"Scientist"],experimenter:[530,60,"Experimenter"],critic:[110,240,"Critic"],
gateway:[530,240,"AgentCore Gateway"]};const esc=s=>String(s??"").replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));
function flow(active){const a=active||"";let s="";const edge=(f,t,on)=>{const[x1,y1]=N[f],[x2,y2]=N[t];
s+=`<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${on?"#0f5fdc":"#c9d3e3"}" stroke-width="${on?4:2}"/>`};
edge("orchestrator","scientist",a=="scientist");edge("orchestrator","experimenter",a=="experimenter");edge("orchestrator","critic",a=="critic");
edge("experimenter","gateway",a=="experimenter");
for(const[k,[x,y,l]]of Object.entries(N)){const on=k==a||(k=="gateway"&&a=="experimenter");const col=C[k]||"#0f5fdc";
s+=`<g><rect x="${x-78}" y="${y-24}" width="156" height="48" rx="12" fill="${on?"#eef4ff":"#ffffff"}" stroke="${on?col:"#c9d3e3"}" stroke-width="${on?3:1.5}"/>`+
`<text x="${x}" y="${y+5}" text-anchor="middle" fill="${on?col:"#1e1e23"}" font-size="15" font-weight="600">${l}</text>`+(on?`<circle cx="${x+66}" cy="${y-14}" r="5" fill="${col}"><animate attributeName="opacity" values="1;.2;1" dur="1s" repeatCount="indefinite"/></circle>`:"")+`</g>`}
s+=`<text x="530" y="292" text-anchor="middle" fill="#5a5c64" font-size="12">&rarr; Lambda: MAVERIC digital twin</text><text x="320" y="200" text-anchor="middle" fill="#5a5c64" font-size="12">Claude Sonnet 5 on Bedrock</text>`;
document.getElementById("flow").innerHTML=s}
async function tick(){try{const r=await fetch("/api/state");const d=await r.json();
const st=document.getElementById("status");st.className="pill "+d.status;st.textContent=(d.status||"").toUpperCase();
if(d.status=="idle"){return}
document.getElementById("meta").textContent=`run started ${d.started} UTC · ${d.model} · ${d.region}`;
k_exp.textContent=d.experiments;k_msg.textContent=d.messages;k_claims.textContent=d.claims.length;k_time.textContent=d.elapsed+"s";k_fail.textContent=d.failed+" / "+d.denied;
flow(d.active);
symptom.textContent=d.symptom?("Symptom: "+d.symptom):"";
claims.innerHTML=d.claims.map(c=>{const cls=c.status=="accepted"?"accepted":c.status=="rejected"?"rejected":c.status.startsWith("needs")?"needs":"";
return `<div class="claim" title="${esc(c.reason)}"><span>${esc(c.id)}</span><span class="chip ${cls}">${esc(c.status.replace("_"," "))}</span></div>`}).join("")||'<div class="meta">waiting for the Scientist...</div>';
feed.innerHTML=d.feed.map(e=>`<div><span class="t">${e.t}</span> <span style="color:${C[e.who]||"#1e1e23"}">${esc(e.who)}</span> `+
(e.type=="msg"?`&rarr; <span style="color:${C[e.to]||"#1e1e23"}">${esc(e.to)}</span>: ${esc(e.text)}`:e.type=="call"?`<span class="${e.ok?"ok":"err"}">${e.ok?"OK":"ERR"}</span> ${esc(e.text)} <span class="t">${e.secs??""}s</span>${e.err?` <span class="err">${esc(e.err)}</span>`:""}`:
`<b class="${e.type=="deny"||e.status=="rejected"?"err":"ok"}">${esc(e.text)}</b>`)+`</div>`).join("");
const mx=Math.max(1,...d.timeline);bars.innerHTML=d.timeline.map(v=>`<i style="height:${100*v/mx}%" title="${v}"></i>`).join("");
const h=document.getElementById("headline");if(d.headline){h.style.display="block";h.textContent="FINDING: "+d.headline+(d.tokens.totalTokens?`   ·   ${d.experiments} experiments, ${d.elapsed}s, ${Math.round(d.tokens.totalTokens/1000)}k tokens`:"")}else{h.style.display="none"}
cards.innerHTML=!d.cards.length?"published when the investigation ends":d.cards.map(c=>`<div class="card"><span>${esc(c.title)}${c.headline?" ★":""}</span><span class="chip ${c.verdict=="confirmed"?"accepted":"rejected"}">${esc(c.verdict)}</span></div>`).join("");
}catch(e){document.getElementById("status").textContent="RECONNECTING"}}
tick();setInterval(tick,3000);
</script></body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/state"):
            try:
                body, code, ctype = json.dumps(state(), default=str).encode(), 200, "application/json"
            except Exception as e:  # noqa: BLE001
                body, code, ctype = json.dumps({"status": "error", "error": str(e)[:200]}).encode(), 500, "application/json"
        else:
            body, code, ctype = PAGE.encode(), 200, "text/html; charset=utf-8"
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8770
    print(f"Live dashboard at http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
