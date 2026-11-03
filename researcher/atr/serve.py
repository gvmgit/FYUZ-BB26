"""Finding-card page: a tiny stdlib web server over the finding store (stand-in for the Amplify/CloudFront console)."""
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from . import config, modes

CSS = """:root{--bg:#fff;--fg:#1b1f23;--line:#d0d7de;--soft:#f6f8fa;--ok:#dafbe1;--bad:#ffebe9;--warn:#fff8c5;--accent:#1a7f37}
@media(prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--line:#30363d;--soft:#161b22;--ok:#033a16;--bad:#5a1e1e;--warn:#4d3b00;--accent:#3fb950}}
body{font:15px system-ui,sans-serif;max-width:1020px;margin:24px auto;padding:0 16px;color:var(--fg);background:var(--bg)}
.card{border:1px solid var(--line);border-radius:8px;padding:16px;margin:16px 0}.head{border-color:var(--accent);border-width:2px}
.tag{display:inline-block;padding:2px 8px;border-radius:12px;font-size:12px;background:var(--soft)}
.confirmed{background:var(--ok)}.refuted,.ruled{background:var(--bad)}.alternative{background:var(--warn)}
table{border-collapse:collapse}td,th{border:1px solid var(--line);padding:3px 8px;font-size:13px}
img{max-width:100%;border:1px solid var(--line)}pre{background:var(--soft);padding:8px;overflow:auto;font-size:12px}
.grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}@media(max-width:700px){.grid2{grid-template-columns:1fr}}
.hitl{background:var(--warn);padding:6px 10px;border-radius:6px;font-size:13px}h2{margin-top:28px}"""


def _img_src(path):
    if not path:
        return None
    if str(path).startswith("s3://"):
        import boto3
        bucket, key = str(path)[5:].split("/", 1)
        return boto3.client("s3").generate_presigned_url("get_object", Params={"Bucket": bucket, "Key": key},
                                                         ExpiresIn=3600)
    return f"/img/{Path(path).name}"


def _img(path):
    src = _img_src(path)
    return f'<img src="{src}" alt="heatmap">' if src else ""


def _verdict_block(c: dict, interactive: bool) -> str:
    """Engineer verdicts so far, and (on the live page) the form to add one."""
    done = "".join(f"<li><b>{html.escape(v['verdict'].replace('_', ' '))}</b> by {html.escape(v['by'])}"
                   f"{(': ' + html.escape(v['comment'])) if v.get('comment') else ''}"
                   f"{(' (asked: ' + html.escape(v['request']) + ')') if v.get('request') else ''}</li>"
                   for v in c.get("engineer_verdicts", []))
    fups = "".join(f"<li>follow-up {html.escape(f['status'])}"
                   f"{(': ' + html.escape(f['request'])) if f.get('request') else ''}</li>" for f in c.get("follow_ups", []))
    form = (f'<form method="post" action="/verdict"><input type="hidden" name="card_id" value="{html.escape(c["id"])}">'
            '<input name="by" placeholder="your name" required> <input name="comment" placeholder="comment"> '
            '<input name="request" placeholder="if requesting: what to test next"> '
            '<button name="verdict" value="accept">Accept</button> <button name="verdict" value="reject">Reject</button> '
            '<button name="verdict" value="request_experiment">Request another experiment</button></form>'
            if interactive else "")
    if not (done or fups or form):
        return ""
    return (f"<div class='hitl'><b>Engineer review</b>"
            f"{('<ul>' + done + '</ul>') if done else ' - no verdict yet.'}{('<ul>' + fups + '</ul>') if fups else ''}"
            f"{form}</div>")


def _prereg_block(c: dict) -> str:
    p = c.get("preregistration")
    if not p:
        return ""
    met = "met" if p["prediction_met"] else "NOT met"

    def show(v):
        return ", ".join(f"{k}: {show(x) if isinstance(x, dict) else x}" for k, x in v.items() if x is not None)

    return (f"<details><summary>Pre-registered {html.escape(p['registered_at'])} "
            f"(hash {p['hash'][:12]}, {'verified' if p['verified'] else 'TAMPERED'}): prediction {met}</summary>"
            f"<p>Predicted before testing: {html.escape(show(p['predicted']))}.<br>"
            f"Null outcome: {html.escape(p['null_outcome'])}</p></details>")


def nlm_text(op) -> str:
    """The network language model's opinion: a {'model', 'text'} record (AWS mode) or plain text (older cards)."""
    return f"{op.get('text', '')} ({op.get('model', 'network LM')})" if isinstance(op, dict) else str(op)


def render_card(c: dict, img=_img, interactive: bool = True) -> str:
    e, rs = c["evidence"]["effect"], c.get("run_summary", {})
    checks = "".join(f"<tr><td>{'–' if k['passed'] is None else ('✔' if k['passed'] else '✘')}</td>"
                     f"<td>{html.escape(k['name'])}</td><td>{html.escape(k['detail'])}</td></tr>" for k in c["checks"])
    insight = c.get("type") == "network_insight"
    sweep = "".join(f"<tr><td>{html.escape(s['label'])}</td><td>{(-s['mean'] if insight else s['mean']):.2f}</td></tr>"
                    for s in sorted(c["evidence"]["sweep"], key=lambda s: s["mean"])[:6])
    sweep_caption = (c.get("sweep_label", "Scanned interventions, by failure-risk reduction (points)") if insight
                     else "Best sweep values (lower score = closer to field)")
    caveat = f"<p class='hitl'><b>Caveat:</b> {html.escape(c['caveat'])}</p>" if c.get("caveat") else ""
    h2h = "".join(f"<tr><td>{html.escape(i)}</td><td>{v['winner_better_by']:.2f} (CI {v['ci95'][0]:.2f}..{v['ci95'][1]:.2f})</td>"
                  f"<td>{'ruled out' if v['ruled_out'] else 'NOT excluded'}</td></tr>"
                  for i, v in (c.get("head_to_head") or {}).items())
    esc = c.get("escalation")
    tok = c.get("tokens")
    cls = "head" if c["headline"] else ""
    vcls = {"confirmed": "confirmed", "refuted": "refuted"}.get(c["verdict"].split()[0], "alternative")
    return f"""<div class="card {cls}"><h3>{html.escape(c['title'])} <span class="tag {vcls}">{html.escape(c['verdict'])}</span></h3>
<p>{html.escape(c['claim']['text'])}</p>
{f"<p><b>{html.escape(c['narrative'])}</b></p>" if c.get('narrative') else ''}
<p>{c.get('metric_label', 'Failure-risk reduction (points)') if insight else 'Improvement'} {e['mean_improvement']:.2f} (95% CI {e['ci95'][0]:.2f} to {e['ci95'][1]:.2f}, d={e['cohens_d']:.1f}, wins {e['win_rate']:.0%})</p>
<table>{checks}</table>
{_prereg_block(c)}
{f"<details><summary>Head-to-head against other accepted claims</summary><table>{h2h}</table></details>" if h2h else ''}
<details><summary>{sweep_caption}</summary><table>{sweep}</table></details>{caveat}
<div class="grid2">{img(c['heatmaps'].get('inventory'))}{img(c['heatmaps'].get('best_fix'))}</div>
{f'<p class="hitl"><b>Human in the loop:</b> {html.escape(esc["status"].replace("_", " "))}. {html.escape(esc.get("proposed_action") or "")} <i>({html.escape(esc["reason"])})</i></p>' if esc else ''}
{f"<p><b>Model's reading of the coverage heatmap:</b> {html.escape(c['visual_observation'])}</p>" if c.get('visual_observation') else ''}
{f"<p><b>Network language model second opinion:</b> {html.escape(nlm_text(c['network_lm_opinion']))}</p>" if c.get('network_lm_opinion') else ''}
<p style="font-size:12px">{rs.get('experiments_run','?')} experiments in {rs.get('wall_seconds','?')}s vs ~
{rs.get('engineer_estimate_hours','?')} h manually ({html.escape(rs.get('engineer_estimate_note',''))}).
{f"Tokens: {tok.get('totalTokens', 0)}." if tok else ''}</p>
{_verdict_block(c, interactive)}
<details><summary>Agent trace</summary><pre>{html.escape(chr(10).join(c.get('trace', [])))}</pre></details></div>"""


def make_scheduler():
    """Follow-up requests go to SQS in AWS mode and to the persistent local queue otherwise."""
    svc = modes.resolve()
    if svc.aws("scheduler"):
        from .aws_adapters import SqsScheduler
        return SqsScheduler(region=svc.region)
    from .scheduler import LocalScheduler
    return LocalScheduler(config.RUNS / "queue.json")


def make_store():
    svc = modes.resolve()
    if svc.aws("store"):
        from .aws_adapters import DynamoDBStore
        return DynamoDBStore(region=svc.region)
    from .store import SQLiteStore
    return SQLiteStore()


CONSOLE_DIR = config.ROOT / "console"
STATIC_TYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css",
                ".json": "application/json"}


def api_graph(graph, kind: str) -> dict:
    """Shared by the local server and the console Lambda: graph summary + recommended test strategy for a fault class."""
    return {"summary": graph.summary(), "strategy": graph.test_strategy(kind)}


def api_verdict(store, body: dict, who: str, scheduler=None, audit=None) -> dict:
    from .verdicts import record_verdict
    if body["verdict"] == "accept":     # accepting a proposed change is a human approval: a checkpoint decision first
        from .runtime import make_checkpoint
        existing = store.get(body["card_id"]) or {}
        make_checkpoint(run_id=existing.get("run_id"), unit="approval").enforce("Human", who, "ApproveChange", body["card_id"])
    card = record_verdict(store, body["card_id"], body["verdict"], who, body.get("comment", ""), body.get("request", ""),
                          scheduler=scheduler, audit=audit)
    return {"card_id": card["id"], "verdicts": card.get("engineer_verdicts", [])}


class Handler(BaseHTTPRequestHandler):
    store = None

    def _json(self, code: int, obj):
        body = json.dumps(obj, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        """Engineer verdict from the form on a card, or from the console (JSON). Local server bound to 127.0.0.1 only;
        the console on AWS takes the reviewer's identity from Cognito instead of the request."""
        from urllib.parse import parse_qs
        from .audit import FileAudit
        from .verdicts import record_verdict
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode("utf-8")
        if self.path == "/api/verdict":
            try:
                b = json.loads(raw or "{}")
                if not b.get("by"):
                    raise ValueError("a reviewer name is required")
                self._json(200, api_verdict(self.store, b, b["by"], scheduler=make_scheduler(), audit=FileAudit()))
            except (KeyError, ValueError, PermissionError) as e:
                self._json(400, {"error": str(e)})
            return
        body = parse_qs(raw)
        field = lambda k: (body.get(k) or [""])[0].strip()   # noqa: E731
        try:
            if self.path != "/verdict":
                raise KeyError(self.path)
            record_verdict(self.store, field("card_id"), field("verdict"), field("by"), field("comment"),
                           field("request"), scheduler=make_scheduler(), audit=FileAudit())
        except (KeyError, ValueError, PermissionError) as e:
            self.send_error(400, str(e))
            return
        self.send_response(303)
        self.send_header("Location", "/")
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/img/"):
            f = config.HEATMAPS / Path(self.path).name
            if f.exists():
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.end_headers()
                self.wfile.write(f.read_bytes())
                return
            self.send_error(404)
            return
        if self.path == "/console" or self.path.startswith("/console/"):
            name = self.path[len("/console/"):].split("?")[0] or "index.html"
            f = (CONSOLE_DIR / name).resolve()
            if CONSOLE_DIR.resolve() in f.parents and f.exists():
                self.send_response(200)
                self.send_header("Content-Type", STATIC_TYPES.get(f.suffix, "application/octet-stream"))
                self.end_headers()
                self.wfile.write(f.read_bytes())
            else:
                self.send_error(404)
            return
        if self.path.startswith("/api/graph"):
            from urllib.parse import parse_qs, urlparse
            from .graph import make_graph
            kind = (parse_qs(urlparse(self.path).query).get("kind") or ["tilt"])[0]
            self._json(200, api_graph(make_graph(), kind))
            return
        cards = self.store.list()
        if self.path == "/api/findings":
            body, ctype = json.dumps(cards).encode(), "application/json"
        else:
            body = (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>"
                    f"<title>Telecom Researcher Findings</title><style>{CSS}</style><h1>Findings</h1>"
                    f"{''.join(render_card(c) for c in cards) or '<p>No findings yet.</p>'}").encode()
            ctype = "text/html; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def serve(port: int = 8765):
    Handler.store = make_store()
    print(f"Finding cards at http://localhost:{port}   review console at http://localhost:{port}/console/")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
