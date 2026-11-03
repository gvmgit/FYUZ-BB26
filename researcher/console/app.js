// Review console: talks to /api/* on the local server (atr/serve.py) or to API Gateway + the console Lambda on AWS.
// config.json: {"apiBase": "", "cognito": null}  or  {"apiBase": "https://.../prod", "cognito": {"domain", "clientId", "redirectUri"}}
let CFG = { apiBase: "", cognito: null }, TOKEN = null, CARDS = [], SEL = null;
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

async function boot() {
  try { CFG = { ...CFG, ...(await (await fetch("config.json")).json()) }; } catch (e) { /* local defaults */ }
  if (CFG.cognito) {
    const m = location.hash.match(/id_token=([^&]+)/);
    if (m) { TOKEN = m[1]; history.replaceState(null, "", location.pathname); }
    if (!TOKEN) {
      const c = CFG.cognito;
      location.href = `${c.domain}/oauth2/authorize?response_type=token&client_id=${c.clientId}&scope=openid+email&redirect_uri=${encodeURIComponent(c.redirectUri)}`;
      return;
    }
  }
  $("filter").onchange = renderList;
  await load();
}

async function api(path, opts = {}) {
  const headers = { "Content-Type": "application/json", ...(TOKEN ? { Authorization: TOKEN } : {}) };
  const r = await fetch(CFG.apiBase + path, { ...opts, headers });
  const body = await r.json();
  if (!r.ok) throw new Error(body.error || r.statusText);
  return body;
}

const accepted = (c) => (c.engineer_verdicts || []).some((v) => v.verdict === "accept");
const reviewed = (c) => (c.engineer_verdicts || []).length > 0;

async function load() {
  CARDS = await api("/api/findings");
  CARDS.sort((a, b) => (b.headline - a.headline) || (b.created - a.created));
  renderList();
  const wanted = new URLSearchParams(location.search).get("card");   // deep link: /console/?card=<id>
  if (!SEL && CARDS.length) select(wanted && CARDS.some((c) => c.id === wanted) ? wanted : CARDS[0].id);
  const kind = (CARDS.find((c) => c.headline) || CARDS[0] || { claim: { kind: "tilt" } }).claim.kind;
  loadGraph(kind);
}

function renderList() {
  const f = $("filter").value;
  const rows = CARDS.filter((c) => f === "all" || (f === "confirmed" && c.verdict === "confirmed") || (f === "pending" && c.verdict === "confirmed" && !reviewed(c)));
  $("list").innerHTML = rows.map((c) => `<button data-id="${esc(c.id)}" class="${c.id === SEL ? "sel" : ""}">
    <b>${esc(c.title)}</b><br><span class="tag ${c.verdict.startsWith("confirmed") ? "confirmed" : c.verdict.startsWith("refuted") ? "refuted" : "review"}">${esc(c.verdict)}</span>
    ${accepted(c) ? '<span class="tag confirmed">accepted</span>' : ""}${c.lab_replication ? `<span class="tag">${c.lab_replication.passed ? "lab replicated" : "lab: not replicated"}</span>` : ""}</button>`).join("") || '<p class="muted">No findings.</p>';
  document.querySelectorAll("#list button").forEach((b) => (b.onclick = () => select(b.dataset.id)));
}

async function loadGraph(kind) {
  try {
    const g = await api(`/api/graph?kind=${encodeURIComponent(kind)}`);
    const s = g.strategy;
    $("graph").innerHTML = `<div class="muted">${Object.entries(g.summary).map(([k, v]) => `${esc(k)} ${v}`).join(" · ") || "empty"}</div>
      <p><b>Test strategy for ${esc(kind.replace("_", " "))} faults</b> (${s.findings} finding${s.findings === 1 ? "" : "s"})</p>
      <ol class="steps">${(s.strategy || []).map((x) => `<li>${esc(x)}</li>`).join("")}</ol>`;
  } catch (e) { $("graph").innerHTML = `<span class="err">${esc(e.message)}</span>`; }
}

function checks(c) {
  return `<table>${c.checks.map((k) => `<tr><td>${k.passed === null ? "–" : k.passed ? "✔" : "✘"}</td><td>${esc(k.name)}</td><td class="muted">${esc(k.detail)}</td></tr>`).join("")}</table>`;
}

function observation(c) {
  const o = c.observation; if (!o) return "";
  const p = o.patterns.find((x) => x.cell === c.claim.cell) || o.patterns[0];
  const mods = ["kpi", "interface_logs", "alarms", "rf_survey"].map((m) => {
    const d = p.modalities[m]; if (!d) return "";
    const top = Object.entries(d.top || {}).slice(0, 3).map(([k, v]) => `${esc(k)} ${typeof v === "object" ? `${v.seen} vs ${v.expected}` : v}`).join("<br>");
    return `<div class="mod ${d.flag ? "on" : ""}"><b>${esc(m.replace("_", " "))}</b> ${d.flag ? "flagged" : "quiet"}<br><span class="muted">${top}</span></div>`;
  }).join("");
  const ev = (p.log_evidence || []).map((l) => `<li><code>${esc(l.text)}</code></li>`).join("");
  return `<h3>Cross-modal observation (cell ${p.cell})</h3><p class="muted">${esc(o.description)}</p><div class="grid">${mods}</div>
    <p class="muted">Anomaly model: ${Math.round(100 * p.anomaly_model_share)}% of windows anomalous.</p>${ev ? `<p class="muted">Log evidence:</p><ul>${ev}</ul>` : ""}`;
}

function replication(c) {
  const r = c.lab_replication; if (!r) return "";
  const rows = Object.entries(r.profiles).map(([p, v]) => `<tr><td>${esc(p)}</td><td>${v.claim_similarity.toFixed(2)}</td><td>${Object.entries(v.rival_similarity).map(([k, s]) => `${esc(k)} ${s.toFixed(2)}`).join(", ")}</td></tr>`).join("");
  const st = Object.entries(r.rival_status).map(([k, v]) => `<li><b>${esc(k)}</b>: ${esc(v)}</li>`).join("");
  return `<h3>Lab replication ${r.passed ? "passed" : "did not pass"}</h3><p class="muted">${esc(r.note)}. Pre-registered ${esc(r.preregistration.registered_at)} (hash ${esc(r.preregistration.hash.slice(0, 12))}).</p>
    <table><tr><th>traffic profile</th><th>claim reproduces signature</th><th>rivals</th></tr>${rows}</table><ul>${st}</ul>`;
}

function artefacts(c) {
  const a = c.artefacts || []; if (!a.length) return "";
  return `<h3>Artefacts</h3><table>${a.map((x) => `<tr><td>${esc(x.type.replaceAll("_", " "))}</td><td>${
    x.type === "early_warning_indicator" ? `${esc(x.rule)} — <b>${esc(x.status)}</b> (false alarms ${x.validation.false_alarm.fired}/${x.validation.false_alarm.of}, detected ${x.validation.detection.fired}/${x.validation.detection.of})`
    : x.type === "test_strategy" ? `<ol class="steps">${x.strategy.map((s) => `<li>${esc(s)}</li>`).join("")}</ol>`
    : x.type === "fine_tuning_sample" ? `corpus size ${x.corpus_size}; ${esc(x.used_for_training_when)}` : esc(x.status || "")}</td></tr>`).join("")}</table>`;
}

function protocolTimeline(c) {
  const h = (c.protocol || {}).history; if (!h) return "";
  return `<h3>Protocol run ${esc(c.protocol.run_id)}</h3><p class="muted">${esc(c.protocol.engine || "")}</p><div class="timeline">${h.map((s) => `<span title="${esc(s.status)}">${esc(s.state)} ${s.seconds}s</span>`).join("")}</div>`;
}

function select(id) {
  SEL = id; renderList();
  const c = CARDS.find((x) => x.id === id); if (!c) return;
  const e = c.evidence.effect, esc_ = c.escalation || {};
  const verdicts = (c.engineer_verdicts || []).map((v) => `<li><b>${esc(v.verdict.replace("_", " "))}</b> by ${esc(v.by)}${v.comment ? ": " + esc(v.comment) : ""}</li>`).join("");
  $("detail").innerHTML = `<h2>${esc(c.title)}</h2><p>${esc(c.claim.text)}</p>
    <p class="muted">Effect ${e.mean_improvement.toFixed(2)} (95% CI ${e.ci95[0].toFixed(2)} to ${e.ci95[1].toFixed(2)}, wins ${Math.round(100 * e.win_rate)}%)</p>
    ${checks(c)}${observation(c)}${replication(c)}${artefacts(c)}${protocolTimeline(c)}
    <div class="hitl"><b>Human in the loop:</b> ${esc((esc_.status || "").replaceAll("_", " "))}. ${esc(esc_.proposed_action || "")}
      ${verdicts ? `<ul>${verdicts}</ul>` : "<p class='muted'>No verdict yet.</p>"}
      <form id="vf">${CFG.cognito ? "" : '<input name="by" placeholder="your name" required>'}
        <input name="comment" placeholder="comment"><input name="request" placeholder="if requesting: what to test next">
        <button name="verdict" value="accept">Accept</button> <button name="verdict" value="reject">Reject</button>
        <button name="verdict" value="request_experiment">Request another experiment</button></form><p id="vmsg" class="muted"></p></div>`;
  $("vf").onsubmit = async (ev) => {
    ev.preventDefault();
    const f = new FormData(ev.target);
    try {
      await api("/api/verdict", { method: "POST", body: JSON.stringify({ card_id: c.id, verdict: ev.submitter.value, by: f.get("by"), comment: f.get("comment"), request: f.get("request") }) });
      $("vmsg").textContent = "Recorded."; await load(); select(c.id);
    } catch (err) { $("vmsg").innerHTML = `<span class="err">${esc(err.message)}</span>`; }
  };
}

boot();
