"""The 5-minute hackathon demo, staged to the script, with a standalone HTML report.

    0:00 problem    0:30 symptom + testable claim    1:30 live experiments, heatmaps changing
    2:45 Critic challenges and runs a control         3:45 reveal next to the finding card
    4:30 the numbers
"""
import base64
import html
import time
from pathlib import Path

from . import config, runtime
from .heatmap import render_filmstrip
from .serve import CSS, render_card
from .strands_agents import run_strands_investigation

STAGES = [
    ("0:00", "THE PROBLEM", None),
    ("0:30", "THE SYMPTOM APPEARS - the Scientist writes a testable claim", "[scientist] symptom"),
    ("1:30", "THE EXPERIMENTER RUNS ONE-SETTING-AT-A-TIME EXPERIMENTS", "[experimenter] testing"),
    ("2:45", "THE CRITIC CHALLENGES THE RESULT AND DEMANDS A CONTROL", "[critic] run the same sweep"),
]


def _b64(path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode()


def _filmstrip(rt, card) -> str:
    """Heatmaps changing as the Experimenter sweeps the faulted setting; ends on the best fix."""
    runner = rt.researcher.runner
    if runner is None:
        return ""
    twin, c = runner.twin, card["claim"]
    key = {"tilt": "tilt", "azimuth": "azimuth", "cell_off": "enabled"}[c["kind"]]
    sweep = sorted((s for s in card["evidence"]["sweep"] if c["kind"] != "cell_off"), key=lambda s: s["value"])
    picks = [sweep[i] for i in sorted({0, len(sweep) // 3, 2 * len(sweep) // 3, len(sweep) - 1}) if sweep]
    panels = [("inventory settings", twin.rsrp_grid(runner.settings_for({}))[0])]
    for s in picks:
        cfg = {"cells": {str(c["cell"]): {key: s["value"]}}}
        panels.append((f"{s['label']}\nscore {s['mean']:.1f}", twin.rsrp_grid(runner.settings_for(cfg))[0]))
    best = {key: False if c["kind"] == "cell_off" else c["value"]}
    panels.append(("BEST FIT", twin.rsrp_grid(runner.settings_for({"cells": {str(c["cell"]): best}}))[0]))
    path = config.HEATMAPS / "filmstrip.png"
    render_filmstrip(twin, panels, path, "Twin best-server RSRP as the Experimenter changes one setting")
    return str(path)


def run_demo(fault, mode=None, n_ue=1200, pace: float = 0.0) -> dict:
    t0 = time.time()
    shown = set()

    def emit(msg: str):
        for ts, title, trigger in STAGES[1:]:
            if trigger and msg.startswith(trigger) and ts not in shown:
                shown.add(ts)
                print(f"\n=== {ts}  {title} ===")
        print(msg)
        if pace:
            time.sleep(pace)

    print(f"=== {STAGES[0][0]}  {STAGES[0][1]} ===")
    print("Engineers can test only a few network theories a month. This system runs its own experiments to find out\n"
          "why the network misbehaves, and a Critic tries to disprove every answer before it is published.\n"
          "A teammate has secretly changed something in the network. The researcher sees only the symptom.")
    rt = runtime.build(mode, fault, n_ue, emit)
    try:
        out = run_strands_investigation(rt.researcher, rt.mesh, rt.model_for)
    finally:
        rt.close()
    head = out["headline"]
    truth = rt.world.truth() if rt.world else None

    print("\n=== 3:45  THE REVEAL ===")
    print(f"hidden fault : {truth}")
    located = False
    if head:
        c = head["claim"]
        located = truth is not None and c["kind"] == truth["kind"] and c["cell"] == truth["cell"]
        print(f"finding card : {head['title']}")
        print(f"fault located: {'YES' if located else 'NO'}")
    else:
        print("finding card : none - no claim survived the Critic")
    rs = (head or out["cards"][0])["run_summary"]
    print("\n=== 4:30  THE NUMBERS ===")
    print(f"experiments needed : {out['experiments']}")
    print(f"researcher time    : {rs['wall_seconds']}s    engineer estimate: ~{rs['engineer_estimate_hours']} h "
          f"({rs['engineer_estimate_note']})")
    print(f"agent tokens       : {out['tokens']['totalTokens']} "
          f"({'offline playbook model: 0 by construction' if rt.services.mode == 'local' else 'Bedrock'})")
    print(f"mode               : {rt.services.mode}  services: {rt.services.choice}")

    report = write_report(rt, out, truth, located, time.time() - t0)
    print(f"\nreport: {report}")
    return {"out": out, "truth": truth, "located": located, "report": str(report)}


def write_report(rt, out, truth, located, wall) -> Path:
    head = out["headline"]
    film = _filmstrip(rt, head) if head else ""
    reveal_png = None
    if rt.world and head:
        from .heatmap import render_best_server
        reveal_png = config.HEATMAPS / "reveal_truth.png"
        render_best_server(rt.researcher.runner.twin, rt.world.reveal_rsrp(), reveal_png,
                           "the real network (hidden fault applied)")
    rs = (head or out["cards"][0])["run_summary"]
    cards = "".join(render_card(c, img=lambda p: f'<img src="{_b64(p)}">' if p and Path(p).exists() else "",
                                interactive=False)
                    for c in sorted(out["cards"], key=lambda c: not c["headline"]))
    critic_table = ""
    if head:
        rows = "".join(f"<tr><td>{'✔' if k['passed'] else '✘'}</td><td>{html.escape(k['name'])}</td>"
                       f"<td>{html.escape(k['detail'])}</td></tr>" for k in head["checks"])
        critic_table = f"<p>Checks the Critic made on the winning claim before it could be published:</p><table>{rows}</table>"
    reveal =(f"<div class='grid2'><div class='card'><h3>Hidden answer</h3><pre>{html.escape(str(truth))}</pre></div>"
              f"<div class='card head'><h3>Researcher's finding</h3><p>{html.escape(head['title'])}</p>"
              f"<p><b>{'Fault located' if located else 'Not matched'}</b></p></div></div>") if head else \
        "<p>No claim survived the Critic.</p>"
    page = f"""<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'>
<title>Telecom Researcher Demo</title><style>{CSS}</style>
<h1>Autonomous Telecom Researcher</h1>
<p>Mode <b>{rt.services.mode}</b> · agents on Strands · {html.escape(str(rt.services.choice))}</p>
<h2>0:00 The problem</h2><p>Engineers can test only a few theories a month. Here the system designs and runs its own experiments on the
MAVERIC RF digital twin to explain why the network misbehaves.</p>
<h2>0:30 The symptom and the claim</h2><pre>{html.escape(next((l for l in (head or out['cards'][0])['trace'] if 'symptom:' in l), ''))}</pre>
<h2>1:30 Experiments, heatmaps changing</h2>{f'<img src="{_b64(film)}">' if film else '<p>(filmstrip needs the local twin)</p>'}
<h2>2:45 The Critic challenges</h2>{critic_table}
<pre>{html.escape(chr(10).join(l for l in (head or out['cards'][0])['trace'] if l.startswith('[critic]')))}</pre>
<h2>3:45 Reveal</h2>{reveal}{f'<img src="{_b64(reveal_png)}">' if reveal_png else ''}
<h2>4:30 The numbers</h2><table><tr><td>Experiments needed</td><td>{out['experiments']}</td></tr>
<tr><td>Researcher time</td><td>{rs['wall_seconds']} s</td></tr>
<tr><td>Engineer estimate</td><td>~{rs['engineer_estimate_hours']} h ({html.escape(rs['engineer_estimate_note'])})</td></tr>
<tr><td>Agent tokens</td><td>{out['tokens']['totalTokens']}</td></tr></table>
<h2>Finding cards</h2>{cards}"""
    path = config.RUNS / "demo_report.html"
    path.write_text(page, encoding="utf-8")
    return path
