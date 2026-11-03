"""Follow-up experiments an engineer asked for ("request another experiment" on a card).

A finding is re-tested on more fresh seeds than the original run used. For a fault finding that means the saved field log
(the card keeps its path) with the claimed setting against inventory, plus the simulator tables for tilt; for an exploration
insight it is `Explorer.retest`. The result is attached to the card; the engineer decides what it means.
"""
from pathlib import Path

import pandas as pd

from . import stats


def retest_fault_card(card: dict, twin, inventory: dict, seeds=range(201, 233)) -> dict:
    from .agents import MIN_IMPROVEMENT, MIN_WIN_RATE
    from .experiment import ExperimentRunner
    log = card.get("field_log")
    if not log or not Path(log).exists():
        return {"error": "this card has no saved field log, so its evidence cannot be re-tested"}
    runner = ExperimentRunner(twin, inventory, pd.read_csv(log))
    c = card["claim"]
    change = {"tilt": {"tilt": float(c["value"])}, "azimuth": {"azimuth": float(c["value"])},
              "cell_off": {"enabled": False}}[c["kind"]]
    cells = {str(c["cell"]): change}
    scores = lambda r: [p["score"] for p in r["per_seed"]]   # noqa: E731
    base, treat = runner.run({"cells": {}}, list(seeds)), runner.run({"cells": cells}, list(seeds))
    out = {"seeds": len(list(seeds)), "effect": stats.paired_effect(scores(base), scores(treat))}
    out["still_holds"] = out["effect"]["ci95"][0] > MIN_IMPROVEMENT and out["effect"]["win_rate"] >= MIN_WIN_RATE
    if c["kind"] == "tilt":
        sb = runner.run({"cells": {}, "instrument": "simulator"}, list(seeds))
        st = runner.run({"cells": cells, "instrument": "simulator"}, list(seeds))
        out["second_instrument_effect"] = stats.paired_effect(scores(sb), scores(st))
    return out


def run_one(card: dict, request: str, runtime) -> dict:
    """Dispatch on card type. `runtime` is a local runtime (twin, inventory, explorer)."""
    if card.get("type") == "network_insight":
        if card["claim"]["kind"] == "handover_rule":
            from .explore_rule import RuleExplorer
            b = card["claim"]["baseline_rule"]
            x = RuleExplorer(runtime.researcher.gw, runtime.researcher.inventory, runtime.store, audit=runtime.mesh.audit,
                             baseline_rule=(b["hyst"], b["ttt"]), emit=lambda m: None)
            return x.retest(card)
        return runtime.explorer.retest(card)
    r = runtime.researcher
    return retest_fault_card(card, r.runner.twin, r.inventory)
