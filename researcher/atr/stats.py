"""Statistics for the Critic. Local by default; runs in AgentCore Code Interpreter in AWS mode.

`paired_effect_source()` returns the self-contained function source, so exactly the same code is executed in
both places. `set_executor` swaps where it runs (see aws_adapters.CodeInterpreterStats).
"""
import inspect
import json
from typing import Callable, Optional


def _paired_effect_impl(baseline, treatment, n_boot=4000, seed=0):
    """Improvement = baseline - treatment per seed (positive: treatment fits better)."""
    import numpy as np
    d = np.asarray(baseline, float) - np.asarray(treatment, float)
    sd = d.std(ddof=1) if len(d) > 1 else 0.0
    rng = np.random.default_rng(seed)
    boots = rng.choice(d, (n_boot, len(d))).mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "n": int(len(d)),
        "mean_improvement": float(d.mean()),
        "cohens_d": float(d.mean() / sd) if sd > 0 else (99.0 if d.mean() > 0 else 0.0),
        "ci95": [float(lo), float(hi)],
        "win_rate": float(np.mean(d > 0)),
    }


_executor: Optional[Callable[[list, list], dict]] = None
calls = {"local": 0, "remote": 0}


def set_executor(fn: Optional[Callable[[list, list], dict]]) -> None:
    global _executor
    _executor = fn


def paired_effect(baseline, treatment) -> dict:
    baseline, treatment = [float(x) for x in baseline], [float(x) for x in treatment]
    if _executor is not None:
        calls["remote"] += 1
        return _executor(baseline, treatment)
    calls["local"] += 1
    return _paired_effect_impl(baseline, treatment)


def paired_effect_source() -> str:
    return inspect.getsource(_paired_effect_impl)


def code_interpreter_script(baseline, treatment) -> str:
    """Program to send to a sandboxed interpreter: the function source plus a call that prints JSON."""
    return (paired_effect_source() + f"\nimport json\nprint(json.dumps(_paired_effect_impl("
            f"{json.dumps(baseline)}, {json.dumps(treatment)})))\n")
