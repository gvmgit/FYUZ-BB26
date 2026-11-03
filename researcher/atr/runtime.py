"""Assemble the system for the chosen mode. This is the only module that knows about the local/AWS toggle.

Heavy imports (MAVERIC, torch) happen only on the branch that needs the local twin, so an AWS-mode agent runtime
carries none of them.
"""
import os
from dataclasses import dataclass
from typing import Callable, Optional

from . import config, modes, playbook as pb, stats
from .anf import AgentMesh
from .audit import FileAudit
from .checkpoint import AvpPdp, Checkpoint, DynamoRegistry, guarded, local_checkpoint, new_run_id
from .faults import Fault
from .gateway import build_gateway
from .researcher import Researcher
from .scheduler import LocalScheduler
from .store import SQLiteStore
from .topology import default_inventory

LOCAL_SCRIPTS = {"scientist": pb.scientist_script, "experimenter": pb.experimenter_script,
                 "critic": pb.critic_script, "orchestrator": pb.orchestrator_script}
EXPLORE_SCRIPTS = {"scientist": pb.explore_scientist_script, "experimenter": pb.explore_experimenter_script,
                   "critic": pb.explore_critic_script, "orchestrator": pb.explore_orchestrator_script}


class ConfigError(Exception):
    pass


@dataclass
class Runtime:
    services: modes.Services
    researcher: Researcher
    mesh: AgentMesh
    model_for: Callable[[str], object]
    store: object
    scheduler: object
    world: object = None          # local only: holds the hidden fault for the reveal
    code_interpreter: object = None
    explorer: object = None       # the exploration pipeline (no planted fault)

    def close(self):
        if self.code_interpreter is not None:
            self.code_interpreter.close()
        stats.set_executor(None)


def make_checkpoint(mode: Optional[str] = None, run_id: Optional[str] = None, unit: str = "check",
                    trace_id: Optional[str] = None) -> Checkpoint:
    """The checkpoint alone (no agents), for human approvals, registry changes and policy probes. In AWS mode its records
    go to the DynamoDB audit table and to the /atr/audit CloudWatch log group (this runs off AgentCore, so no stdout)."""
    svc = modes.resolve(mode)
    audit = FileAudit()
    if svc.aws("audit"):
        from . import aws_adapters as aws
        audit = aws.TeeAudit(aws.DynamoAudit(region=svc.region), aws.CloudWatchAudit(region=svc.region))
    if svc.aws("policy"):
        cp = Checkpoint(AvpPdp(region=svc.region), DynamoRegistry(region=svc.region), audit, run_id or new_run_id(unit),
                        trace_id, echo=False)
    else:
        cp = local_checkpoint(audit, run_id or new_run_id(unit), echo=False)
        cp.trace_id = trace_id or cp.trace_id
    return cp


def build(mode: Optional[str] = None, fault: Optional[Fault] = None, n_ue: int = 1200,
          emit: Callable[[str], None] = print, twin_bias: Optional[dict] = None, field_seed: int = 12345,
          twin_model_file: Optional[str] = None, twin=None, reality_twin=None, task: str = "investigate",
          explore_objective: str = "network", baseline_rule=(3.0, 3), run_id: Optional[str] = None,
          trace_id: Optional[str] = None) -> Runtime:
    """`twin_bias`/`field_seed`: local blind-test knobs. `twin_model_file` (or env ATR_TWIN_MODEL) selects a fine-tuned
    twin; `twin` passes an already-loaded one. `reality_twin` is what the simulated real network is built from; it
    must stay fixed when the twin under test is being fine-tuned, or reality would move with the model."""
    svc = modes.resolve(mode)
    missing = svc.missing_env()
    if missing:
        raise ConfigError("AWS mode is missing configuration: " +
                          "; ".join(f"{s} needs {', '.join(v)}" for s, v in missing.items()))
    aws = None
    if any(svc.aws(s) for s in modes.SERVICES):
        from . import aws_adapters as aws

    audit = aws.DynamoAudit(region=svc.region) if svc.aws("audit") else FileAudit()
    run_id = run_id or new_run_id(task)
    if svc.aws("policy"):
        cp = Checkpoint(AvpPdp(region=svc.region), DynamoRegistry(region=svc.region), audit, run_id, trace_id)
    else:
        cp = local_checkpoint(audit, run_id, echo=svc.mode == "aws")
        cp.trace_id = trace_id or cp.trace_id
    mesh = AgentMesh(audit=audit, checkpoint=cp)
    store = aws.DynamoDBStore(region=svc.region) if svc.aws("store") else SQLiteStore()
    scheduler = (aws.SqsScheduler(region=svc.region) if svc.aws("scheduler")
                 else LocalScheduler(config.RUNS / "queue.json"))
    inventory = default_inventory()

    world = None
    if svc.aws("gateway"):
        gw = aws.RemoteGateway(os.environ["ATR_GATEWAY_URL"], os.environ.get("ATR_GATEWAY_TOKEN"), audit,
                               policy=mesh.authorize)
        runner = None
    else:
        from .experiment import ExperimentRunner
        from .twin import RFTwin
        from .world import World
        twin = twin or RFTwin.load_or_build(model_file=twin_model_file or os.environ.get("ATR_TWIN_MODEL"))
        world = World(reality_twin or twin, inventory, fault, twin_bias)
        runner = ExperimentRunner(twin, inventory, world.field_observation(n_ue, field_seed))
        gw = build_gateway(runner, audit=audit, policy=mesh.authorize)

    ci = None
    if svc.aws("code_interpreter"):
        ci = aws.CodeInterpreterStats(svc.region, os.environ.get("ATR_CODE_INTERPRETER_ID"))
        stats.set_executor(ci)
    else:
        stats.set_executor(None)

    if svc.aws("model"):
        model_for = lambda role: aws.bedrock_model(svc.region, role, cp)          # noqa: E731
    else:
        scripts = EXPLORE_SCRIPTS if task == "explore" else LOCAL_SCRIPTS
        model_for = lambda role: guarded(pb.PlaybookModel(scripts[role], role), role, cp, "playbook")  # noqa: E731

    from .verdicts import priors_from
    researcher = Researcher(gw, runner, inventory, store, emit=emit, audit=audit, priors=priors_from(store))
    if svc.aws("model"):   # a vision-capable model reads the heatmap images; the offline playbook cannot
        researcher.vision = aws.HeatmapReader(lambda: aws.bedrock_model(svc.region, "scientist", cp))
    if svc.aws("nlm"):
        researcher.nlm = aws.NetworkLM(region=svc.region)
    if explore_objective == "handover":
        from .explore_rule import RuleExplorer
        explorer = RuleExplorer(gw, inventory, store, audit=audit, priors=researcher.scientist.priors, emit=emit,
                                baseline_rule=baseline_rule)
    else:
        from .explore import Explorer
        explorer = Explorer(gw, inventory, store, audit=audit, priors=researcher.scientist.priors, emit=emit)
    return Runtime(svc, researcher, mesh, model_for, store, scheduler, world, ci, explorer)
