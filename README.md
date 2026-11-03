Primary track: Track 3 - Multi-Agent Swarms for Application Enablement through NLMs (team09, NetworkIQ)

# Autonomous Telecom Researcher

**Problem.** Finding why one area of a radio network under-performs takes an engineer weeks of trial and error.
Operators will not let AI agents help without enforced policy, an audit trail and a person who approves every change.
We built a swarm of agents that proves the cause on a digital twin, with every call crossing one policy checkpoint.

**What it does.** Four Strands agents on Amazon Bedrock AgentCore investigate a coverage complaint: they write testable
claims, test them on the MAVERIC RF digital twin, try to break their own answer, and hand the engineer one finding with
its evidence. A network language model (OTel-LLM-1.7B-IT, also LoRA-tuned by us) gives the scientist a second opinion.
No agent can approve or apply a change; only a person can.

## Results

All units pass (`python researcher/tools/check_units.py`). Run ids, live-test and rebuild commands are in `judging.json`.

| Unit | What it shows | Result (target, declared before the run) | Run id |
|---|---|---|---|
| Live allow | permitted calls go through, each with its decision record | 4/4 (1.0) | `2026-11-02T09-38-45-allow-e107` |
| Live deny | 5 calls refused by Cedar policy, 1 prompt attack blocked by Bedrock Guardrails | 6/6 (1.0) | `2026-11-02T09-39-03-deny-00dc` |
| Pause | model withdrawn in the registry: calls refused, nothing published, resumed after | 1.0 (1.0) | `2026-11-02T09-39-46-pause-5845` |
| Rebuilt trace | one trace id from the first request to the human approval (279 records) | 1.0 (1.0) | `2026-11-02T10-06-19-trace-3768` |
| Blind diagnosis | six sealed faults on the frozen set `diagnosis_v1`, found by the swarm | 6/6 located (0.8) | `2026-11-02T09-40-40-diagnosis-9194` |
| NLM tuning | base vs LoRA-tuned OTel model on the frozen set `nlm_test_v1` | 0/35 to 11/35, gain 0.31 (0.10) | `2026-11-03T09-08-10-nlm_tuning-ceac` |

Demo video: `researcher/demo_video/NetworkIQ_Demo.mp4` (live AWS run `2026-11-02T15-04-57-demo-live`).
Presentation: `researcher/docs/NetworkIQ_Presentation.pptx`. What did not work: `GAPS.md`. Data sources: `DATA.md`.

## Architecture

```
            request (run id = trace id)                    Amazon Bedrock AgentCore Runtime (ARM64 container)
 engineer ─────────────────────────────▶  Orchestrator ──ask_scientist / ask_experimenter / ask_critic──▶ specialists
                                              │                Strands agents, Claude Sonnet 5 on Bedrock;
                                              │                OTel-LLM-1.7B-IT on SageMaker (network language model)
                                              │                + Amazon Bedrock Guardrails on model input and output
          every model call, tool call and     ▼
          agent-to-agent request  ───▶  CHECKPOINT (researcher/atr/checkpoint.py)
                                         ├─ policy decision: Amazon Verified Permissions, researcher/policies/atr.cedar
                                         ├─ registry (DynamoDB): models/tools approved or withdrawn (the pause)
                                         └─ decision record → DynamoDB audit (trace index) + CloudWatch Logs
                                              │ ALLOW
                                              ▼
                     AgentCore Gateway (MCP, SigV4) ──▶ Lambda: MAVERIC RF digital twin, emulated lab
                                              │
          finding cards (DynamoDB) ──▶ review console (Amplify + Cognito + API Gateway)
          ApproveChange: only a Human principal is permitted; every agent is forbidden (a person approves any change)
```

Also deployed: Step Functions experiment protocol, Kinesis, S3 log lake, Glue + Athena features, OpenSearch Serverless
log search, Neptune finding graph, SQS + EventBridge scheduling, CloudWatch dashboard. Every IAM role is under `/bb26/`
with the `bb26-Boundary` permissions boundary. Region: us-east-2. Detail: `researcher/docs/AWS_MIGRATION.md`,
`researcher/docs/RESEARCH_PIPELINE.md`, `researcher/docs/PROPOSAL_COVERAGE.md`.

## How the agents work

| Agent | Job | Tools the Cedar policies permit | What it cannot do |
|---|---|---|---|
| Orchestrator | runs the investigation | `ask_scientist`, `ask_experimenter`, `ask_critic`, `publish_findings` | run experiments |
| Scientist | reads the field log, asks the OTel network language model for a second opinion, writes testable claims (sealed with a hash before testing) | `observe_symptom`, `propose_claims` | publish, review |
| Experimenter | tests one setting at a time on the twin, through AgentCore Gateway | `test_claim`, `run_control`, `run_null`, `run_holdout`, `run_replication` | review its own results |
| Critic | tries to break every claim: control, twin-bias null test, held-out seeds, second instrument | `review_claim` | run experiments (no evidence of its own) |

Flow: request → orchestrator → scientist (claims) → for each claim: experimenter tests, critic reviews and asks for the
checks still missing → publish only when every claim has a final verdict → a person approves or rejects. Every arrow is
a checkpoint decision with a record under the run's trace id.

## How to run it: step by step

Commands run from the repository root unless a step says `cd researcher`. Windows PowerShell and bash forms are given
where they differ.

### Step 1. Prerequisites

- Python 3.10 (MAVERIC's pinned packages have no wheels for newer versions) and AWS CLI v2.
- Access to the team AWS account (651343917891) in us-east-2.
- `cdk bootstrap` run once in the account (the deploy pipeline uses the bootstrap roles).
- The OTel model copied once from the event bucket to the team bucket:
  `aws s3 sync s3://fyuz-otel-leeds-2026/base-model/otel-llm-1.7b-it/ s3://bb26-p-team09-data-651343917891-use2/models/otel-llm-1.7b-it/ --copy-props none`

### Step 2. Install

```bash
python -m venv .venv
.venv\Scripts\activate                 # bash: source .venv/bin/activate
pip install -r researcher/requirements.txt
```

### Step 3. Sign in to AWS and load the session

```bash
aws login --profile atr                # or: aws sso login --profile <your SSO profile>
cd researcher
. .\deploy\aws_env.ps1                 # bash: eval "$(aws configure export-credentials --profile atr --format env)"; export AWS_REGION=us-east-2
```

### Step 4. Deploy the stack (once)

```bash
python deploy/make_package.py          # packages the solution for CodeBuild
python deploy/codebuild/launch.py      # CodeBuild: CDK stack, AgentCore Gateway and Runtime, Lambda images, console
python deploy/stack_outputs.py         # writes deploy/cdk/outputs.json from the CloudFormation stack
python deploy/nlm_endpoint.py create   # optional: the OTel network language model on SageMaker (status | test | delete)
```

The pipeline's `deploy/post_deploy.py` writes `deploy/cdk/atr_aws.env` (the deployed resource names). The endpoint
addresses are not kept in the repository; these two commands regenerate them.

### Step 5. Run the units

Each command writes its declaration (target, test-set hash, model, grader) to `researcher/evidence/declared/` and to
S3 before the run starts, then the per-item outputs and the grader's record.

```bash
python tools/track3_units.py diagnosis                                    # blind test: six sealed faults through the deployed agents
python tools/track3_units.py allow                                        # live allow
python tools/track3_units.py deny                                         # live deny (Cedar policy + Bedrock Guardrails)
python tools/track3_units.py pause                                        # withdraw the model: calls refused, then resume
python tools/track3_units.py trace --from <diagnosis run id> --approved-by reviewer-team09   # human approval + rebuilt trace
python tools/nlm_tuning.py start                                          # freeze the NLM test set, declare, start the SageMaker job
python tools/nlm_tuning.py finish --run-id <nlm run id> --wait            # grade base vs tuned, write items and record
```

### Step 6. Publish the results

```bash
python tools/finalize.py               # evidence/results.json and evidence/MANIFEST.sha256 (copies in the bucket)
python tools/write_judging.py          # judging.json at the repository root (copy in the bucket under judging/)
```

### Step 7. Check and rebuild (offline, no AWS needed)

```bash
python researcher/tools/check_units.py     # pass/fail per unit: declared first, one run id, produced by the pipeline
python researcher/tools/rebuild.py         # reprints every result from researcher/evidence/, exits 1 if any data changed
```

### Step 8. Live test (read-only)

Works with read calls only (S3, DynamoDB, CloudWatch Logs), so it keeps working when the account is read-only.
Runbook with every run and what to point at: `researcher/docs/LIVE_TEST.md`.

```bash
python researcher/tools/show_record.py --run-id <run id> --item <n>          # one item of one run, where it lives, AWS times
python researcher/tools/show_record.py --window <start>,<end> [--item <n>]   # find the runs active in a time window
```

### Step 9. See it working (optional)

```bash
.\researcher\deploy\run_console.ps1    # review console http://localhost:8765/console/ : finding cards, accept or reject
.\researcher\deploy\run_dashboard.ps1  # live dashboard http://localhost:8770 : an investigation as it runs
```

### Offline mode

The same agents run on the local twin with the same Cedar policies evaluated locally, with no AWS account:

```bash
cd researcher
python -m atr.cli blind-test --random-seed 7     # one blind investigation on the local twin
python scripts/check_all.py                       # every offline check (the CDK check needs `python app.py` in deploy/cdk first)
```

### Clean up

```bash
python researcher/deploy/nlm_endpoint.py delete  # stop the SageMaker endpoint (the model stays in S3)
```

## Evidence chain

- `researcher/evidence/frozen/` - the frozen test sets (`diagnosis_v1.json`, `nlm_test_v1.jsonl`); each sha256 is in the
  declaration of every run that uses it.
- `researcher/evidence/declared/<run_id>.json` - written to the repository and to
  `s3://bb26-p-team09-data-651343917891-use2/evidence/` before the run starts: unit, target, test-set hash, model, grader.
- `researcher/evidence/items/<run_id>.jsonl` (per-item outputs), `.../records/<run_id>.json` (grader verdict),
  `.../results.json` (reported results), `.../MANIFEST.sha256`. All mirrored to the bucket.
- Grader: `researcher/atr/grader.py`, rule-based; it compares with the planted truth or the expected decision. No model
  grades. The answering model (Claude) cannot read the truth table (explicit IAM deny).
- The tuned adapter is in S3 at `model_artifact_s3_prefix` in `judging.json`.

## Before Nov 1

Built before the event: the MAVERIC digital-twin wrapper and blind-test world (`twin.py`, `world.py`, `experiment.py`),
the scientific logic of the agents (`agents.py`, `researcher.py`, `prereg.py`, `stats.py`), the emulated lab and research
pipeline modules (`lab.py`, `ingest.py`, `observer.py`, `replicate.py`, `protocol.py`, `graph.py`, `artefacts.py`,
`nlm.py`), the language-model training data, the offline playbook model, the review console, and their checks.

Built during the event: the AWS deployment (CDK stack, AgentCore Gateway and Runtime, Lambda images, CodeBuild
pipeline), the Strands agent wiring used in AgentCore, the checkpoint (Cedar policies on Verified Permissions, registry
and pause, decision records with trace ids), Bedrock Guardrails, the evidence chain (`atr/evidence.py`, `atr/grader.py`,
`tools/`), the LoRA fine-tune of the OTel network language model (`finetune/train_eval_lora.py`, `tools/nlm_tuning.py`),
and all runs listed in `judging.json`.

## Repository layout

| Path | What it is |
|---|---|
| `researcher/atr/` | the solution: `strands_agents.py` (agents), `checkpoint.py` (policy checkpoint, registry, decision records), `agents.py` (scientific logic), `anf.py` (agent mesh, human in the loop), `evidence.py` + `grader.py` (evidence chain) |
| `researcher/policies/atr.cedar` | the Cedar policies |
| `researcher/deploy/` | CDK stack, AgentCore Gateway and Runtime, Lambda images, CodeBuild pipeline, NLM endpoint, console and dashboard launchers |
| `researcher/tools/` | units, NLM tuning, live test, finalize, rebuild, check |
| `researcher/finetune/` | language-model training data and training scripts |
| `researcher/evidence/` | frozen test sets, declarations, items, records, results, manifest |
| `researcher/scripts/` | offline checks |
| `researcher/console/` | review console |
| `maveric-1.0.0 1/` | MAVERIC RF digital twin library (MIT licence) |
| `data/` | simulation data the twin is trained on |
| `README.md`, `judging.json`, `DATA.md`, `GAPS.md` | hand-in files |
