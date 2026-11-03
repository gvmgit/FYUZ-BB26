# Running locally and on AWS (one toggle)

The same code runs both ways. `ATR_MODE=local|aws` (or `--mode`) flips every AWS-backed service at once, and
`ATR_<SERVICE>=local|aws` flips one, so you can bring AWS up piece by piece. `python -m atr.cli status` shows what is
selected and which environment variables are still missing.

| Service (hackathon blueprint) | Local | AWS | Variables for AWS |
|---|---|---|---|
| Strands agents' model | offline playbook model (real Strands loop, scripted tool choice) | Claude on Amazon Bedrock | `ATR_MODEL_ID` |
| AgentCore Gateway (`run_experiment`) | in-process gateway | MCP endpoint -> Lambda running the twin | `ATR_GATEWAY_URL` (`ATR_GATEWAY_TOKEN` or SigV4) |
| AgentCore Code Interpreter (Critic's statistics) | local numpy | same function source executed in the sandbox | optional `ATR_CODE_INTERPRETER_ID` |
| Finding cards | SQLite | DynamoDB | `ATR_DDB_TABLE` |
| Audit trail | JSON lines | DynamoDB | `ATR_AUDIT_TABLE` |
| AgentCore Scheduler | in-process priority queue | SQS fed nightly by EventBridge | `ATR_SQS_URL` |
| Network language model (OTel-LLM-8.3B) | off | SageMaker endpoint, second opinion on suspects | `ATR_NLM_ENDPOINT` |
| Headless Kiro (IaC) | off | `kiro_generate` runs your Kiro command on `deploy/kiro/SPEC.md` | `ATR_KIRO_CMD` |
| AgentCore Runtime | `python -m atr.cli ...` | `deploy/agentcore_app.py` | - |
| Lambda | - | `deploy/lambda_fn` (container image with the MAVERIC twin) | - |
| Agent mesh / policy (ANF stand-in) | in-process | same, plus IAM in `deploy/cdk/stack.py` | - |
| KPI and alarm streams | JSONL stream files | Kinesis Data Streams `atr-kpi`, `atr-alarms` | `ATR_KINESIS_PREFIX` |
| Interface-log lake + parsers | folder + `ingest.parse_line` | S3 lake; parsers run in the protocol/lab Lambdas | `ATR_LAKE_BUCKET` |
| Feature tables | SQLite | Glue tables queried with Athena | `ATR_GLUE_DATABASE`, `ATR_ATHENA_OUTPUT` |
| Log text + vectors | TF-IDF vectors | OpenSearch Serverless (Titan embeddings) | `ATR_OPENSEARCH_ENDPOINT` |
| Finding graph | networkx JSON | Amazon Neptune serverless (openCypher, IAM auth) | `ATR_NEPTUNE_ENDPOINT` |
| Experiment protocol | local ASL interpreter, same definition | Step Functions + task Lambda; EventBridge Scheduler nightly | `ATR_STATE_MACHINE_ARN` |
| Lab executor (test automation API slot) | in-process `lab_execute` tool | Lambda target on AgentCore Gateway | `ATR_LAB_GATEWAY_URL` |
| Review console | `atr.cli serve` -> `/console/` | Amplify Hosting + Cognito + API Gateway/Lambda; Managed Grafana over Athena | stack outputs |
| Research-tuned NLM | LoRA on a local base model (`train-nlm`) | SageMaker training; Bedrock Custom Model Import to serve | `ATR_MODEL_IMPORT_ROLE`, `ATR_MODEL_BUCKET` |

### Research pipeline on AWS (after the steps below)
1. `python app.py` in `deploy/cdk` synthesises the full stack (verified); `npx aws-cdk@latest deploy` creates it (not run).
2. Add the `lab_execute` target to the gateway with the `LabExecutorLambdaArn` output (same call as run_experiment in
   `deploy/provision_gateway.py`).
3. Publish `console/` to the Amplify app (zip deploy) with `config.json` set to the `ConsoleApiUrl`, `CognitoDomain` and
   `CognitoClientId` outputs; add reviewers to the Cognito pool.
4. Plant the lab's secret fault (`plant-fault`), then `python -m atr.cli --mode aws research` starts a Step Functions
   execution of the same protocol; the nightly schedule starts one every night.
5. After an engineer accepts findings, run the training job on SageMaker, `nlm.export_merged`, copy the folder to the
   model bucket and `BedrockModelImport.start_import`.

All of these AWS calls are written and checked against fakes (`scripts/check_research_pipeline.py`); none has run in an
AWS account.

## What is verified, and what is not

| Piece | Verified here | Needs your AWS account |
|---|---|---|
| Strands agents, orchestration, guards, policy | Yes: `scripts/check_agents.py` | - |
| Toggle and configuration errors | Yes: `scripts/check_adapters.py` | - |
| DynamoDB/SQS/Code Interpreter/MCP/SageMaker/S3 adapter logic (serialisation, parsing, policy, audit) | Yes, against in-memory fakes | The calls themselves |
| Code sent to the Code Interpreter | Yes: run as a standalone program, result equals the local one | That the sandbox has numpy |
| Lambda handler | Yes: real twin, JSON-safe result, safety bounds | Cold start, memory/time limits |
| Gateway `create_gateway` / `create_gateway_target` requests | Yes: valid against the installed botocore service model | Creating them |
| SigV4 signing of MCP requests | Signer produces a signed request | That the gateway accepts it |
| Bedrock model call, heatmap reading by Claude | Image block plumbing only | The calls, and a Bedrock model id with access |
| CDK stack | Yes: `cdk synth` succeeds (aws-cdk-lib 2.275, CLI via npx) and the template was inspected: 3 tables, 3 GB/120 s container Lambda, nightly rule, deny on the Truth table | `docker build` of the image (Docker not installed here) and `cdk deploy` |
| Kiro | - | Command and flags (nothing about Kiro's CLI is assumed in code) |
| Network language model | Response parsing only | Endpoint and its request/response schema |

## Bring-up on AWS (in order)
1. `python -m atr.cli build-twin --retrain` (produces `models/twin_model_map.pickle`, baked into the Lambda image).
2. `cd deploy/cdk && pip install -r requirements.txt && npx aws-cdk@latest synth && npx aws-cdk@latest deploy`
   (needs Node, Docker for the image, and AWS credentials; synth is already known to pass).
3. With the stack outputs: `set ATR_GATEWAY_ROLE_ARN=...; set ATR_LAMBDA_ARN=...; python deploy/provision_gateway.py`
   and note the `ATR_GATEWAY_URL` it prints.
4. Set `ATR_MODE=aws`, `ATR_MODEL_ID` (a Bedrock model you have access to), `ATR_GATEWAY_URL`, `ATR_DDB_TABLE`,
   `ATR_AUDIT_TABLE`, `ATR_SQS_URL`, `ATR_FIELD_BUCKET`, `ATR_TRUTH_TABLE`. Run `python -m atr.cli status`.
5. The teammate plants a fault: `python -m atr.cli --mode aws plant-fault --fault tilt:3:16` (field log to S3, truth to
   a table the agents' role cannot read).
6. `python -m atr.cli --mode aws blind-test` (agents on Bedrock, experiments through the Gateway).
7. `python -m atr.cli --mode aws reveal` shows the planted fault.
8. For an always-on agent: build `deploy/agentcore/Dockerfile` and run `agentcore launch` (or create the runtime from the
   image); `deploy/agentcore_app.py` is the entrypoint. SQS jobs queued by `sweep`/EventBridge are consumed there.

## Not done / limits
- The local field is the twin plus noise; on AWS, point the field at your teammate's live MAVERIC run for an independent
  test. Detection against an independent simulator is untested.
- The offline model scripts the tool order; with Claude the model chooses, so runs vary. The publish guard, mesh policy
  and Critic rules are code and hold either way.
- Federated learning, the E2E 5G generator and the Transport/Core/Service-Assurance domain agents are not built. Language-model
  fine-tuning is built up to the training run (see `docs/FINE_TUNING.md`); twin fine-tuning runs locally.
