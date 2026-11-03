# GAPS

What did not work, or is weaker than it looks, with the row it limits. Run ids refer to `researcher/evidence/` and `judging.json`.

## Platform and environment blocks

| When (UTC) | Service | What happened | Exact error / request id | Who we told | Limits |
|---|---|---|---|---|---|
| 2026-11-02 07:55:10 | Amazon Bedrock (ConverseStream), us-east-2, model `us.anthropic.claude-sonnet-5-5` | First agent run in us-east-2 failed: the account has no Marketplace subscription for Sonnet 5.5 and the agents' role may not subscribe. Switched the runtime to `us.anthropic.claude-sonnet-5` (already subscribed); the deploy now prefers it. | `AccessDeniedException ... not authorized to perform the required AWS Marketplace actions (aws-marketplace:ViewSubscriptions, aws-marketplace:Subscribe)`; AgentCore request id `e74c7bfc-1184-461d-9721-4ebc0aa9dab8` | not reported to the organisers (worked around) | none after the switch |
| 2026-11-01..02 | Hugging Face Hub (from the team laptop network) | Downloading an open base model was refused, so the LoRA fine-tune starts from the event's OTel-LLM-1.7B-IT in our team bucket instead (run `2026-11-03T09-08-10-nlm_tuning-ceac`, SageMaker training job, `tools/nlm_tuning.py`). | HTTP 403 from huggingface.co (corporate proxy) | not reported to the organisers | none for the result: base and tuned are compared on the frozen set `nlm_test_v1` (see limit 11). |
| 2026-11-02 | AWS CDK bootstrap | The CDK bootstrap roles (`cdk-hnb659fds-*`) are not under `/bb26/` (the guide's known limit). Every role in our stack and the CodeBuild role are under `/bb26/` with `bb26-Boundary`. | n/a (bootstrap succeeded from the participant session) | not reported | if the role guardrail is switched on, a redeploy through the bootstrap roles may be refused |

## Design limits we know about

1. **Where the policy is enforced.** The decision is made outside the agents (Amazon Verified Permissions, policies in
   `policies/atr.cedar`, the agents' IAM role cannot change policies or the registry). The enforcement point is the
   checkpoint inside the agent process, which calls Verified Permissions before every model, tool and agent-to-agent
   call. The tool Lambdas behind AgentCore Gateway do not re-check the Cedar decision; they are reachable only through
   the gateway (IAM). A modified agent process that skips the checkpoint could still call the gateway. AgentCore
   Gateway's own policy feature is not used. Limits: Technical innovation, Architecture.
2. **Guardrail scope.** Bedrock Guardrails filter harmful content and prompt attacks on the latest model input and on the
   output. No denied topics are configured: a "do not change the network" topic would likely also block the agents'
   legitimate twin experiments (we did not test one). Change of state is prevented by policy
   (`agents-never-change-state`) instead.
3. **Small blind test set.** `diagnosis_v1` has 6 planted faults (one per kind and cell), because each investigation
   takes about 3 to 4 minutes and about 0.7 M tokens. Results on 6 items are a demonstration, not a statistic.
4. **The twin is a model.** The field logs are simulated from a Gaussian-process twin trained on MAVERIC simulation data,
   with a planted fault. A finding on the twin is not a finding on a real network.
5. **One investigation at a time.** The planted field log is one S3 object, so blind-test items run sequentially.
6. **Managed Grafana not deployed.** It needs IAM Identity Center in the account, which the team account does not have;
   the CloudWatch dashboard `TelecomResearcher` replaces it.
7. **Fixed during the event.** A warm run_experiment Lambda kept the first field log it had loaded; it now reloads when
   the S3 object changes. Runs before 2026-11-02 14:25 UTC used one planted fault per deployment, so they were unaffected,
   but they are not part of the reported evidence.
8. **Demo recording.** `researcher/demo_video/NetworkIQ_Demo.mp4` (also in the submission bucket under `video/`) was recorded on this
   build. Its live part is one demo run, `2026-11-03T12-04-04-live-demo`, started from AWS CloudShell and shown in CloudShell, the
   AWS console and our dashboard; it is not one of the reported runs. Earlier demo runs (`2026-11-02T15-04-57-demo-live`,
   `2026-11-03T10-23-57-cloudshell-live`, `2026-11-03T11-39-40-live-demo`) are in the account with their records; one recording
   attempt of the dashboard failed (a stale local server) and was redone with a new run.
9. **Data provenance to confirm.** The 3-cell simulation data came with the MAVERIC material we started from; the
    exact public download location is not recorded in `DATA.md` yet.
10. **Logs Insights misses the allow run.** The allow unit (`2026-11-02T09-38-45-allow-e107`) was the first run after
    deploy: its 4 events in the `/atr/audit` stream are stamped a few seconds before CloudWatch dates the log group's
    creation, and Logs Insights does not search those. The events are there (FilterLogEvents reads them); `tools/show_record.py`
    falls back to that read and says so. For allow, deny and pause the DynamoDB record and the CloudWatch event are 2 to 3 s apart
    (DynamoDB is written first, then the log stream is created); agent runs agree to 0.0 s.
11. **The tuned network language model: what it does and does not do.** On the 35 frozen `suspect_ranking` items the base
    OTel model answers 0 correctly and the LoRA-tuned model 11 (gain 0.31, target 0.10). The base model ignores the answer
    schema (free-form JSON with its own fields), and the 48-token answer limit, the same for both models, cuts its longer
    replies. The tuned model always answers in the schema and names the right cell in 29 of 35 items, but the right fault kind
    mainly for azimuth faults (10/10); tilt 1/10, cell off 0/5, no fault 0/5, twin miscalibration 0/5, because it leans to
    'azimuth'. One training run, 35 scored items. The agents keep calling the served base model as an advisory second
    opinion; the adapter is evaluated on the frozen set and stored in S3, not deployed to the agents.

## Failed runs

Every run with a record stays in `evidence/` whether it passed or failed; `judging.json` lists them. Runs that failed
before producing a record are listed here with their run id and error.

| Run id | What happened | Effect on the evidence |
|---|---|---|
| `2026-11-02T09-40-40-diagnosis-9194` | All 6 investigations finished (6/6 located, logged in DynamoDB and CloudWatch under trace ids `...-i0` to `...-i5`), then the S3 upload of the items file failed: `ExpiredToken` on PutObject at about 10:04 UTC (the laptop's 1-hour credentials ran out during the 25-minute run). The items file had already been written locally; it was uploaded and graded 1 minute later from that file, and the record carries a note saying so. The tool now renews credentials before each upload. | The items file in S3 is dated after the run (10:04:57 UTC) rather than at the end of item 5; the per-item agent records in DynamoDB/CloudWatch carry the original times. |
| `2026-11-02T10-05-12-trace-a6a5` | Passed its grader, but its items carried the traced run's id instead of its own, so `tools/check_units.py` rejects it (`one_run_id`). Fixed in `atr/evidence.py` (the traced run's id is kept as `source_run_id`) and re-run as `2026-11-02T10-06-19-trace-3768`. The re-run adds a second human approval to the same trace. | Not reported; kept with its record. |
| first us-east-2 agent run (no run id; before the evidence chain) | `RuntimeClientError` 500: the Sonnet 5.5 Marketplace denial above. | Not part of the evidence. |
