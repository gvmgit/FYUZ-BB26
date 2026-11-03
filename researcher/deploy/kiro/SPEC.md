# Infrastructure spec for headless Kiro

The hackathon blueprint lists Headless Kiro for "infrastructure-as-code generation" and "headless execution mode for
CI/CD pipeline integration". This file is the spec Kiro reads to generate and maintain the CDK app in `deploy/cdk/`.
`atr.aws_adapters.kiro_generate` runs it (command from `ATR_KIRO_CMD`; verify it against your Kiro installation).

## Goal
Deploy the Telecom Researcher: Strands agents on AgentCore Runtime, `run_experiment` behind AgentCore Gateway, findings
in DynamoDB, nightly scheduled investigations.

## Resources (generate with AWS CDK, Python)
- S3 bucket `Field`: private, holds `field/current.csv` and `heatmaps/`.
- DynamoDB tables `Findings`, `Audit`, `Truth` (partition key `id`, on-demand billing).
- Lambda container function `RunExperiment` built from `researcher/deploy/lambda_fn/Dockerfile`, 3 GB, 120 s, with
  read/write on the bucket. It is the Gateway's Lambda target.
- IAM role for AgentCore Gateway allowed to invoke only that Lambda.
- SQS queue `Jobs` fed nightly (02:00 UTC) by an EventBridge rule: the scheduler.
- IAM role for the agents: Bedrock invoke, Code Interpreter, Gateway invoke, read/write `Findings`, write `Audit`,
  consume `Jobs`.

## Steering rules (must hold in every generated change)
1. The agents' role must have an explicit deny on the `Truth` table. The planted fault of a blind test must be
   unreadable by the agents.
2. No wildcard resource on anything that writes data.
3. Nothing in the stack may write to a live network. Findings are proposals that wait for a person.
4. Every table and bucket is removable (`DESTROY`) because this is a hackathon environment.

## CI/CD (headless)
1. `cdk synth` must succeed.
2. `python scripts/check_adapters.py`, `scripts/check_agents.py` and `scripts/check_deploy.py` must pass.
3. `cdk diff`, then `cdk deploy` on approval.
