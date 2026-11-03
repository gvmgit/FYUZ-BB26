# Live test and 2-minute talk: runbook for judging

What the judges do (FYUZ judging guide): they name an item and a time window, or pick a run from `judging.json` or
one they saw in our account. Our read-only tool shows the record and prints where it lives; the chair opens it with
read-only access and compares. Two minutes, plus 30 seconds to read. After the freeze the account is read-only, so the
tool uses read calls only: S3 GetObject/ListObjects, DynamoDB Query, CloudWatch Logs Insights and FilterLogEvents.

## Before the slot

1. Sign in (SSO device code, no password typed into a terminal), then load the session:
   `aws sso login --profile atr-sso --use-device-code`, then the environment as in `deploy/aws_env.ps1`.
2. `cd` to the repository root. Each call below takes 7 to 12 seconds.

## The three ways the judges can ask

| They say | Command (from the repository root) |
|---|---|
| a run from judging.json and an item | `python3 researcher/tools/show_record.py --run-id <run_id> --item <n>` |
| only a time window (and maybe an item) | `python3 researcher/tools/show_record.py --window 2026-11-02T09:38:00Z,2026-11-02T10:10:00Z [--item <n>]` |
| the run they saw in the video | `python3 researcher/tools/show_record.py --run-id 2026-11-02T15-04-57-demo-live --item <n>` |

With only a window the tool lists every run active in it; with exactly one, it shows that run directly. Add `--window`
to any call to restrict the CloudWatch search; the tool says how many records fall inside and warns when none do.

## What to point at in the output

1. **Where it lives:** `s3://bb26-p-team09-data-651343917891-use2/evidence/declared|items|records/<run_id>...` with S3 LastModified.
   The declaration's `declared_at` is earlier than the items and the record: the target was set before the run.
2. **The item:** the request, the expected decision, and the decision record (principal, action, decision, policy id).
3. **DynamoDB:** table `TelecomResearcher-Audit…`, index `trace`, the trace id, the first and last records.
4. **CloudWatch:** the Logs Insights query string, printed so the chair can paste it; `@timestamp` and `@ingestionTime`.
5. **TIMES:** the first DynamoDB record against the first CloudWatch event.

## Cheat sheet (rehearsed 3 Nov against AWS)

| Run id | Unit | Items | DynamoDB / CloudWatch | Times |
|---|---|---|---|---|
| `2026-11-02T09-38-45-allow-e107` | allow | 0-3 | 4 / 4 | 2.4 s |
| `2026-11-02T09-39-03-deny-00dc` | deny | 0-5 (item 5 = Guardrails) | 6 / 6 | 3.1 s |
| `2026-11-02T09-39-46-pause-5845` | pause | 0-2 | 4 / 7 | 3.1 s |
| `2026-11-02T09-40-40-diagnosis-9194` | blind diagnosis | 0-5 (one per fault) | 279 / 279 for item 0 | 0.0 s |
| `2026-11-02T10-06-19-trace-3768` | rebuilt trace | 0-278 | 279 / 279 | 0.0 s |
| `2026-11-02T10-05-12-trace-a6a5` | trace (rejected, kept) | 0-276 | 279 / 279 | 0.0 s |
| `2026-11-03T09-08-10-nlm_tuning-ceac` | NLM tuning (base vs tuned) | 0-34 (one per frozen test item) | S3 items + SageMaker job (no trace) | n/a |
| `2026-11-02T15-04-57-demo-live` | demo run in the video | 0-314 (decision records) | 315 / 315 | 0.0 s |

What to say if asked:

- **The allow run's CloudWatch events come from FilterLogEvents.** They are in the `/atr/audit` stream of that trace,
  but they are stamped a few seconds before CloudWatch dates the log group's creation (it was the first run after
  deploy), and Logs Insights does not search those. The tool falls back and says so.
- **Allow, deny and pause differ by 2 to 3 seconds.** The checkpoint writes the DynamoDB item first, then creates the
  trace's log stream and writes the CloudWatch event. Agent runs, which log through AgentCore, agree to 0.0 s.
- **Pause has 4 DynamoDB records and 7 CloudWatch events.** The CloudWatch search matches the run id as text, so it also
  finds the paused investigation's own trace (`<run_id>-agents`).
- **The demo run is not a reported unit.** It has no declaration, by design: its evidence is its trace, ending with the
  engineer's ApproveChange (Human ALLOW). Reported results come only from runs with a declaration.

## 2-minute talk track

| Time | Slide | Say | Criterion |
|---|---|---|---|
| 0:00-0:15 | 2 | Operators won't let AI agents near the network: no policy, no audit, no approval. The barrier is trust. | Relevance 20% |
| 0:15-0:35 | 5, 6 | A swarm that must prove itself: four Strands agents, a critic with no evidence of its own, claims sealed first. | Innovation 25% |
| 0:35-0:50 | 8, 9 | Built on the Breaking Barriers architecture; every call crosses one checkpoint outside the agents, in our account. | Production 10% |
| 0:50-1:15 | 10, 11 | Live run: cell 3 found after 95 experiments, Human ALLOW. Allow, deny, pause and trace: 4 of 4 pass. | Demonstration 20% |
| 1:15-1:35 | 12, 13 | Six sealed faults, six found, targets declared first, graded by code; every number has a record. | Evidence 15% |
| 1:35-1:50 | 14 | What did not work is in GAPS.md, failed runs kept. | Openness 5% |
| 1:50-2:00 | 15 | Minutes instead of weeks, and the engineer still decides. Then the live test. | Presentation 5% |
