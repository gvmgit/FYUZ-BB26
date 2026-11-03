# The full research pipeline (Proposal 4, end to end)

`python -m atr.cli research --fault cell_off:2` runs one autonomous research cycle as the experiment protocol
(`atr/protocol.py`, an Amazon States Language document). Locally a small interpreter runs it; on AWS the same document is a
Step Functions state machine started nightly by EventBridge Scheduler.

```
RunLabCampaign -> Ingest -> Observe -> Investigate -> CrossCheckModalities -> ClaimSurvived?
  -> PreRegisterReplication -> ReplicateAcrossProfiles (Map: busy_hour, commuter) -> Analyse -> Falsify
  -> Publish -> UpdateFindingGraph -> DeriveArtefacts -> QueueRetraining -> Done
  (no surviving claim: PublishWithoutReplication -> UpdateFindingGraph -> ...)
```

## What each step does, and what stands in for what

| Step | What happens | Stand-in (local) | AWS |
|---|---|---|---|
| Lab | An overnight campaign: one-minute windows of moving UEs under a traffic profile, with the lab's secret injected fault | `lab.EmulatedLab` on MAVERIC physics (twin signal, track generator, attachment rule); stands in for VIAVI + the E2E 5G generator + the test automation API | `lab_execute` Lambda on AgentCore Gateway |
| Ingest | KPI windows and alarms on streams; decoded O-RAN interface logs (RRC, F1AP, E1AP, XnAP, NGAP) to the lake, parsed to events; text indexed | JSONL streams, folder lake, `ingest.parse_line`, SQLite, TF-IDF | Kinesis, S3, Glue + Athena, OpenSearch Serverless |
| Observe | Per-cell Welch z against a healthy baseline, per modality; isolation-forest anomaly model; a cell flagged by 2+ modalities is a cross-modal pattern | `observer.Observer` (ANSI SQL works on both engines) | same code, Athena |
| Investigate | The Strands team on the drive-test survey: pre-registered claims, one-setting sweeps, control, twin-bias null, fresh seeds, second instrument (tilt), head-to-head | offline scripted model | Claude on Bedrock, AgentCore Gateway/Code Interpreter |
| Cross-check | The survey's per-cell suspicion becomes a fourth modality; does the claim sit on a cross-modal pattern? | | |
| Replicate | Inject the claim and each rival into a clean lab on fresh tracks under two traffic profiles (load and mobility intervention); cosine similarity to the observed signature | `replicate.LabReplicator` | protocol task Lambda |
| Analyse | The Analyst agent (may compute, nothing else) computes paired effect sizes with 95% intervals | numpy | AgentCore Code Interpreter |
| Falsify | Pre-registered rule: the claim reproduces the signature in every profile and every rival is separated in the lab or was already excluded by the RF investigation | | |
| Publish | The card gains the observation, modalities, lab replication, protocol history; escalation recomputed (nothing reaches the network without a person) | SQLite | DynamoDB |
| Graph | Finding, cell, fault class, pattern, tests, rivals ruled out, artefacts | networkx JSON | Neptune (openCypher) |
| Artefacts | Early-warning indicator (calibrated on one set of fresh lab windows, validated on another, gated), test strategy for the fault class (from the graph), fine-tuning sample, detector rule (on acceptance) | | |
| Retrain | Queue a language-model retraining job when enough new findings have accumulated | local queue | SQS -> SageMaker; Bedrock Custom Model Import |

Engineers review in the console (`atr.cli serve`, then `/console/`): evidence, falsification attempts, cross-modal
observation, lab replication, artefacts, protocol timeline; accept, reject or request another experiment.

## Measured (local, simulated lab)

| Case | Cross-modal pattern | Finding | Lab replication | Early warning |
|---|---|---|---|---|
| cell 2 off | cell 2, 4 modalities (KPI, interface logs, alarms, survey) | confirmed, located | passed; every rival separated in the lab | `ue_share < 0.136`: deployable, 0/24 false alarms, 24/24 detected |
| cell 3 tilt 16 | cell 3, 3 modalities (KPI, interface logs, survey) | confirmed, located | passed; cell-off rival not separable from telemetry, excluded by the RF twin-bias test (reported) | rejected (7/24 detected) |
| cell 1 azimuth 225 | - | confirmed, located (check suite, classic engine) | ran | - |
| healthy lab | none | nothing published | skipped | - |

Early-warning indicator for a cell 4 tilt 10 -> 6 fault on its own: `ue_share > 0.376`, deployable, 0/24 false alarms,
24/24 detected. For a cell 5 azimuth 210 fault the telemetry signature is too weak for an indicator (rejected), and no
cross-modal pattern appears from telemetry alone; the survey still finds it.

A cycle takes about 40-50 s here: ~95 twin experiments plus ~32 lab campaigns.

## Honest limits
- The lab is emulated on the same MAVERIC physics as the twin. Lab replication is a different VIEW (mobility telemetry,
  interface events, fresh tracks, other traffic profiles), not independent physics. The simulator tables remain the
  independent second instrument, for tilt only.
- Interface logs are synthesised from the attachment timeline in a syslog text format; real O-RAN PCAP through HCLTech's
  parsers would replace `EmulatedLab._interface_logs` and `ingest.PARSERS`.
- Alarm thresholds, cell capacity (150 Mbps) and the gNB grouping (co-sited cells share a CU) are assumptions, documented
  in `atr/lab.py`.
- Language-model training: the code runs end to end except the base-model download, which this network blocks
  (huggingface.co returns 403). See `docs/FINE_TUNING.md`.
- No AWS call has run against a real account; all adapters are checked against fakes and `cdk synth` passes.
