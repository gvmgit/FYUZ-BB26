# Coverage of "Proposal 4: Autonomous Telecom Researcher"

Every bullet of the proposal now has an implementation. Three kinds of status:
- **Runs locally**: built and covered by `scripts/check_all.py` (10 suites, 309 checks, all passing).
- **AWS written**: the AWS implementation exists behind the toggle and is checked against fakes; the CDK stack synthesises;
  nothing has run in an AWS account.
- **Emulated**: a lab asset that is not available here (VIAVI, PCAP parsers, test automation API, E2E generator) is replaced
  by a stand-in on MAVERIC physics with the same interface, so the real asset can be plugged in.

## Expected outcomes
| Outcome | Status |
|---|---|
| Blind rediscovery of an injected causal pattern | **Runs locally.** Survey: 46/50 planted faults found, 0 wrong. Full research cycle: the lab's secret fault is found, seen as a cross-modal pattern, and replicated in the lab (`atr.cli research`) |
| At least one validated finding the engineers did not know | **Runs locally, twin-validated.** `atr.cli explore`: cell 4 azimuth 90 -> 110 cuts failure risk 1.8 points, survives the Critic; a candidate for a field trial |

## Proposed solution, bullet by bullet
| Proposal | Built | Status |
|---|---|---|
| **Observe, multimodally**: KPIs, handover statistics, alarms through Kinesis; syslog and PCAP from O-RAN interfaces through S3 and Lambda, parsed by F1/NGAP/E1 parsers; Glue + Athena feature tables; OpenSearch for log text and vectors; patterns across modalities | Per-cell KPI windows (load, handovers, ping-pongs, failures, SINR), alarms, decoded RRC/F1AP/E1AP/XnAP/NGAP logs, drive-test survey, heatmap images. Streams -> feature tables; logs -> lake -> parsers -> events; text -> search. Observer flags a pattern only when 2+ modalities agree; isolation-forest anomaly model | Runs locally; AWS written (Kinesis, S3, Glue/Athena, OpenSearch Serverless); interface logs **emulated** (PCAP + HCLTech parsers plug into `ingest.PARSERS`) |
| **Hypothesise**: NLM and Claude propose testable, pre-registered statements with predicted effect and null outcome | Scientist writes claims sealed with a hash (prediction, thresholds, null); the lab replication is pre-registered too | Runs locally (offline model); Claude on Bedrock AWS written; the research-tuned NLM: see "new model" |
| **Intervene**: change one variable, inject a fault, alter load or mobility, replicate on a second instrument | One-setting sweeps on the twin; lab fault injection; traffic-profile (load) and mobility changes; replication in the lab under two profiles; simulator tables as an independent instrument for tilt; handover-rule search | Runs locally; lab **emulated** |
| **Execute**: Step Functions protocol; EventBridge Scheduler and SQS for overnight runs; lab executor as an AgentCore Gateway tool | The protocol is an ASL state machine (local interpreter, same document deployed to Step Functions); nightly schedule; job queue; `lab_execute` gateway tool | Runs locally; AWS written (Step Functions, EventBridge Scheduler, SQS, Gateway target) |
| **Analyse and falsify**: analyst agent with Code Interpreter; critic tries confounders and alternatives; findings carry confidence and failed falsification attempts | Separate Analyst agent (statistics only); Critic: control, twin-bias confounder, fresh seeds, second instrument, head-to-head rivals, lab replication against rivals; every attempt on the card | Runs locally; Code Interpreter AWS written |
| **Publish for review**: structured cards in DynamoDB; accept / reject / request another experiment in the Amplify console; verdicts train priors | Cards with observation, hypothesis, design, evidence, confidence, limits, artefacts; review console (single-page app) with the three verdicts; verdicts reorder the Scientist and become training records | Runs locally (console at `/console/`); AWS written (DynamoDB, Amplify Hosting, Cognito, API Gateway, Managed Grafana) |
| **Reusable artefacts and a new model**: early-warning indicator, detector rule, test strategy, fine-tuning sample; finding graph in Neptune; NLM adapter fine-tuned on SageMaker, served via Bedrock Custom Model Import | All four artefacts (indicator calibrated and validated on fresh lab windows, gated); finding graph with a test-strategy query; corpus of accepted findings; LoRA training and evaluation code; merge/export for Custom Model Import | Artefacts and graph run locally; Neptune, SageMaker, Custom Model Import AWS written. **LoRA training has not run**: this network blocks the base-model download (huggingface.co returns HTTP 403) |

## Assets
| Asset | Status |
|---|---|
| Maverick digital twin | Used: RF twin, simulator tables, UE track generator, attachment rule, MRO counters |
| E2E 5G network generator | **Emulated**: four traffic profiles (`lab.TRAFFIC_PROFILES`) |
| VIAVI RAN simulator with fault injection | **Emulated**: `lab.EmulatedLab` (fault injection, parameters, load, mobility) |
| O-RAN test twin PCAP and log parsers | **Emulated**: synthesised decoded logs + our parsers, same downstream interface |
| Test automation API | **Emulated**: `lab.LabExecutor` behind the `lab_execute` gateway tool |
| OTel-LLM-8.3B-IT | Training/serving code ready (SageMaker, Custom Model Import); weights not available. Local small-model training blocked by the network (see above) |
| Strands, Bedrock AgentCore (Gateway, Code Interpreter, Runtime), scheduler role | Strands used throughout; AgentCore pieces AWS written |

## What still needs something from outside this machine
1. **A base language model**: copy an approved model folder under `models/` (or allow huggingface.co) and run
   `python -m atr.cli train-nlm --base models/<folder>`; or run training on SageMaker.
2. **An AWS account**: `cdk deploy`, then the bring-up steps in `docs/AWS_MIGRATION.md`.
3. **The real lab assets**: VIAVI, PCAP + parsers, test automation API and the E2E generator replace the emulated stand-ins
   through the interfaces named above.
