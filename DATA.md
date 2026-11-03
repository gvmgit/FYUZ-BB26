# DATA

Synthetic or public data only. No operator logs, no customer data.

## Datasets

| Dataset | Where | Source and licence, or generator |
|---|---|---|
| MAVERIC RF digital twin code (`radp`) | `maveric-1.0.0 1/maveric-1.0.0/radp` | MAVERIC 1.0.0, open source, MIT licence (Copyright Meta Platforms, Inc. and affiliates; `LICENSE` in that folder). Public repository: github.com/facebookresearch/maveric |
| Simulation training data for the twin (3-cell topology, `sim_001`..`sim_005`) | `data/sim_data/sim_data/3cell` | Simulated RF data in MAVERIC's format, distributed with the MAVERIC material we started from (same MIT terms). No real network measurements. |
| OTel-LLM-1.7B-IT (network language model) | `s3://fyuz-otel-leeds-2026/base-model/otel-llm-1.7b-it/` (event bucket), copied to `s3://bb26-p-team09-data-651343917891-use2/models/otel-llm-1.7b-it/` | Published for the event; model card: Qwen3-1.7B fine-tuned on OTel telecom data, Apache 2.0. Served unchanged on SageMaker (`researcher/deploy/nlm_endpoint.py`). |
| Trained twin (Bayesian GP per cell) | `researcher/models/twin_model_map.pickle` | Generator: `atr/twin.py` `build_model_map(per_cell_samples=400, seed=0)` on the data above (`python -m atr.cli build-twin --retrain`). |
| Field measurement logs (what the agents investigate) | S3 `TelecomResearcher-Field*/field/current.csv` (per run, overwritten) | Generator: `atr/world.py` `World(twin, inventory, fault).field_observation(n_ue=1200, seed=12345)`: the twin with a planted fault, 1200 simulated UEs. Planted by `tools/track3_units.py` from the frozen test set. |
| Frozen blind test set `diagnosis_v1` | `researcher/evidence/frozen/diagnosis_v1.json` (sha256 in every declaration that uses it) | Generator: `atr.cli.random_fault(seed)` with the seeds listed in the file (one fault per kind and cell). Frozen 2026-11-02 before any run that uses it. |
| Emulated lab campaigns (KPI windows, alarms, interface logs) | S3 log lake, Glue tables | Generator: `atr/lab.py` (MAVERIC mobility and attachment code, seeded traffic profiles). |
| Network language model fine-tuning data (train 385 / val 210 records) | `researcher/finetune/data/train.jsonl` (sha256 `2f8b9853…`), `val.jsonl` (`02209ba6…`) | Generator: `python -m atr.cli export-dataset`: investigations on the twin with randomly planted faults, split by fault setting (`finetune/data/manifest.json`, 28 Oct). The answer in each record is the planted truth from code; no model writes labels. |
| Frozen NLM test set `nlm_test_v1` (245 records; 35 `suspect_ranking` items are scored) | `researcher/evidence/frozen/nlm_test_v1.jsonl` (sha256 `2675cc2d…` in the declaration) | Same generator; the test split's fault settings are disjoint from train and val. Frozen 2026-11-03 09:08 UTC, before the fine-tuning run. |
| Policy probe requests (allow / deny units) | `tools/track3_units.py` (`ALLOW`, `DENY`) | Hand-written request lists, declared in each run's declaration before the run. |
| Guardrail probe text (deny unit) | `tools/track3_units.py` `GUARDRAIL_PROBE` | Hand-written prompt-injection sentence. |

## Model outputs we rely on

| Output | Model id | Prompt | Output hash |
|---|---|---|---|
| Headline finding per blind-test item (kind, cell, estimate) | `us.anthropic.claude-sonnet-5` (Amazon Bedrock, us-east-2), through the agents in AgentCore Runtime | System prompts: `PROMPTS` in `atr/strands_agents.py` (orchestrator, scientist, experimenter, critic); request text: "Investigate the network symptom in the field log and publish findings." | Per run: the sha256 of `evidence/items/<run_id>.jsonl` in `evidence/MANIFEST.sha256` |
| Network language model second opinion (advisory, recorded as `nlm_opinion` in the trace) | `otel-llm-1.7b-it` on SageMaker endpoint `atr-otel-nlm` | Built in `atr/aws_adapters.py` `NetworkLM.second_opinion`: the symptom, per-cell anomaly scores and inventory as context | stored in the run's audit records (DynamoDB, CloudWatch) |
| Finding cards | same | same | stored in DynamoDB `TelecomResearcher-Findings*`, keyed by card id; card ids listed in the items file |
| Base and tuned answers per frozen test item (NLM tuning unit) | `otel-llm-1.7b-it`, then the same model with our LoRA adapter (SageMaker training job, `researcher/finetune/train_eval_lora.py`) | The record's own system and user messages in the model's chat template, thinking off, greedy decoding; identical for both | `evidence/items/<run_id>.jsonl` in `evidence/MANIFEST.sha256`; adapter in the job's `model.tar.gz` |

The numbers in a finding (experiment results, effect sizes, Critic checks) come from code (`atr/agents.py`,
`atr/experiment.py`, the twin), not from the model: the model chooses which tool to call next. The grader
(`atr/grader.py`) is rule-based and compares the headline with the planted truth; no model labels or grades anything.
