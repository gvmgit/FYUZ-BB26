# Fine-tuning

Two separate jobs, with different status.

| | Status |
|---|---|
| **A. Fine-tune the digital twin** on field data from approved findings | Built and measured, runs locally |
| **B. Fine-tune a language model** (Scientist's suspect identification, claim triage) | Dataset, evaluation harness, training script and SageMaker launcher built and checked. **The training itself has not been run** (needs a GPU, a base model and your AWS account) |

## What building this found

The first honest test of twin fine-tuning exposed a flaw in the researcher, not in the tuning. When the twin was wrong
about a cell (a constant dB offset, as a real twin is) and nothing was broken, the researcher **confirmed a false fault in
10 of 10 runs** ("cell 4: tilt about 8 degrees"), because a tilt change can soak up a bias. My earlier "no fault means no
finding" check had no mismatch in it, so it could not see this.

Fix: the Critic now has a **null hypothesis** to beat. For every claim that passes the control run, the Experimenter sweeps a
constant dB offset on the same cell ("the network is fine, the twin is biased here"), and the claim is accepted only if it
beats the best offset by a margin (`agents.NULL_MIN_GAP`, CI lower bound 0.10 score units). When the offset explains the field
as well, the claim is rejected and the card carries a **twin-calibration alert** that goes to a human: take a calibration
survey and fine-tune the twin. That is what connects the two halves: the Critic says when the twin is the problem, and twin
fine-tuning is the response.

## Measured results (local, simulated field)

The "field" is the twin plus 2 dB noise plus, where stated, a hidden per-cell offset standing in for twin error. Nothing here
is a test against an independent simulator.

**The null test removes false findings**

| Condition | Before | After |
|---|---|---|
| No fault, twin mismatch on 1-3 cells, 4 configs x 10 fresh surveys (not used to pick the threshold) | 10/10 false findings (earlier tuning runs) | **0 / 40** |
| Calibration alert raised in those runs | - | 30 / 40 |

**What it costs: recall on subtle tilt faults.** 50 planted faults (every cell x 4 tilts, 5 azimuth shifts, cell off), plain pipeline:

| | Located | Wrong finding | Abstained |
|---|---|---|---|
| Before the null test, no mismatch | 50 / 50 | 0 | 0 |
| After, no mismatch | 46 / 50 | 0 | 4 (small tilt changes on cells 3 and 4) |
| After, twin mismatch on cells 1 and 4, survey A | 42 / 50 | 1 (azimuth on cell 1 reported as cell off) | 7 |
| After, twin mismatch on cells 1 and 4, survey B | 44 / 50 | 0 | 6 |

The abstentions are mostly 3 degree tilt changes: a small tilt change shifts a cell's signal almost uniformly, which a bias
reproduces just as well. More field measurements did not reliably help (2 of 8 recovered at 3,000 UEs, 1 of 8 at 6,000), so
this looks like a limit of the evidence, and the system now says "cannot tell from a twin error" instead of guessing.
`NULL_MIN_GAP` was chosen from 180 mismatch-only cases (largest margin seen 0.061) and 12 real tilt faults, then validated on
the 40 unseen runs above.

**Twin fine-tuning (A)**

| Measure | Base twin | Tuned twin |
|---|---|---|
| Held-out field error, mismatched cells 1 / 4 (dB RMSE) | 3.27 / 3.19 | 2.24 / 2.10 |
| Same, on a fresh survey the update never saw | 3.38 / 3.30 | 2.19 / 2.15 |
| Unmatched cells 2, 3, 5 | unchanged | unchanged (guard kept the old models) |
| Healthy network, calibration alerts over 12 fresh surveys | 12 / 12 | **0 / 12** |
| Healthy network, discrepancy score | 5.53 | 4.25 |
| 50-fault grid with the mismatch, fresh survey: located | 44 / 50 | 42 / 50 |

So twin fine-tuning **improves the twin's fit and clears false calibration alerts, but it does not improve fault
localisation** (the 44 vs 42 gap is within the survey-to-survey swing of the base twin alone). An earlier run suggested a gain;
that was the test sharing a survey seed with the training data, and it is not reported as a result.

**Language-model baselines (B), 35 held-out test cases** (`python -m atr.cli eval-ranker`)

| Predictor | Kind right | Cell right | Both |
|---|---|---|---|
| Majority class | 14% | 14% | 14% |
| Rule-based top suspect (0 experiments; names a cell, not a kind) | - | 77% | - |
| Full rule-based pipeline (about 84 experiments) | 94% | 94% | **89%** |
| Claim triage: confirmed claim's rank under the Scientist's current order | MRR 0.75, top-1 52% | | |

These are the numbers a tuned model has to beat. Its honest pitch is not accuracy over the pipeline (89% is already high; the
misses are tilt faults on cells 3/4 read as twin error, and twin-error cases placed on the wrong cell) but doing it with zero
experiments, or ordering claims so fewer experiments are needed.

## How it works

**A. Twin fine-tuning** (`atr/finetune_twin.py`, `python -m atr.cli finetune-twin --approved-by "Name"`)
1. The researcher explains a field log; a finding says which settings the network really had.
2. A person approves the finding (`atr.cli approve`). Unapproved findings, and "cell off" findings (no signal to learn from),
   are refused.
3. The field rows are split 70/30. The 70% update each cell's model through MAVERIC's `update_trained_gpmodel`.
4. Each cell is judged on the held-out 30%: it keeps the new model only if its error fell by at least 0.3 dB and its
   predictions moved by no more than 6 dB on average. Otherwise it keeps the old model.
5. Accepted cells are saved as a new versioned file with a registry entry (approver, source finding, before/after per cell) and
   an audit record. The base twin is never overwritten; rollback is loading an earlier file (`ATR_TWIN_MODEL` or
   `RFTwin.load_or_build(model_file=...)`).

**B. Language-model fine-tuning**
1. `python -m atr.cli export-dataset` runs the researcher on planted cases and writes `finetune/data/{train,val,test}.jsonl`.
   Two tasks: `suspect_ranking` (diagnostics in, `{"kind", "cell"}` out; kinds include `twin_miscalibration` and `none`) and
   `claim_triage` (will this claim survive the Critic, and why not). Labels come from the planted truth, so they are
   verifiable. The prompt shows raw diagnostics, not the rule-based suspicion scores, and `meta` (truth, the pipeline's own
   answer) is stripped before training.
2. Splits are by planted value and survey seed, not by random row: train tilts {4,7,13,16} and azimuth shifts {+-60,+-120}; validation
   {6,15} and {180}; test {5,14} and {+-90}. Cell-held-out splits are not used, because the label is a cell id.
3. Pilot dataset in `finetune/data`: 55 / 30 / 35 cases, 385 / 210 / 245 records. Enough to prove the pipeline, small for
   training; `--repeats` and the nightly sweep scale it.
4. `finetune/launch_sagemaker.py` uploads the data and code and creates the SageMaker training job; `finetune/train_lora.py` is
   the LoRA script (transformers, peft, trl). The job request is validated against the installed SDK; the script compiles and its
   data handling is tested.
5. After training, host the adapter on a SageMaker endpoint and run `atr.cli eval-ranker --endpoint NAME`.

## Not done, and limits
- **The language model is not trained.** No GPU here, and the document lists OTel-LLM-8.3B-IT with weights "TBA". Whether Bedrock
  can fine-tune a model you can use is unchecked. The training script is untested against real transformers/peft/trl versions.
- **The twin mismatch is simulated** as constant per-cell offsets. Real twin error has structure (clutter, terrain) that a
  constant offset does not capture, so the improvement in the real network may be smaller.
- **Small datasets and one mismatch configuration** for the twin results; the numbers are indicative, not tight estimates.
- **The pilot dataset's labels are noisy** where a fault is indistinguishable from twin error (small tilt changes on cells 3 and
  4): the label says "tilt", the evidence says "could be a bias".
- Federated fine-tuning (each operator tuning its own twin and sharing only the update) is a design, not built.


## Research-tuned language model on CPU (`atr.cli train-nlm`, atr/nlm.py)
The full loop is code, not a plan: dataset (blind-test cases plus research findings an engineer ACCEPTED), LoRA training
with loss on the answer only, evaluation on the held-out test split against the untuned base model and the rule baselines,
a versioned registry (`models/nlm_registry.json`), and `export_merged` for Bedrock Custom Model Import. The research cycle
queues a retraining job after every 3 new findings.

Status: **not run here**. The default base (SmolLM2-135M-Instruct) cannot be downloaded: huggingface.co and its mirrors return
HTTP 403 on this network, and the block is not bypassed. To run it, place an approved model folder under `models/` and run
`python -m atr.cli train-nlm --base models/<folder>`, or run the same training on SageMaker.
