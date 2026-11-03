# The research loop: pre-registration, a second instrument, engineer verdicts, exploration

Four additions that move the system from "recovers a hidden fault" toward the proposal's loop: hypothesise, intervene,
falsify, learn from engineers, and look for something nobody planted. All four run locally with MAVERIC only
(`python scripts/check_research.py`, 48 checks). Everything below says what it is and what it is not.

## 2. Pre-registered hypotheses
Before any experiment on a claim runs, the Scientist seals it (`atr/prereg.py`): the hypothesis, the design and controls,
the predicted effect with its numeric thresholds, and the null outcome. The statement plus its time is hashed (SHA-256) and
written to the audit trail at that moment.
- The Experimenter refuses to test an unregistered claim or one whose registration no longer verifies.
- Publication refuses a tampered registration. Changing a threshold afterwards is detected.
- Every card shows the hash, the time, the predicted effect, and whether the prediction was met (a refuted claim records that
  its prediction failed).
- Checked: all 6 registrations of an investigation are in the audit trail before the Experimenter's first call.

Limit: the hash proves the statement existed before the result and was not edited since. It does not stop someone with write
access to the store from replacing a whole card; the separate audit record is what makes that visible.

## 3. A second instrument
MAVERIC ships the simulator's own RSRP tables (tilts 4, 7, 10, 13, 16). A tilt claim is replicated against those tables, not
the twin's fitted model (`atr/instrument_b.py`, `run_experiment` with `instrument: "simulator"`), on fresh seeds.

| Measured | Result |
|---|---|
| 16 tilt faults the researcher finds, replication on the tables | improvement 1.3 to 5.8, CI lower bound at least 1.26 (margin required: 0.25) |
| A wrong tilt claim (cell 2 at 8 degrees, when cell 3 is the fault) | improvement -0.75: does not replicate |
| Effect on detection | none: 16/20 tilt faults and 46/50 overall, same as before |

Limits, not hidden: the tables cover **tilt only**, so azimuth and cell-off claims show "not applicable" on the card and are not
faked. The tables exist for one tilt at a time, so a claimed value is mapped to the nearest simulated tilt within 1.5 degrees.
The twin was trained on tilts 4, 10 and 16 (7 and 13 held out), so for 4, 10 and 16 the instrument is independent of the twin's
fit but not of its training data.

## 4. Engineer verdicts
An engineer can **accept**, **reject** or **request another experiment** on any card, from the review page
(`python -m atr.cli serve`, buttons on each card) or `atr.cli verdict`. Verdicts are stored on the card, audited, and used three ways:
- **Priors:** accept/reject counts per fault kind become a Beta weight. They *reorder* what the Scientist tries first and which
  exploration hypotheses get tested first. They never skip anything: the Critic still gates every claim, so a prior can change the
  order of work, not the evidence required. The benefit in experiments saved has not been measured.
- **Follow-ups:** "request another experiment" queues a job (a persistent local queue, SQS on AWS). `atr.cli run-followups`
  re-tests the card on 32 fresh seeds (the saved field log for a fault finding, plus the simulator tables for tilt) and attaches the
  result to the card.
- **Training:** accepted and rejected findings become labelled records (`finetune_data.records_from_verdicts`, written to
  `engineer.jsonl` by `export-dataset`), kept apart from the generated cases.

`verdict` judges the *finding*; `approve` authorises a proposed *action*. They are separate on purpose. The page's form is served
on 127.0.0.1 only and has no login; do not expose it beyond a trusted machine.

## 1. Exploration: looking for something nobody planted
`python -m atr.cli explore` (Strands agents, or `--engine classic`). No fault is planted. The Scientist scans 30 single-setting
interventions (each cell: tilt plus or minus 2 and 4 degrees, azimuth plus or minus 20) on the twin, then pre-registers the best three.

**The objective was not the obvious one.** On the documented network coverage is already 99.75%, so "improve coverage" can never
find anything. The headroom is radio-link-failure risk (10.7%) and signal quality (mean SINR 5.5 dB), so the objective is failure
risk, with coverage and SINR as guardrails. Both are proxies computed from RSRP (failure risk = best RSRP below -115 dBm or SINR
below 0 dB; SINR is a full-load proxy), not MAVERIC's own handover or RLF simulation.

The Critic then tries to break each hypothesis:
1. the gain must survive **the twin's own error**: every seed also draws a random 1.5 dB per-cell offset, so the interval includes model
   uncertainty (CI lower bound above 0.5 points, and at least 90% of seeds);
2. **guardrails**: coverage may not fall more than 0.5 points, mean SINR not more than 0.3 dB;
3. tilt changes must **replicate on the simulator tables**;
4. and hold on **fresh seeds under a harsher twin error** (2.5 dB per cell).

**What it found (both engines, 41 experiments):**

| Hypothesis | Outcome |
|---|---|
| Cell 4 tilt 10 to 14 | set aside at the scan: cuts failure risk 1.5 points but costs 6.1 points of coverage |
| Cell 1 tilt 10 to 14 | set aside: costs 0.9 points of coverage |
| Cell 4 azimuth 90 to 110 | **survived**: failure risk -1.8 points (CI 1.40 to 2.21 under twin error, wins 100% of seeds); holdout at 2.5 dB: -1.78 (CI 1.08 to 2.74) |
| Cell 1 tilt 10 to 6 | rejected: gain 1.24 (CI 0.64 to 1.80) but wins only 88% of seeds under twin error |
| Cell 2 tilt 10 to 6 | rejected: gain 1.90 (CI 0.50 to 3.36), wins only 75% of seeds |

So the Critic rejected two of three and a guardrail blocked two more at the scan.

**How much to believe the one survivor.** It is a candidate for a field trial, not a validated finding:
- it is an **azimuth** change, the one kind with no second instrument, so its evidence is the twin alone (stress-tested against the
  twin's own error, but still the twin);
- the KPIs are proxies, and the twin has its own error (1 to 2 dB on held-out tilts);
- it was picked as the best of 30 scanned candidates; the twin-error draws and the fresh-seed holdout guard against picking noise,
  but selection among 30 is not fully corrected for;
- whether engineers already knew it is unknown to this system. Calling it "a finding the engineers did not know" would need an
  engineer's answer.

The card says all of this ("validated on the twin, not the real network") and proposes a **human-approved field trial** with a roll-back
condition. Nothing changes the live network.

## 6. Detector rules: an early-warning screen from accepted findings
`python -m atr.cli detectors build | report | scan` (`atr/detector.py`, `scripts/check_detector.py`, 31 checks).
Once an engineer accepts a confirmed fault finding, a **rule** is derived from it: a screen that looks at a field survey and says
"this cell looks anomalous, like finding X did", with no experiments (the full investigation costs about 90). It is a trigger for an
investigation, not a diagnosis.

Per cell, against what the twin predicts for the documented settings: `shape_rmse` (the RSRP residual after removing the cell's
constant offset), `heard_gap` (share of UEs that hear the cell vs expected) and `share_gap` (share it serves vs expected). The
offset is removed on purpose: a twin that is simply biased on a cell leaves a constant offset, a real fault changes the pattern.
When a cell shows a large offset with an ordinary pattern, the heard/serve features stand down and the scan says **"twin offset
suspected: calibrate before suspecting the network"**. A dead cell has no offset to measure, so it is unaffected.

**A rule has to earn deployment on surveys it was not built from:** thresholds come from 200 healthy calibration surveys (some with
twin mismatch, some on the rule's own cell: largest value seen x 1.15); the **false-alarm rate** is then measured on 100 fresh healthy
surveys and **detection** on 30 fresh surveys with the finding's fault, with Wilson 95% intervals. It is deployable only if false
alarms are at most 5% and detection at least 90%; otherwise it is kept as `rejected` with the reason. A rule only borderline on
false alarms (upper bound above 10%) is flagged.

| Measured | Result |
|---|---|
| Five rules, one per cell (tilt 13, azimuth, cell off, tilt 16, azimuth) | all deployable: false alarms 0-1 of 100, detection 29-30 of 30 |
| Three rules from small tilt changes (7, 13, 7 degrees on cells 3, 4, 5) | **rejected by the gate**: detection 0-3% (the same faults the researcher abstains on) |
| All five rules together on 200 fresh healthy surveys (never used for any decision) | **0 triggers** (95% interval 0 to 1.9%) |
| False alarms where the twin is wrong by 2-4.5 dB on the rule's own cell | 0 of 65 (9 of 80 before the offset guard) |
| Faults on the watched cell of any kind or value | 538 of 600 caught (90%): azimuth 100%, cell off 100%, tilt 21-48 of 48 depending on the cell |

**What it is not, measured:**
- **It is a screen, not a locator.** A fault on a neighbouring cell changes who serves the UEs nearby, so it can fire another cell's rule:
  13% of the time overall (209 of 1600), up to 61% for some neighbouring pairs (a cell 2 fault fires the cell 5 rule). It says "something is
  wrong near here"; the investigation finds the cell.
- **Small tilt changes (about 3 degrees) are not screenable** (see the rejected rules). The detector cannot beat the same limit the
  researcher has.
- **How the numbers were obtained, for honesty:** the first version used 60 calibration surveys and no offset guard; I looked at the
  validation numbers, added the guard (it fixed the own-cell false alarms) and enlarged the calibration bank to 200 to reduce threshold
  noise. The validation surveys (seeds 5000/6000+) were therefore seen during development and their numbers are slightly optimistic. The
  network figure (0 of 200) uses a set of surveys (seeds 8000+) created after those decisions and used for none of them.
- All surveys are **simulated** (the twin plus 2 dB measurement noise plus a simulated mismatch). Thresholds fitted to that noise will
  need recalibrating on real healthy surveys.
- The rule needs the twin and the documented inventory as its reference; it is not wired into the AWS Lambda or a schedule.

## 7. Mobility and handover (started)
`python -m atr.cli mobility [--fault kind:cell[:value]] [--sweep-rule]`, `run_experiment` with `objective: "mobility"`
(`atr/mobility.py`, `scripts/check_mobility.py`, 26 checks).

**What is MAVERIC's and what is ours.** MAVERIC's own pieces do the work: `UETracksGenerator` (Gauss-Markov tracks for stationary,
pedestrian, cyclist and car UEs), `perform_attachment_hyst_ttt` (hysteresis, time-to-trigger and a SINR failure threshold, the function
its MRO notebook uses), and the MRO counters (`_count_handovers`, `_count_rlf`, `calculate_mro_metric`). Ours: speeds and noise for this
1.6 km area (MAVERIC's examples assume a different scale), the twin's signal at arbitrary positions along each track (the geometry
equals MAVERIC's `GISTools` to 1e-9), per-cell KPIs and ping-pongs, and a vectorised attachment.

**The vectorised attachment is MAVERIC's logic, not an approximation.** MAVERIC's function takes 8 to 11 s per run (row-by-row pandas);
experiment sweeps need hundreds. The fast version takes 7 to 14 ms and reproduces MAVERIC's output exactly, quirks included (after a
reverted handover MAVERIC leaves the SINR of the cell it reverted from, which then drives the failure check). Checked identical on real
tracks (3 rules x 2 shadowing levels) and on 8 random signal sets that force handovers, reverts, failures and recovery (431 failure
ticks): zero disagreements, and the counters match too. Experiments do not import MAVERIC's MRO app, only `radp`, so the Lambda image
does not need it.

**Measured (30 UEs for 60 s, healthy network):**
- Realised speeds: pedestrians 1.5 m/s, cyclists 5.3, cars 15.6 (targets 1.4, 5, 15). A first measurement showed 0 m/s: that was
  `GISTools.dist` returning zero for any step under about 22 m, not the tracks.
- With MAVERIC's smooth predictions there are almost no ping-pongs or failures, and the KPIs barely respond to hysteresis and
  time-to-trigger (handovers 13.6 to 16.2 across the whole grid). Real fluctuation is what makes these problems appear, so I added
  optional correlated shadowing (**an assumption of mine: 2 dB, tick correlation 0.7, off by default**). At 2 dB: about 1.6 handovers
  per UE-minute, ping-pongs appear, and the rule matters: 3 dB hysteresis cuts handovers about 40% and ping-pongs about 65% against 0 dB.
  Beyond that it is non-monotonic (longer time-to-trigger raises handovers again); that is MAVERIC's algorithm, replicated exactly and not tuned.

**Do planted faults show up in these KPIs?** Against the healthy network, over 12 paired track seeds (|paired t| above 3), for the 50 faults:

| KPI | Significant shift |
|---|---|
| Faulted cell's share of attached time | 43 of 50 (tilt 20/20, azimuth 18/25, cell off 5/5) |
| Handover rate | 30 of 50 |
| Ping-pongs | 21 of 50 |
| Radio-link-failure ticks | **4 of 50** |

This is a best case (the healthy baseline is known exactly and the "field" is the twin). So handover KPIs are a partial fault symptom, the
time share is the strongest and is close to what the static survey already provides, and **radio-link failures are almost useless as a
symptom here**: the 5-cell cluster rarely drops below MAVERIC's failure threshold. The honest value of this work is **tuning the
handover rule** and adding mobility KPIs to what-if experiments, not finding faults better.

**Limits.** All KPIs are simulated on the twin's predicted signal, not measured. 30 UEs, 60 s, one carrier, no traffic load. The shadowing
level and correlation, and the speeds, are my choices. `GISTools.dist` treats points within about 22 m as the same place; my vectorised
geometry does not, which is irrelevant here except for a UE standing within that distance of a cell.
**Not done:** a time-series field-log format; the Scientist and Critic reading mobility logs; MAVERIC's traffic-demand simulator
(load interventions); the mobility objective on a real AWS Lambda.

### Handover-rule search
`python -m atr.cli explore --objective handover [--baseline-hyst 0 --baseline-ttt 2]` (`atr/explore_rule.py`,
`scripts/check_rulesearch.py`, 32 checks). The same discipline as the network search: single-knob changes to the rule (hysteresis, or
time-to-trigger; one setting at a time against the documented rule), pre-registered, judged on MAVERIC's interruption cost (50 ms per
handover, 1 s per failure tick, per UE-minute), guardrails (ping-pongs and failure ticks may not rise), a stress against the twin's own
error (1.5 dB/cell), and a holdout on fresh seeds. The holdout also re-tests at **1 dB and 3 dB signal fluctuation under a harsher twin
error and keeps the weaker result**, because the fluctuation level is my assumption. No second instrument exists for mobility, and the
card says so.

**The baseline rule is an assumption.** The documented network has no handover-rule inventory, so it defaults to 3 dB and 3 ticks;
pass the real one with `--baseline-hyst/--baseline-ttt`. The answer depends on it.

| Documented rule | What the search did | Result |
|---|---|---|
| 3 dB, 3 ticks (default) | 13 experiments, no hypothesis | **Nothing clears the 3% bar**: the best alternative (time-to-trigger 2) gains 0.8% in the scan (1.7% in a larger test, lower bound -0.0%, wins 62% of seeds). The rule is already near the best in this search space, and it says so |
| 0 dB, 2 ticks (a poor rule) | 31 experiments, 3 hypotheses, all survived | Hysteresis 0 to 4 dB cuts the cost **40.9%** (CI 33.1 to 51.3 under twin error, 100% of seeds). In the holdout it is **23.3% at 1 dB** (CI 13.6 to 34.3) and **22.9% at 3 dB** (18.3 to 27.9); quote the weaker figure. Hysteresis 2 and 3 also survive (36% and about 40%); 6, 8 and 10 have an identical effect to 4 and are collapsed to the smallest change |
| 3 dB, 8 ticks | 31 experiments, 3 hypotheses | **Rejected**: time-to-trigger 8 to 4 gains 11.8% at 2 dB but only 5.7% at 1 dB (wins 75% of seeds), so it "depends on my shadowing assumption" |

**What to keep in mind.**
- The cost is dominated by handovers: radio-link failures are rare here (see above), so this is mostly "fewer handovers without more ping-pongs".
- The gain depends on the fluctuation model (my assumption). That is why the weaker holdout figure is the one to quote, and why the third case was rejected.
- Longer time-to-trigger *raises* handovers in MAVERIC's algorithm (replicated exactly, not tuned), and hysteresis stops mattering above about 4 dB.
- It moves one knob at a time. A joint hysteresis-and-time-to-trigger move, or a second round from the new rule, is not implemented.
- "Nothing survived" is reported two ways: the scan found nothing worth testing, or candidates were tested and the Critic rejected them all.

## What is not done
- The Strands exploration agents run offline (scripted tool order); with Claude on Bedrock the model chooses. Not run on AWS.
- Load and mobility interventions, and handover / RLF KPIs from MAVERIC's other modules: see `docs/PROPOSAL_COVERAGE.md`.
- Pre-registration is not enforced on the AWS Lambda path beyond what the agents do before calling it.
