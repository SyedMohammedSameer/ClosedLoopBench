# Track A (final wave): concept nodes, 6 models x independent reference. Pre-registered protocol

Written 2026-09-25 ~12:05 local time. Runner: `scripts/53_multiseed_nodes.py`. The metrics and faithfulness
stages refuse to run unless this file's sha256 equals `protocol.sha256`.

**Seen before registration.**
- Wave-1 reference-standard results for the v1 seed-42 model (`runs/iclr/reference/report.md`,
  ledger A5-A7) and wave-4 faithfulness results for seed 42 (ledger I1-I5; protocol 6037da73).
- Training logs of the 5 new checkpoints (loss curves only) and the label-overlay summary
  (`data/label_overlay/summary.json`: monitor-vs-rule label agreement in the training bundle).
- Inference for the new models was started before this file was hashed (it produces probabilities
  only; no metric against any label or reference was computed or looked at). A 2-patient benchmark
  confirmed that the new inference path reproduces the cached wave-1 seed-42 probabilities exactly
  (max |diff| = 0) and printed the mean device_42 probability per head on those 2 patients (no labels).

## 1. Questions

- **Q1 (multi-seed v1 table).** Seed mean +- sd of the rule-label model (seeds 42, 1337, 2024) per
  concept: AUROC vs the rule labels (v1's metric) and vs the independent monitor reference.
- **Q2 (key question).** Do heads trained on monitor labels ("device" models) fix the errors the
  rule-label heads inherit from the rule detector? Pre-registered focus concepts, where monitor-vs-rule
  label agreement was 0.54-0.85: **tachypnea, bradypnea, tachycardia_ppg, bradycardia**. The other
  three concepts (tachycardia, hypotension, hypertension) are reported with the same statistics.
- **Q3 (faithfulness).** Is the seed-42 counterfactual result (Test 1: 5/7 concepts pass) stable across
  rule seeds 1337 and 2024, and do device heads (device_42) change it?

## 2. Data, models, windows

- **Models.** rule_s = `checkpoints/seed_s` (rule labels); device_s = `checkpoints/device_seed_s`
  (phase 2 on monitor labels from `scripts/49_device_label_overlay.py`), s in {42, 1337, 2024}. All run
  through `clbench.fused_inference` semantics (encoder -> CrossModalFusion -> CBM), MPS, fp32.
  rule_42 probabilities are the cached wave-1 probabilities (`runs/iclr/reference/cache`; checkpoints
  md5-identical to `checkpoints/{encoders,fusion,cbm}`).
- **Splits.** MIMIC-IV-WDB **test = primary** (26 patients with processed windows). **val =
  secondary** (27 patients).
  - Device models: 8 val patients (80351079, 81178488, 81811182, 81826943, 82439920, 84213440,
    84304393, 84421559) supplied the validation windows used to pick the phase-2 checkpoint on
    monitor labels. Device-model val results are flagged selection-contaminated. A `val_clean` subset
    (the other 19 val patients) is also reported for all models.
  - Rule models also used those val windows for checkpoint selection, but on rule labels.
- **Windows** (identical for every model, from script 42's cache): home modality of the concept,
  `quality_ok`, v1 channel count, probability present for all 6 models, reference present.
  The artifact (non-quality_ok) subset is not evaluated (wave 1: <= 8 such windows).
- **Reference** (script 42, alignment validated): window median (>= 10 valid 1-Hz samples) of monitor
  HR (tachycardia, bradycardia), SpO2 pulse (tachycardia_ppg), impedance RR (tachypnea, bradypnea),
  ABPm/ARTm (hypotension, hypertension); label = Table-1 threshold. **Cuff NBPm** (nearest measurement
  event within 300 s of the window centre) is the second, physically independent reference for the
  pressure concepts on ABP-present windows. Secondary: ABP heads on ECG windows at keys with no ABP
  waveform vs cuff (pressure inference without the arterial line).
- **Rule detector**: its continuous estimate (AUROC) and binary label (error), as in script 42.

## 3. Operating thresholds (fixed before any metric is computed)

- **OP-A, graph operating point (reference-free).**
  - Rule models: the v1 procedure `persistence.calibrate_activation_threshold` (best F1 against the
    rule labels, grid 0.05-0.95 step 0.05) on VAL windows of that model. For rule_42 the persisted
    v1 threshold (`persistence_calibration.json`, fitted on train+val) is also reported.
  - Device models: **0.5**, fixed, not tuned. No threshold is chosen on test, and none on val for
    device models.
- **OP-B, reference-calibrated (rule models only).** The grid threshold minimising binary error
  against the monitor reference on VAL windows. Device models stay at 0.5. This is the strongest
  operating point available to a rule-label head, so a device-vs-rule(OP-B) comparison is
  conservative for the device models.

## 4. Metrics and statistics

Per model x concept x split (clean windows):
- AUROC and AUPRC vs the reference.
- Binary error, sensitivity, specificity and balanced accuracy at OP-A (and OP-B for rule models).
- Error correction: among windows where the rule detector's binary label disagrees with the
  reference, the fraction the head (at OP-A) gets right; and the same among windows where the rule is
  right.
- AUROC vs the rule labels (v1's metric) for every model.

**Bootstrap.** Patient-cluster percentile bootstrap, 2000 resamples, 95 % CIs. Within a concept x split
all models and the rule detector are evaluated on the same windows with the same resample weights,
so paired differences are valid.
- Per seed: device_s - rule_s (AUROC, binary error, error-correction fraction).
- Pooled over seeds: the mean over s of (device_s - rule_s), computed on every resample.
- Seed mean +- sd (sd over the 3 seeds, ddof = 1) per family. The pooled CI reflects patient sampling
  only, and the seed sd is reported beside it.

## 5. Pre-registered hypotheses (test split; each of the 4 focus concepts separately; unadjusted)

- **H1 (primary): binary error.** Pooled over seeds, device(0.5) - rule(OP-A) binary error vs the
  reference has 95 % CI upper bound < 0.
- **H2: ranking.** Pooled device - rule AUROC vs the reference has CI lower bound > 0.
- **H3: error correction.** Pooled device - rule fraction correct on rule-wrong windows has CI lower
  bound > 0.
- **H4 (conservative): error vs the tuned rule head.** Pooled device(0.5) - rule(OP-B) binary error has
  CI upper bound < 0.

Verdict per focus concept: **"monitor-label heads fix inherited rule errors" iff H1 and H3 hold.** It
is "partial" if exactly one of them holds and "not supported" otherwise. H2 and H4 are reported
alongside. Per-seed differences are reported. A concept where the pooled result holds but a single
seed's CI goes the other way is flagged as seed-unstable.

Also reported, without hypotheses: the three non-focus concepts, val and val_clean, AUPRC, cuff
NBPm, and ABP-absent pressure inference.

## 6. Faithfulness (Q3)

The counterfactual battery of protocol `runs/iclr/faithfulness/protocol.md` (sha256 6037da73...) is
re-run **unchanged** (same keys, same edits and RNG seeds, same criteria C1-C4 / P1-P3 / M1-M2, same
randomized baselines re-derived from each model) on **device_42, rule_1337, rule_2024**. Key
selection depends only on rule estimates and monitor references, so keys are identical across
models. Test 1 is the priority. Tests 2 and 3 run in the same per-patient pass and are reported if
they complete.

Per-model inputs that necessarily differ from the seed-42 run:
- Stored probabilities (pointer runs, active mask keys, re-run check) are the model's own (section 2).
- The operating threshold is the model's own OP-A (section 3; device_42: 0.5). C2 at v1's seed-42
  thresholds is reported as secondary.

**Pre-registered read-outs.**
- Test-1 pass count per model, compared with seed 42's 5/7.
- "Seed-stable" if rule_1337 and rule_2024 each pass Test 1 on the same concepts as seed 42 (5 pass;
  bradycardia and bradypnea fail). Otherwise, the concepts that differ are listed.
- For device_42, a per-concept pass/fail change vs rule_42.

## 7. Compute and reporting

MPS, single process per stage, at most 2 concurrent processes for this track. Every number in the
report is read from `metrics.json` or the faithfulness `summary.json`. Failures and deviations are
listed plainly.
