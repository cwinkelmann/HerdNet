# Data-scaling experiment — design

**Status**: design document. Defines the experiment **requirements** so the data-preparation work (handled elsewhere) and the training/evaluation work (handled here) can be specified independently. No implementation steps; the open-questions section at the end is where the user choices live.

## Goal

Quantify how the production stack's accuracy scales with training-set size, on **fixed** validation and test sets. Concretely we want answers to:

1. **Where do diminishing returns start?** Is it linear, log, or already plateaued at our current 19 training frames?
2. **Does more data fix the recall ceiling?** Phase 11 identified 4 hard-camouflage FNs that the architecture / ensemble couldn't recover. Are those FNs that *more iguanas in the training set* would fix, or are they intrinsic-to-the-resolution and need a different attack?
3. **In-distribution vs cross-location generalisation.** When we add training images from one mission, do we get better at *that* mission only, or does it transfer to other islands? Is the gap between random-fold and location-fold validation widening or narrowing as training data grows?
4. **What's the smallest training set we could ship with?** If a future deployment has only ~5–10 reference frames available, what does the curve at low N look like?

The output is a learning-curve plot (training-size on x-axis; F1 / MAE / recall on y-axis; one curve per fold type) and a final test-set number for the largest training run.

---

## Data requirements

Three disjoint datasets. They must not share images.

### Test set `T`

The single most important resource. Used **once**, at the very end of the experiment, on the largest-training-data run, to report a final unbiased number.

Requirements:
- **Held out for the entire experiment.** Never used for training, validation, threshold tuning, model selection, or per-epoch monitoring.
- **Large enough for statistical power.** Current val (12 frames, 181 iguanas) is on the small side — a single missed iguana moves recall by 0.55 percentage points. The test set should be **substantially larger** (target: ≥ 50 frames, ≥ 1000 iguana annotations) so single-iguana noise is below the deltas we care about.
- **Diverse.** Should span all islands present in the training pool, ideally also a different *mission* (date / drone altitude / lighting) so the test number reflects deployment-style generalisation, not just held-out-from-the-same-mission performance.
- **Annotations are ground-truth quality.** This is the gold standard; if there are dataset-quality issues (per the existing memory note about FMO03 missing annotations), they should be cleaned in `T` even if not in the training pool.

### Validation pool `V`, partitioned into two fold sets

Used during training for early stopping, threshold tuning, and model selection. **Smaller than `T`** because it's hit many times per run, but large enough to give stable per-fold metrics.

Requirements:
- **Disjoint from `T`** (no image overlap).
- **Disjoint from the training pool `P`** (no image overlap).
- **Spans multiple locations** (e.g. ≥ 3 islands). Critical because the location-based fold definition needs at least one held-out island per fold.
- Target size: **≥ 30 frames, ≥ 500 iguana annotations**. Each fold should have ≥ 100 iguanas to keep per-fold metrics meaningful.

`V` is partitioned **two different ways** for analysis purposes (the *images* are the same, only the assignment-to-folds differs):

#### Fold set A — by location

3 folds, each fold = one island (or one geographically distinct sub-region).

```
V_L1 = images from island 1
V_L2 = images from island 2
V_L3 = images from island 3
```

When we train on the data the model has seen, then evaluate on `V_L1`, we are measuring **cross-location generalisation**: how well does a model trained on islands 2 and 3 (plus whatever's in the training pool from island 1) recognise iguanas on island 1?

If the training-data subset under test happens to include images from island 1 (because we're sampling from the training *pool* `P`, which spans all islands), this is *not* a strict cross-location test — it's "how well does the model do on this island given some sampled training data". That's still informative, but a stricter cross-location experiment would require sampling training subsets that *exclude* the held-out island. **See open question Q3 below.**

#### Fold set B — random

3 folds, random partition of `V` into 3 equal-sized splits, stratified by per-image iguana count to keep iguana density balanced.

```
V_R1, V_R2, V_R3   ← each is ~⅓ of V, random assignment
```

Used as the **in-distribution baseline**. The random partition is a control for the location-fold partition: if location matters, location-fold metrics should differ systematically from random-fold metrics; if location doesn't matter, the two fold sets should give comparable numbers.

The two fold sets cover the same images. We compute metrics per-fold and per-fold-type.

### Training pool `P`

The universe from which training subsets are drawn.

Requirements:
- **Disjoint from `T` and `V`** (no image overlap with either).
- **As large as available.** We sample subsets from it; the maximum subset size is `|P|`.
- **Per-image annotations available.** All images in `P` must have GT iguana points (otherwise they can't be used for supervised training).
- **Span multiple locations**, ideally the same locations as `V` (so that the cross-location fold story holds).

Estimated minimum: **~200 frames** to give a meaningful learning curve up to "8× our current training set". Larger is better.

---

## Experimental protocol

### Training schedule

Train at increasing data sizes. Geometric progression with the current 19 frames as the smallest point:

| run | training size N | rationale |
|---|---|---|
| 1 | 19 | current baseline (ties to Phase 1–8 production stack) |
| 2 | 38 | 2× |
| 3 | 76 | 4× |
| 4 | 152 | 8× |
| 5 | 304 | 16× — only if `\|P\| ≥ 304` |
| 6 | full `\|P\|` | upper bound |

Sizes 5 and 6 are conditional on training-pool availability. The number of runs (and exact sizes) is an open question — see Q1.

### Sampling protocol — nested vs independent

**Nested** (`S_1 ⊂ S_2 ⊂ ... ⊂ S_k`): training set i+1 contains all images in set i, plus new ones. Cleanest learning curve because each successive run is "what was already there, now with more". Only k samples drawn total.

**Independent**: training set i is sampled fresh from `P`, ignoring previous samples. Each run is a draw from the same data distribution; the learning curve has independent variance per point. Requires k separate samples.

**Recommendation: nested.** Matches the user's phrasing ("start with our already fine base model … then add more iguanas") and gives a cleaner curve. Sacrifices independent variance estimates per point — which we partly recover via multi-seed (next subsection).

### Multi-seed at each size

To distinguish data-scaling effect from random-init noise, train **≥ 3 seeds per training size**. Phase 5 established the seed-noise floor at ~0.02 F1; differences below that are not interpretable. Without multi-seed we can't tell whether a 0.01 F1 jump from N=19 to N=38 is signal or noise.

**Recommendation: 3 seeds per size minimum.** With 5 sizes that's 15 trainings; at our current 1.5 h/training that's ~22 h sequential. Acceptable.

### Architecture, loss, augmentation, threshold

**Hold all other variables constant** during the scaling study. Per Phase 8, the current production stack is:

```
backbone   :  fmo03_full_v2_bifpn  (B4: ConvNeXt-T + BiFPN + DeformConv)
loss       :  herdnet_fmo03         (Focal + weighted CE)
dataset    :  augplus pipeline + ObjectAwareRandomCrop
threshold  :  ts=0.25 for counting, ts=0.30 for F1
ensemble   :  cross-arch B3+B4 (post-hoc on the final largest-N run only)
```

We train **a single architecture (B4)** at each size to keep the experiment tractable. Cross-architecture ensembling (B3+B4) is run **once** at the end on the largest-N run, to confirm the production recipe still wins at larger scale.

If a separate question is "does the value of cross-arch ensembling change with training size?", that's a follow-up — not part of this scaling experiment.

### Starting point — fixed warm-start from a saved Phase-8 checkpoint

Every run in the scaling sweep starts from the **same single checkpoint** — `best_models/phase8/b4_seed42/best_model.pth`, the strongest single B4 from the Phase-8 production stack (best single-seed F1 = 0.955, best single-seed MAE = 0.75). See [`best_models/phase8/README.md`](../best_models/phase8/README.md) for the full set of 6 wrapped production checkpoints; the b4_seed42 one is designated as the canonical warm-start target.

This is **deliberately not** "warm-start from the previous scaling run" (which would make run 5's quality path-dependent on runs 1–4) and **deliberately not** "from-scratch / ImageNet" (which would throw away the iguana-domain prior we already paid 12 phases of compute to learn). Instead:

- Run i starts from the **same fixed B4 checkpoint** as every other run in the sweep.
- Run i then continues training for 30 epochs on its specific training subset `S_i`.
- The only things that vary between runs are `|S_i|` and the random seed.

Why this design:

1. **Path-independence within the sweep**. Each run is an independent draw of *"what does adding N − 19 new frames to the production B4 produce?"* — the actual question we want a curve for. If we instead warm-started run i from run i−1, every run would inherit the previous run's noise.
2. **Iguana-domain prior preserved**. The warm-start already knows what an iguana looks like in drone imagery. Each run's training set adds *new locations / new lighting / more individuals* on top of that prior. Starting from ImageNet at every point on the curve would effectively re-do the Phase 1–8 work for free, wasting ~22 h of compute on a story we already know.
3. **Curve is interpretable for deployment decisions**. "F1 vs N" answers *"how much extra labelling effort buys us how much accuracy on top of the model we have today?"* — the question that maps to a real labelling-budget decision.
4. **Reproducible**. The warm-start checkpoint is committed (as a symlink to its canonical location in `output/`) so anyone re-running the sweep starts from the same place.

Caveat: this is **not** a from-scratch scaling law in the strict literature sense. The curve is conditional on the Phase-8 prior. If a future question is *"what would the curve look like starting from ImageNet?"*, that's a separate (more expensive) experiment — see Q4 below.

The 30-epoch budget per run is preserved. Empirically, a warm-started fine-tune is usually close to its final performance within 5–10 epochs because it's already near a good minimum; the remaining epochs let it adapt to the new subset's idiosyncrasies. If we see a clear plateau well before epoch 30, we can shorten the budget to save compute on the bigger sweeps.

### Per-run protocol

For each (training size N, seed) pair:

1. Sample training subset S of size N from `P` (nested across N for fixed seed; independent across seeds at same N).
2. Train B4 with the locked architecture/loss/augmentation/30 epochs.
3. Per-epoch monitor on the cropped val of `V` (cheap, just for early stopping). **Caveat**: per-epoch metrics on cropped val mislead — see refactor.md section E. For this experiment we should evaluate on full-size stitched `V` at least every K epochs for honest selection.
4. After training, full-size stitched evaluation on **each fold** of `V`:
   - `V_L1`, `V_L2`, `V_L3` (location)
   - `V_R1`, `V_R2`, `V_R3` (random)
5. Threshold sweep at the same grid as Phase 6/8 (`adapt_ts ∈ {0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70}`), per fold.
6. Save: best_model.pth, all sweep CSVs, all fold metrics, detections artefacts.

After all runs complete, run the **test set `T`** evaluation **once**, on the largest-N model (or its cross-arch ensemble, if we want to confirm the production recipe). This gives the unbiased final number.

---

## Metrics and reporting

### Per-run metrics (computed for every (N, seed, fold) combination)

Standard set, identical to Phase 12:

- F1, precision, recall (at threshold-optimal point per fold)
- MAE (mean absolute per-frame counting error)
- ME (signed; +ve = over-count, -ve = under-count)
- AP (with caveat that AP is threshold-dependent in this codebase, per Phase 2b)
- Per-frame counting bias (visualised as histogram or scatter)

### Cross-run aggregations

- **Learning curves**: training size N (log-scale x-axis) vs metric (y-axis). One curve per fold-type, with seed variance shown as error bars or shaded bands. The slope, the inflection point, and the asymptote are all interesting.
- **Location vs random gap**: difference between mean-of-location-folds and mean-of-random-folds at each N. If this gap shrinks with more data, generalisation is improving with scale.
- **Per-frame error decomposition**: at each N, how many frames have zero error? How many are off by ±1? This was the most interpretable Phase 11 / Phase 12 signal — repeating it across N tells us whether more data shrinks the long tail or just trims the easy errors.
- **Hard-camouflage FN persistence**: do the 4 hard-camouflage FNs from Phase 11 (DJI_0317 (3472, 2839), DJI_0322 (1414, 1592), DJI_0322 (1350, 1678), DJI_0331 (2191, 2764) — all on `V` if we keep `V` similar to current val) get recovered as N grows? This is the sharpest data-scaling question we can ask of this dataset.

### Test-set report (final, single-shot)

After scaling-curve trends are established, run inference on `T` with:

1. The largest-N model (single seed = 42).
2. The largest-N model averaged across seeds (3-seed ensemble).
3. The cross-architecture B3+B4 ensemble at the largest N (if we also retrained B3 in this study).

Report F1 / precision / recall / MAE / ME for each. This is the unbiased generalisation number we'll cite when describing system performance to outside readers.

---

## What we expect to learn

A few hypotheses to commit to before seeing the data:

- **F1 will plateau before MAE does.** F1 is bounded above by 1; MAE is bounded below by 0. The error-count tail (the long-tail hard examples) usually keeps shrinking even after F1 looks flat.
- **Location-fold generalisation will lag random-fold generalisation by a fixed gap.** Both curves rise with N, but the gap reflects something the data can't fix (e.g. genuinely different rock textures across islands). If the gap closes with N, then enough cross-location data is the answer; if it stays constant, location-specific fine-tuning would help.
- **The hard-camouflage FNs are mostly NOT data-scaling problems.** Phase 11 argued they're at the recall ceiling of the architecture/resolution. If 16× more training data only recovers, say, 1 of the 4, that confirms the bottleneck is elsewhere (input resolution, model capacity).
- **Diminishing returns set in around N = 50–100 frames.** Pure guess. The current 19-frame model already gets F1 = 0.97 on a small val set; doubling N might give +0.005 F1 and very little MAE improvement past that.

If any of these are wrong, the experiment was worth running.

---

## Open questions for the user (data-prep + protocol decisions)

These are the choices that should be made *before* the data-prep team starts splitting frames. Each affects how the data must be assembled.

### Q1. How many training-size points do we want, and how large does `P` need to be?

The geometric schedule {19, 38, 76, 152, 304, |P|} covers a 16× range and gives 6 points on the curve. If `|P| < 304` we lose the upper end. **Trade-off**: more points = more compute (3 seeds × N points × 1.5 h ≈ 4.5 h × N points, so 6 points = ~27 h training). Fewer points = a coarser curve.

Possible alternatives:
- 4 points: {19, 50, 150, full} — shallower curve, ~18 h
- 5 points: {19, 38, 76, 152, full} — current proposal trimmed, ~22 h
- 7 points: {10, 19, 38, 76, 152, 304, full} — adds a "below baseline" point if `|P|` is generous; useful for low-data deployment scenarios

### Q2. Is `T` available, and how big is it?

Hard prerequisite. The experiment design assumes `|T| ≥ 50 frames`. If only a smaller test set is available, the test-set report will have wider error bars. If `T` doesn't yet exist as a labelled set, the data-prep team needs to allocate frames before any training subset selection happens.

### Q3. Strict cross-location or "natural" cross-location?

For Fold set A (location-based validation):

- **Strict**: when evaluating on `V_L1`, the training subset must contain *zero* island-1 images. Requires per-fold training subsets — 3× the training runs (one set per held-out island).
- **Natural**: training subset is sampled from `P` regardless of location. `V_L1` evaluates "performance on island 1, given whatever island 1 data happened to be sampled". Cheaper (one training run per N) but less interpretable for cross-location generalisation.

The natural variant is what the design above assumes. Strict is more interpretable but 3× as expensive. Probably depends on whether the user cares about "ship to a new island we've never seen" (strict) or "fold-stratified val across our existing islands" (natural).

### Q4. Warm-start or from-scratch?

Recommendation above is from-scratch. The user's phrasing ("start with our already fine base model") could be read as warm-start. Warm-start is faster but path-dependent; from-scratch gives the cleaner scaling-law answer.

If warm-start is preferred for compute reasons, mark it as such — the curve will still be useful, just with the caveat that "+38 images warm-started" is a different intervention than "trained from scratch on 38+19=57 images".

### Q5. Same architecture across all sizes, or also vary?

Recommendation above: lock B4 throughout. If the user wants to measure whether smaller models suffice at lower N (or whether larger models pay off at higher N), that's a 2-D study (size × architecture) and 3× the compute. **Suggested follow-up, not part of this experiment.**

### Q6. Per-epoch validation: cropped val (cheap) or full-size stitched (honest)?

Per refactor.md section E and Phase 5 findings, per-epoch cropped val misleads. For an experiment whose whole point is the *scaling curve*, getting honest per-epoch signal matters more than usual. Options:

- Cheap cropped val (current): fast, misleading.
- Full-size stitched on a 1–2 frame subset of `V` every 5 epochs: ~2 min extra per run, much more honest selection.
- Full-size stitched on all of `V` every 5 epochs: ~10 min extra per run; perfectly honest, slower.

The middle option is probably right but worth confirming.

### Q7. Are the validation folds disjoint from training subsets even at the largest N?

Hard requirement: yes. `V` is held out from `P`. The data-prep team must guarantee no image overlap. (Stating it explicitly so the assumption is on paper.)

### Q8. Iguana-density stratification when sampling training subsets

When we sample S from `P`, do we stratify on per-image iguana count? Otherwise small-N subsets might accidentally contain only sparse frames or only crowded frames, creating uninterpretable variance. **Recommendation: yes — stratify by per-image iguana count (e.g. quartiles) when sampling each subset.**

### Q9. What goes into the "production stack ensemble" report at the end?

After the scaling curves are produced for B4, do we additionally retrain B3 across the same N points so the final B3+B4 cross-arch ensemble can be reported as "the production recipe at largest N"? Adds 3 seeds × 5 sizes × 1.5 h ≈ 22 h of B3 training. Optional but useful if we want a clean "production-grade scaling curve".

---

## Out of scope for this experiment

- **Architecture tuning at each N.** Locked at B4 + production hparams.
- **Augmentation tuning at each N.** Locked at augplus + ObjectAwareRandomCrop.
- **Threshold tuning per training size.** We sweep at the same grid; do not assume the threshold-optimal point shifts with N (it might — that's a finding, not a design choice).
- **Cross-validation of the scaling curve itself.** A learning curve is one realisation of "what scaling looks like for this data + architecture + protocol". Multi-seed gives variance bars; running multiple curves with different val-fold definitions or architectures is a meta-study.
- **Active learning / which images to add.** We pick training subsets via stratified random sampling. "What if we added the *most informative* frames first?" is a separate experiment.
- **Implementation details** — data-prep tooling, where the new images live, how the train/val/test splits get persisted as CSVs. Those follow once the answers to Q1–Q9 are settled.

---

## Quick checklist for the data-prep team

When the data-prep work starts, what they need from this doc:

- [ ] Test set `T` defined: ≥ 50 frames, ≥ 1000 iguanas, multi-location, cleaned annotations, **disjoint from V and P**
- [ ] Validation pool `V` defined: ≥ 30 frames, ≥ 500 iguanas, multi-location, cleaned annotations, **disjoint from T and P**
- [ ] `V` partitioned two ways: by location (`V_L1, V_L2, V_L3`) and randomly (`V_R1, V_R2, V_R3`), partition assignments persisted
- [ ] Training pool `P` defined: as large as possible, multi-location, **disjoint from T and V**
- [ ] Iguana-count quartiles per image computed for `P` (for stratified subset sampling)
- [ ] Per-image metadata: location label, iguana count, source mission (date / drone altitude if available)
- [ ] Three reproducible random seeds for subset sampling persisted (so the same `S_i` can be regenerated)




#### TODO
Think about Density aware verification. The original herdnet paper even did this. per tile might be misleading, average dinstance between objects or even a graph between them would be a better measure. So then we would could inverse the correction: When we found many iguanas somewhere, what is estimated recall there.

