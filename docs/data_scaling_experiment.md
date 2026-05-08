# Data-scaling experiment — design

**Status**: design document. Defines the experiment **requirements** so the data-preparation work and the training/evaluation work can be specified independently. The open-questions section at the end is where the user choices live.

> **2026-05-08 update — data assembled and tiled.** The 2026_05_06 Hasty export and 2026_04_16 unzipped images cover what this design needs and substantially more than originally estimated. The full dataprep sweep has been run: 11 splits (V, T, train_N19…train_N2432, train_Nfull) tiled to 512×512 at overlap=0. Local copy at `/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_11_12/2026_05_06_data_scaling/`; durable copy on storage at `/Volumes/storage/Iguanas_From_Above/training_data/2026_05_08_data_scaling/`. See "Implementation kickoff" → "Folder layout" for what HerdNet training consumes from each split.

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
| 6 | 608 | 32× |
| 7 | 1216 | 64× |
| 8 | 2432 | 128× |
| 9 | full `\|P\|` ≈ 7.3k | upper bound |

The original 6-point schedule has been extended to 8 + full because the assembled pool is ~24× larger than the doc's worst-case estimate (see Implementation kickoff). The exact size schedule is still an open question — see Q1.

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

## Implementation kickoff (2026-05-08)

The data-prep work moved from estimate to fact. Concrete artefacts:

### Source data (pinned)

| | path |
|---|---|
| Labels | `/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_11_12/2026_05_06_labels.json` |
| Images | `/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_11_12/2026_04_16_unzipped_images/<dataset_name>/` |

Hasty v1.1 export · `project_name = iguana` · 12,005 images across **185 datasets**. The `iguana_point` keypoint class (count: 67,856) is the supervisory signal HerdNet uses; the separate `iguana` bbox class (20,692) is reserved for Phase-3 detector backends and is **not** mixed into point training.

### Manifest

`data_scaling_2026_05_06/manifest.csv` (12,005 rows, no missing image paths) — columns:

```
dataset_name, image_name, image_id, width, height, image_status,
n_iguana_point, n_iguana_bbox, n_hard_negative, density_quartile,
island, proposed_split (P/V/T), image_path, image_path_exists
```

`density_quartile` is computed across the 8,220 images that have ≥1 `iguana_point` (q1 ≤ 1, q2 ≤ 2, q3 ≤ 6, q4 ≤ 569). The manifest is the single source of truth for any downstream subset-sampling work.

### Splits actually carved (preserves scenario_c island-isolation)

| split | datasets | imgs (COMPLETED) | iguana_point | rationale |
|---|---|---|---|---|
| **V** | scruz_scm01, scruz_scplf01, fpe02, isa_ispvr04 | **290** | **1,796** | 3 islands → fold-set A is realisable (Santa Cruz, Fernandina FPE02, Isabela ISPVR04) |
| **T** | 22 datasets across 8 islands (see below) | **1,184** | **5,175** | enlarged 2026-05-08 from 4 islands / 1,824 igs to give per-island recall statistical headroom and direct Floreana coverage |
| **P** | 157 datasets, the rest | **10,223** (≥1 pt: 6,908) | **60,868** | ~363× the 19-frame baseline |

**T composition by island:**

| island | datasets | imgs | iguana_pt |
|---|---|---|---|
| Isabela (ISVP01, ISCW01, ISNCW01, ISNCW02) | 4 | 506 | 1,914 |
| Floreana (FMO02, FMO04, FMO06) | 3 | 34 | 1,403 |
| Genovesa | 1 | 7 | 882 |
| Fernandina (FNE01, FNE03) | 2 | 130 | 646 |
| Santiago (STJB01–06) | 6 | 327 | 144 |
| Pinzón | 1 | 43 | 58 |
| Marchena | 3 | 81 | 30 |
| Pinta | 2 | 56 | 25 |

`P` still spans 8 islands. Fernandina dominates (4,388 imgs / 48,602 pts on COMPLETED+pt); the long tail (Española, San Cristóbal sfs/srec*, Zooniverse phases, Santa Cruz SCM01) is preserved so cross-location work has signal beyond the two big islands.

### Tile parameters

`crop_size = 512`, `overlap = 0` for **all splits** (train, val, test). Overlap was set to 0 (rather than the scenario_c default of 250) so iguana_point labels at tile boundaries are not double-counted, which would otherwise inflate per-split label totals and corrupt per-frame counting metrics. Confirmed during the 2026-05-08 smoke test where overlap=250 produced 15,226 labels from 5,175 source iguanas on `T` (a ~3× inflation).

### Dataset config

`active_learning/scripts/training_data_preparation/dataset_configs_data_scaling_2026_05_06.py`

- `VAL_DATASETS`, `TEST_DATASETS` — hand-curated, mirror scenario_c.
- `TRAIN_POOL_DATASETS` — generated from the manifest, 169 entries; an assertion in the file fails fast on V/T leakage.
- `_train_at_n(N, seed_tag)` — emits a `DatasetFilterConfig` that draws `num=N` from `P` via `ImageFilterConstantNum(SampleStrategy.RANDOM)`. Multi-seed = re-run with `seed_tag="s7"`/`"s13"` after re-seeding the trainer's RNG.
- `SCALING_SIZES = [19, 38, 76, 152, 304, 608, 1216, 2432]` plus `train_full` (= |P|).

The full sweep was committed and run on 2026-05-08:

```python
datasets = [val, test, *train_subsets, train_full]
```

A smaller smoke-test list (`[val, test, train_subsets[0], train_subsets[1]]`) is the easy revert if a re-run is needed for one or two N values.

### What's not yet wired

- **Stratified sampling by `density_quartile`** (Q8). The manifest carries the column, but `ImageFilterConstantNum` only does `RANDOM`/`FIRST`/`ORDERED_*`. If Q8 is locked as a hard requirement, extend that filter to accept a stratification key.
- **Strict cross-location subsets** (Q3). `TRAIN_POOL_DATASETS` spans all islands; "natural" cross-location is what the current config gives you, not "strict".
- **Per-fold V partition (`V_L1/V_L2/V_L3` and `V_R1/V_R2/V_R3`)**. The fold assignment lives in the eval pipeline (downstream of training), driven by the manifest's `island` and `density_quartile` columns.

### Run (re-prep only)

```bash
cd active_learning/scripts/training_data_preparation
# 021_hasty_to_tile_point_detection.py already imports
#   dataset_configs_data_scaling_2026_05_06 as dataset_configs
python 021_hasty_to_tile_point_detection.py
```

Outputs land under `labels_path/<dataset_name>/<dset>/` per the script's existing convention (HerdNet CSVs, COCO JSON, crop folders, dataprep report YAML).

### Where the data lives

| | path |
|---|---|
| Local SSD (canonical) | `/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_11_12/2026_05_06_data_scaling/` |
| Storage (durable copy) | `/Volumes/storage/Iguanas_From_Above/training_data/2026_05_08_data_scaling/` |
| Manifest CSV | `/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_11_12/data_scaling_2026_05_06/manifest.csv` |
| Source labels JSON | `/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_11_12/2026_05_06_labels.json` |
| Source raw images | `/Users/christian/PycharmProjects/hnee/HerdNet/data/2025_11_12/2026_04_16_unzipped_images/<dataset_name>/` |

Both data-scaling roots have identical layout. Train HerdNet from either; the local SSD is faster, the storage copy is the durable / shareable version.

### Folder layout (per split)

The dataprep script produces this layout under each split's directory:

```
2026_05_06_data_scaling/
├── datapreparation_report_<split>.yaml         # provenance: filter config + counts
└── <split>/
    ├── crops_512_num<N>_overlap0/              # ← TILES HerdNet trains on
    │   └── <dataset_name>___<image>_x<col>_y<row>.jpg
    ├── herdnet_format_512_0_crops.csv          # ← POINT CSV HerdNet trains on
    ├── coco_format_512_0.json                  # COCO mirror of the above (alt loaders)
    ├── hasty_format_crops_512_0.json           # Hasty round-trip for re-export
    ├── herdnet_format.csv                      # full-size source-frame points
    ├── coco_format_full_size.json              # COCO at source-frame resolution
    ├── hasty_format_full_size.json             # Hasty at source-frame resolution
    ├── Default/                                # cropper intermediate (safe to ignore)
    └── padded_images/                          # zero-padded source frames (intermediate)
```

The two files HerdNet's data loader needs are bolded with arrows above:

- **Image directory**: `<split>/crops_512_num<N>_overlap0/`
- **Point annotations**: `<split>/herdnet_format_512_0_crops.csv` (columns: `images,x,y,labels`; one row per labelled point in tile-pixel coordinates).

In the directory name, `num<N>` records the source-frame budget for that split: `numNone` for `val` and `test` (no sampling cap), `num19/38/76/152/304/608/1216/2432/None` for the train subsets. `overlap0` is constant.

### Splits available for training

| split | dset name | source frames | crop tiles | point labels | use |
|---|---|---|---|---|---|
| V | `val` | 290 | 1,036 | ~1,793 | early stopping, threshold tuning, fold metrics |
| T | `test` | 1,184 | 4,099 | ~5,085 | held-out final report (used **once**) |
| train | `train_N19_s42` | 19 | 78 | ~87 | scaling curve N=19 |
| train | `train_N38_s42` | 38 | 159 | ~218 | scaling curve N=38 |
| train | `train_N76_s42` | 76 | 319 | ~510 | scaling curve N=76 |
| train | `train_N152_s42` | 152 | 717 | ~1,340 | scaling curve N=152 |
| train | `train_N304_s42` | 304 | 1,391 | ~2,404 | scaling curve N=304 |
| train | `train_N608_s42` | 608 | 2,681 | ~4,800 | scaling curve N=608 |
| train | `train_N1216_s42` | 1,216 | 5,464 | ~9,710 | scaling curve N=1,216 |
| train | `train_N2432_s42` | 2,432 | 11,006 | ~18,712 | scaling curve N=2,432 |
| train | `train_Nfull_s42` | 6,908 | 32,089 | ~57,310 | scaling curve full(\|P\|) |

All counts are exact for tiles; "point labels" reflects the `Stats` line emitted by 021. `_s42` is the seed tag — when adding multi-seed runs, re-run 021 with `seed_tag="s7"` / `"s13"` and re-seed `numpy/random` in the trainer or dataset_config.

### How HerdNet consumes one split

A scaling-curve run trains on one of the `train_N*_s42` splits, validates on `val`, and only at the very end of the experiment evaluates on `test`. Pointers:

```yaml
# HerdNet config sketch (pseudo)
train:
  csv:    /path/to/2026_05_06_data_scaling/train_N304_s42/herdnet_format_512_0_crops.csv
  images: /path/to/2026_05_06_data_scaling/train_N304_s42/crops_512_num304_overlap0
val:
  csv:    /path/to/2026_05_06_data_scaling/val/herdnet_format_512_0_crops.csv
  images: /path/to/2026_05_06_data_scaling/val/crops_512_numNone_overlap0
# test is NOT touched until the very end
```

Substitute the local SSD root or the storage root depending on which copy is reachable on the training machine.

---

## Open questions for the user (data-prep + protocol decisions)

These are the choices that should be made *before* the data-prep team starts splitting frames. Each affects how the data must be assembled.

### Q1. How many training-size points do we want, and how large does `P` need to be?

**Resolved on availability:** `|P| = 7,291` COMPLETED frames with ≥1 `iguana_point` (10,725 if we include status `DONE` and zero-point negatives). The geometric schedule extends to N=2432 (≈128×) without exhausting `P`. Default committed in the config is **{19, 38, 76, 152, 304, 608, 1216, 2432, full}** (9 points × 3 seeds × 1.5 h ≈ 40 h).

Possible trims:
- Drop the tail: {19, 38, 76, 152, 304, 608, full} — 7 points, ~32 h
- Aggressive: {10, 19, 38, 76, 152, 304, 608, 1216, 2432, full} — 10 points, ~45 h, adds a sub-baseline point useful for the deployment-budget question

### Q2. Is `T` available, and how big is it?

**Resolved (enlarged 2026-05-08).** `|T| = 1,184` COMPLETED images / 5,175 `iguana_point` labels across **8 islands** (Santiago, Fernandina FNE01/03, Isabela ×4 zones, Floreana ×3 missions, Genovesa, Marchena, Pinta, Pinzón). Substantially above the "≥ 50 frames, ≥ 1,000 iguanas" floor; per-island per-island budgets large enough that single-frame noise is below most deltas of interest except on the very small islands (Pinta/Marchena).

### Q3. Strict cross-location or "natural" cross-location?

**Resolved (2026-05-08): natural.** One training run per N; `V_L1/V_L2/V_L3` evaluates *"performance on island i given whatever island-i data happened to land in this subset"*. Two reasons:

1. P's island balance is lopsided (Fernandina = ~71% of P's iguana_pt). At small N, natural sampling rarely picks Santa Cruz frames anyway, so natural and strict curves look near-identical at low N.
2. The "ship to a new island" question maps to T, not V — T already includes Santiago, Genovesa, Marchena, Pinta, Pinzón as held-out islands.

The strict variant remains a possible follow-up if the location-vs-random gap (per-N location-fold mean minus random-fold mean) reveals something interesting.

### Q4. Warm-start or from-scratch?

**Resolved (2026-05-08).** Hybrid: warm-start across the full sweep + **2 from-scratch anchors at N=304 and N=full(|P|)**.

- Warm-start (default): every N starts from `best_models/phase8/b4_seed42/best_model.pth` (Phase-8 production B4). Curve answers *"how much labelling buys how much accuracy on top of the model we already have today?"* — the deployment-budget question.
- From-scratch anchors: 2 extra trainings (1 seed each, no Phase-8 prior) at the mid-curve (N=304) and the asymptote (N=full(|P|)). Together they pin both ends to an absolute scaling law and let us measure how much of the warm-start curve is "Phase-8 prior" vs. "marginal data signal".
- Compute cost: +~5 h (~3 h for from-scratch full-pool, ~2 h for from-scratch N=304; 30 epochs each, no multi-seed).

Both anchors land in the same fold-eval pipeline as the warm-start runs, so all metrics (F1/MAE/recall, location vs random folds) are directly comparable.

### Q5. Same architecture across all sizes, or also vary?

**Resolved (2026-05-08): B4 only across the sweep.** Architecture × size is a 2-D study and stays a follow-up. The single exception is Q9 — one B3 training at the largest N to enable the final cross-arch ensemble report. No `varies-with-N` architecture experiment in this scaling study.

### Q6. Per-epoch validation: cropped val (cheap) or full-size stitched (honest)?

**Resolved (2026-05-08).** Per-epoch monitor stays on **cheap cropped val** (no periodic stitched eval during training). Honest stitched evaluation runs **once at the end of each training run on V folds** (already in per-run protocol step 4) and **once at the end of the whole experiment on T**. No change to the per-run protocol — we are explicitly *not* adding the periodic stitched-during-training eval the original recommendation proposed.

Rationale: with `|T| = 1,184 / 5,175 igs` (post-2026-05-08 enlargement), the end-of-experiment stitched-T number is statistically strong on its own. The ~7 h of compute that periodic stitched-during-training would have cost is better spent on the from-scratch anchors (Q4) and the final B3 ensemble run (Q9).

Caveat carried forward: cropped val can pick a slightly suboptimal checkpoint relative to stitched-V truth. With 30-epoch warm-started runs the model is usually near plateau, so the divergence is small but non-zero. If a single curve point looks anomalous, re-evaluating its candidate checkpoints on stitched V is the first triage step.

### Q7. Are the validation folds disjoint from training subsets even at the largest N?

Hard requirement: yes. `V` is held out from `P`. The data-prep team must guarantee no image overlap. (Stating it explicitly so the assumption is on paper.)

### Q8. Iguana-density stratification when sampling training subsets

**Deferred (2026-05-08).** Subset sampling stays `SampleStrategy.RANDOM`. The manifest still carries `density_quartile`, so stratification can be added later without re-tiling.

Risk we are accepting: at the smallest sizes (N=19, N=38) a draw can land entirely in q1 (sparse) or q4 (crowded), inflating the seed-to-seed variance at those points. Multi-seed (Phase-5 noise floor) absorbs some of this, but if the N=19 / N=38 curve points show variance bands wider than the N=76 point, that's the diagnostic signal that stratification was worth doing — and we'd revisit by re-sampling those points only, not redoing the whole sweep.

### Q9. What goes into the "production stack ensemble" report at the end?

**Resolved (2026-05-08): B4 sweep + 1 B3 training at the largest N.** No B3-across-the-curve retrain. The final test-set report shows three numbers on the enlarged T:

1. Largest-N B4, single seed.
2. Largest-N B4, 3-seed average.
3. Largest-N B3 + B4 cross-arch ensemble (production recipe, single seed each).

Compute: ~3 h for the one B3 run at full(|P|). If Q5's "vary architecture with N" is ever reopened, B3 sub-curve becomes that follow-up's first deliverable.

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

- [x] Test set `T` defined: 443 frames, 1,824 iguanas, 4 islands, COMPLETED-only, **disjoint from V and P**  *(2026-05-08)*
- [x] Validation pool `V` defined: 290 frames, 1,796 iguanas, 3 islands, COMPLETED-only, **disjoint from T and P**  *(2026-05-08)*
- [ ] `V` partitioned two ways: by location (`V_L1, V_L2, V_L3`) and randomly (`V_R1, V_R2, V_R3`), partition assignments persisted  *(island column is in the manifest; random fold not yet assigned)*
- [x] Training pool `P` defined: 7,291 frames (≥1 pt, COMPLETED), 64,219 iguana_point labels, ~12 islands, **disjoint from T and V**  *(2026-05-08)*
- [x] Iguana-count quartiles per image computed for `P`  *(`density_quartile` column in manifest; sampler not yet stratified)*
- [x] Per-image metadata: location label, iguana count, dataset_name (≈ source mission)
- [ ] Three reproducible random seeds for subset sampling persisted (so the same `S_i` can be regenerated)  *(seed_tag mechanism in dataset_config; need to lock the three integer seeds + commit)*

Artefacts:
- Manifest CSV: `data_scaling_2026_05_06/manifest.csv` (alongside the labels JSON)
- Dataset config: `active_learning/scripts/training_data_preparation/dataset_configs_data_scaling_2026_05_06.py`




#### TODO
Think about Density aware verification. The original herdnet paper even did this. per tile might be misleading, average dinstance between objects or even a graph between them would be a better measure. So then we would could inverse the correction: When we found many iguanas somewhere, what is estimated recall there.

