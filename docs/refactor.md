# Refactor sketch — design flaws and rebuild vision

**Status**: vision document, not an implementation plan. Captures pain points discovered during the Phase 1–12 benchmark work and the shape a rebuild should take, so we have a single reference before any code changes.

**Scope**: animaloc package + tools/. The Hydra config system, dataset code, and CSVDataset all need overhaul too but most of the immediate friction is in the model-output → decoder → metrics pipeline.

## TL;DR

Today the codebase fuses **model**, **loss**, **decoder (LMDS)**, **stitcher**, **evaluator**, and **metrics** into one tightly-coupled inference path. That fusion was probably reasonable when there was exactly one architecture (HerdNet-DLA) and one decoder (heatmap → local-max → adaptive threshold). It now actively obstructs:

1. Tuning a single decoder hyperparameter (`adapt_ts`) without re-running the full forward pass.
2. Batched inference (we run tiles one at a time, even though they're independent).
3. Reusing intermediate artefacts (per-tile heatmaps, stitched full-frame heatmaps) for downstream analysis or A/B sweeps.
4. Adding a fundamentally different architecture (U-Net segmentation, DETR-style queries, bbox detector) without rewriting half of the inference path.
5. Aligning per-epoch validation with deployment — the cheap cropped-val numbers consistently mislead.

The rebuild target is a **clean separation between produce → decode → score** with persistable, replayable artefacts at each boundary, batched inference, and decoder-agnostic model outputs.

---

## What the codebase looks like today

Roughly:

```
                                                                  ┌── losses ─────────────────────────┐
data → CSVDataset → Albumentations → end_transforms (FIDT/PointsToMask)
                                                                  └── computed inside LossWrapper ────┘
                                                                                   │
            backbone forward (+ aux heads) → LossWrapper.__call__ ─────────────────┘
                          │                                                        │
                          │ during training: loss dict                             │
                          │ during eval:  (predictions, loss_dict)                 │
                          ▼                                                        │
                   HerdNetStitcher                                                 │
                   ├── tiles input (size 512, overlap 120)                         │
                   ├── per-tile forward with optional 4-fold TTA                   │
                   ├── stitches per-tile (heatmap, cls) into a full frame          │
                   ├── (does NOT persist the stitched maps)                        │
                   └── hands the merged heatmap to LMDS internally                 │
                                                                                   │
              HerdNetEvaluator → PointsMetrics → metrics CSV / detections CSV     │
                                                                                   │
              Trainer ←──── ties everything together: optimizer, EMA, checkpoint  │
                                                                                   │
              tools/infer.py ── thin wrapper around inference() in animaloc/utils ┘
```

The arrows aren't lying — every stage really does talk to every other stage. Examples observed during Phase 1–12:

- **`adapt_ts` change → re-run forward.** The threshold lives inside LMDS, which lives inside the stitcher, which is invoked by the evaluator, which is built by the trainer. To change the threshold at evaluation time you re-execute the entire pipeline from disk read onward. Phase 12 documents this in detail; Phase 8's threshold sweep cost ~96 min when ~12 min would have sufficed.
- **Cross-architecture ensembling required a custom builder.** `tools/ensemble_infer.py` had to monkey-patch `_build_model` because the inference pipeline can't accept a pre-built model. It also has to *guess* that members return `(heatmap, cls_out)` tuples — works because all current backbones do, breaks for any future architecture with a different output shape.
- **`LossWrapper` couples loss computation to the model.** The wrapper holds losses internally and dispatches based on a tuple-position contract (`output_idx`, `target_idx`). Adding a new loss function for a new architecture means understanding this dispatch contract.
- **Per-epoch validation operates on cropped 512×512 patches** (`herdnet_format_512_0_crops.csv`), but final evaluation uses full-size stitched inference on `data_fmo03/val/Default`. Phase 1, Phase 2, Phase 5 all documented mismatches where per-epoch F1 disagreed with final F1 by 0.05–0.15 — the cheap signal misled selection until we made the full-size eval the gating metric.
- **Stitcher overlap fails silently above ~50 % of patch size.** Phase 10 showed that `overlap=320` on a 512-tile produces 692 736 detections (vs ~180 expected). No assertion, no warning. The `mean` reduction produces unbounded heatmap values when too many tiles overlap.
- **Detection file schema is bespoke.** `images,labels,scores,dscores,x,y,count_1,species` — `count_1` is a per-image total reproduced on every row; `dscores` (whatever those are) sit alongside `scores`; `species` is denormalised from `labels`. Not a format any external tool will be familiar with.

None of these are blocking the production stack — Phase 8 is locked. They are friction every time we want to *change* something.

---

## Pain points, organised

### A. Decoder is coupled to the forward pass

- LMDS (`animaloc/eval/lmds.py`) is invoked from inside `HerdNetStitcher` (`animaloc/eval/stitchers.py:248-268`).
- `adapt_ts`, `kernel_size`, and `neg_ts` are configured upstream, baked into the stitcher, and only consumed at the very end of inference.
- There is no public API for *"give me the merged full-frame heatmap"*. The stitcher produces one but only as an internal tensor handed straight to LMDS.
- Consequence: any sweep over decoder parameters costs full forward + stitch per setting. With 6-model cross-arch ensemble + TTA that's 12 min per evaluation; an 8-point sweep is 96 min instead of ~12 min if we cached the stitched heatmap once.
- Smaller consequence: LMDS is in `animaloc/eval/`, but it's not an evaluator — it's a decoder. The package layout is misleading.

### B. Inference doesn't batch across tiles

- `HerdNetStitcher._inference` (`stitchers.py:248-268`) builds a `TensorDataset` of tiles with `batch_size=self.batch_size`, but `Stitcher.__init__` defaults `batch_size=1` (`stitchers.py:62`) and nothing in the codebase ever sets it higher.
- For the 12-frame val set at 512px tiles with overlap 120 → ~80 tiles per frame → ~960 tile forward passes, each at batch=1.
- A 4080 SUPER could comfortably run batch=8 or batch=16 with 33M-parameter models. We're leaving a lot on the table.
- TTA's 4-fold inflation is also batch=1 today; trivially batchable as 4 augmented copies in one pass.
- Consequence: every full-size inference is 5–10× slower than necessary, which compounds with the decoder-coupling waste in section A.

### C. No persisted intermediate artefacts

- Per-tile model outputs are not saved. The only inference output is the post-LMDS detections CSV.
- The merged stitched heatmap is not saved either — it lives as an intermediate tensor inside the stitcher.
- Consequence: every re-evaluation, no matter how small the change, re-executes the most expensive part of the pipeline. Threshold sweeps, decoder swaps, ensemble experiments — all bottlenecked on re-running the forward pass on data that hasn't changed.
- `tools/inference_hparam_search.py` already implements the right caching pattern, but only for the cropped-val regime (no stitcher); the full-size stitched flow has no equivalent.

### D. Architecture and decoder are locked together

- `EnsembleHerdNet.forward` (`tools/ensemble_infer.py`) hardcodes `outs[0]` and `outs[1]` — assumes models return a `(heatmap, cls_out)` tuple. Works for every current backbone (all are HerdNet variants); breaks for:
  - U-Net segmentation (single mask output, full resolution)
  - DETR-style query decoders (set of `(class, box)` predictions)
  - Pure bbox detectors (Faster R-CNN — already in the repo at `animaloc/models/faster_rcnn.py`, awkward integration)
  - Multi-task heads (e.g. count regression + localisation)
- `HerdNetStitcher` assumes the model output is two tensors that can be tiled and stitched independently. A U-Net would output one full-resolution mask which doesn't tile cleanly without overlap-aware blending.
- `PointsMetrics` is the only metrics class wired into the production path. `BoxesMetrics` exists but is rarely used and its integration with the stitcher is untested.
- Consequence: introducing U-Net (or any non-heatmap architecture) requires rewriting the stitcher, the decoder pluggability, the loss wrapper, the metrics pipeline. There's no clean injection point.

### E. Validation does not match deployment

- Per-epoch validation: 98 pre-cropped 512×512 patches with point coordinates already remapped into patch space. Cheap (~30 s per epoch), but the model never has to deal with the full-size stitching artefacts that dominate real-world inference.
- Final/deployment: full-size 5472×3648 frames, stitched at overlap=120, decoded with LMDS. ~3 min for a 12-frame val.
- Phases 1, 2, 5 all encountered cases where the per-epoch ranking of two configs *inverted* on the full-size eval. We learned to "never trust per-epoch single-decimal differences" but the architectural fix would be to make per-epoch validation actually do at least one full-size stitched eval (e.g. on a 1–2 frame subset) so the early-stopping signal is honest.
- Consequence: dozens of training runs were wasted following per-epoch signal that didn't predict deployment performance. The Phase 5 seed-replication writeup contains the cleanest demonstration.

### F. Configuration sprawl with weak coupling guarantees

- A loss config (e.g. `density_aware_fmo03_aux`) silently expects the model to emit four outputs (`heatmap, cls_out, aux_p3, aux_p4`). Pair it with a model that only emits two (e.g. `HerdNetTimmConvNext`) and Phase 4 found out — `LossWrapper.forward` skips losses whose `output_idx >= len(output_used)` (`models/utils.py:160-161`), training proceeds quietly, the aux supervision contributes nothing. No error, no warning.
- Hydra config keys come from many directories and override-via-CLI strings (`losses=foo`, `+training_settings.stitcher.kwargs.tta=True` — note the leading `+` for non-existent keys, a footgun for users).
- Consequence: a config and a checkpoint can be combined in many silently-wrong ways. The phase docs are full of "we discovered X by accident".

### G. Silent failures and missing assertions

- Stitcher overlap ≥ 256 px on a 512 patch produces invalid output (Phase 10). No guard.
- Loss/output-index mismatch: silent (section F).
- Checkpoint loaded into wrong architecture: `load_state_dict(strict=False)` reports missing/unexpected keys to stderr but doesn't abort — easy to miss in a script.
- Threshold of 0.0 or 1.0 doesn't error: just produces degenerate output.
- Consequence: hours-long training runs can complete with semantically wrong outputs. Phase 9 / Phase 11 error analyses caught some of these post-hoc but only because we looked.

---

## Vision: rebuild around three explicit boundaries

The pieces above all reduce to one observation: **today's code is one big function that goes from data to metrics**. The rebuild should be three functions joined by two artefact contracts.

```
            ┌────────────────────────────────────────────┐
            │  PRODUCE                                   │
data ───────▶  backbone + head → "DetectionField"       │ ─── persist:  field artefact (per-tile + stitched)
            │  (heatmap, density map, bbox grid, masks) │
            └────────────────────────────────────────────┘
                              │
                              ▼
            ┌────────────────────────────────────────────┐
            │  DECODE                                    │
            │  DetectionField → list of detections       │ ─── persist:  detections artefact
            │  (LMDS, NMS, Hungarian, top-k, ...)        │
            └────────────────────────────────────────────┘
                              │
                              ▼
            ┌────────────────────────────────────────────┐
            │  SCORE                                     │
            │  detections + GT → metrics                 │ ─── persist:  metrics artefact
            │  (PointsMetrics, BoxesMetrics, MaskIoU)    │
            └────────────────────────────────────────────┘
```

Each boundary should:
- have a **typed schema** (Pydantic / dataclass / TensorDict / similar)
- be **persistable** (`to_disk` / `from_disk`) so a sweep over parameters of a downstream stage doesn't re-run the upstream stage
- be **swappable** — replacing the producer (e.g. heatmap → U-Net mask) doesn't force rewrites in the decoder or scorer, and replacing the decoder (LMDS → Hungarian matching) doesn't force a new forward pass

### What "DetectionField" buys us

The first artefact is the most consequential. A `DetectionField` is whatever the model produces, before any thresholding or selection happens:

- HerdNet-style: `(heatmap[H, W, C], cls_logits[H/down, W/down, C])`
- U-Net segmentation: `mask_logits[H, W, C]`
- Bbox detector: `(boxes[N, 4], scores[N], class_logits[N, C])`
- DETR-style: `(query_boxes[Q, 4], query_logits[Q, C])`

Concrete consequences if we have a typed `DetectionField` API:

1. **Threshold sweeps become essentially free** (section A). Cache the stitched `DetectionField` once; rerun the decoder at every parameter setting in milliseconds.
2. **Cross-architecture ensembling becomes a method on the field**, not a new tool. `mean([f1, f2, ...])` if the fields are compatible; otherwise a registered combiner (e.g. heatmap + bbox → vote union).
3. **Adding U-Net is "implement a new field type and a new decoder"**, not "rewrite the inference pipeline". The producer and decoder for U-Net live alongside the heatmap producer and decoder; the trainer, stitcher, scorer don't change.
4. **TTA is a producer-side combiner** that takes K augmented views and returns one merged field. It has nothing to do with the decoder or scorer.
5. **Per-epoch validation becomes "compare a few stitched fields against GT"** — same path as deployment, just on a smaller image set. The cheap-vs-honest mismatch goes away.

### Stitching as a field-merge, not a model wrapper

`HerdNetStitcher` should become a stage in the producer that:

1. Takes input image and a model.
2. Tiles, **batches** the tiles (real batching, not 1-at-a-time).
3. Runs forward passes.
4. Merges per-tile fields into one stitched field via a registered merge strategy (heatmap-mean, max, gaussian-blend, ...).
5. Returns the stitched `DetectionField`. **Does not run any decoding.**

That single change addresses sections A, B, and C simultaneously. It also adds an obvious place to assert overlap sanity (section G) and to persist intermediate artefacts.

### Decoders as plugins

LMDS as it stands is a decoder (heatmap → list of points) hardcoded to one supervision style. A decoder registry (`@DECODERS.register()` similar to the existing `@MODELS.register()`) lets us:

- Swap LMDS for a learned P2P-style decoder without rewriting the rest of the pipeline.
- A/B compare decoders on the *same cached field artefact* — a useful experiment we can't do today.
- Fix the LMDS-specific oddities cleanly: `adapt_ts` semantics (relative vs absolute), `neg_ts` interaction, kernel size effects. Each becomes a hyperparameter of the decoder, not of the inference pipeline.

### Schema standardisation

The detections CSV today has columns `images,labels,scores,dscores,x,y,count_1,species` — none of which are standard. A rebuild should pick one schema (probably COCO-like for cross-tool compatibility) and reuse it everywhere — training, evaluation, error analysis, downstream tools. Same goes for checkpoints (we already started this with the `training_info` dict, but old checkpoints don't have it; ensemble loading has fragile fallbacks).

---

## Principles for the rebuilt architecture

1. **Separate produce / decode / score.** Each stage is a function with a typed input and typed output. The output of one stage is the input to the next; the output of any stage can be cached on disk with a single call.
2. **Make every stage replayable from artefacts.** If you have a `DetectionField` artefact, you should be able to run any decoder on it without instantiating a model. If you have a detections artefact, you should be able to run any scorer without re-running the decoder.
3. **Batched inference is the default.** Anything that does inference should accept a batch dimension. The current `batch_size=1` default is a footgun; at minimum it should default to "as large as fits".
4. **Validation matches deployment.** Per-epoch validation must run the same produce → decode pipeline as final inference, even if it does so on a small subset of full-size frames for speed.
5. **Architecture choice is a producer choice, not a pipeline rewrite.** Adding U-Net, DETR, or a Mask R-CNN should add files in `producers/`, not edit files in `stitcher/`, `decoder/`, `scorer/`.
6. **Loud failures over silent ones.** Overlap-too-large, output-index-out-of-range, decoder-incompatible-with-field — all hard errors with explanatory messages, not silent skips or unbounded outputs.
7. **Detection schema is one well-defined format, used everywhere.** Probably COCO point/bbox JSON. Bespoke columns (`count_1`, `dscores`, `species`-mirroring-`labels`) get retired.
8. **Hydra configs validate against the producer/decoder they target.** If a loss expects 4 outputs and the model emits 2, the config compose step should error before training starts — not silently skip the surplus losses.

---

## What this enables (concrete examples from the Phase work)

If the rebuild had been in place going into Phase 1:

- **Phase 6 / Phase 8 sweeps would have run in ~12 min instead of 96 min.** The threshold sweep would have been "produce once, decode 8 times".
- **Phase 4 would have failed loudly instead of silently.** The native loss config expecting 4 outputs against a 2-output model would have been a config-validation error, not a silent silent training run with dead aux heads.
- **Phase 5 (3-seed replication) would have given honest per-epoch signal.** Because per-epoch validation would have been at least one full-size stitched frame, the ranking of B3 vs B4 across seeds wouldn't have hidden behind cropped-val noise.
- **The Phase 9 error analysis is most of a stage already**: it consumes the detections artefact, joins with GT, computes per-error visualisations. Today it's a one-shot script in `/tmp`; it should be a stage in `tools/`.
- **Adding U-Net for the hard-camouflage FN bucket** (Phase 11's recall ceiling) becomes: implement a `MaskField`, a U-Net producer, a mask-based decoder, plug into the existing scorer. Doesn't touch any of the existing HerdNet code path.

---

## Open questions before any implementation starts

These are decisions you'll need to make. None are obvious; many trade off effort against future flexibility.

- **Field representation.** TensorDict (PyTorch native, simple), or a small Pydantic / dataclass hierarchy (typed, extensible)? Tensor-only (simple) or include metadata (origin tile coordinates, model identifier, etc.) for traceability?
- **Persistence format.** `.pt` (Torch native, opinionated), `.npy` per channel (universal, more files), or a single `.zarr` / `.h5` per frame (good for streaming + chunking)? The choice affects how easy it is to peek at intermediate results from a notebook.
- **Detection schema.** COCO-style JSON (industry standard, verbose), or a flat parquet / CSV with COCO column names? COCO supports both points and bboxes; we'd need to extend it for FIDT density-radius targets.
- **Backwards compatibility window.** A rebuild can migrate the existing checkpoints (`training_info` is enough metadata to instantiate the right model class) but the existing CSV outputs and Hydra configs would break. Do we keep a v1 compatibility shim, or hard-cut at v2?
- **Scope of the per-epoch validation alignment.** "Run the production pipeline on a 1-frame subset every N epochs" is cheap and fixes the worst of the per-epoch lying. "Run the production pipeline on the full val set every N epochs" is slow but the most honest. Where's the right point?
- **Whether to keep Hydra at all.** Configs accumulate cruft fast. A simpler config (Pydantic settings with overlay files, or just dataclasses with CLI parsing) might serve a focused codebase better. But Hydra is the user-facing surface today and a switch is intrusive.
- **What goes in `animaloc` vs `tools`.** Today the package contains training, evaluation, and helpers; `tools/` contains entry-point scripts. The split is fuzzy. A rebuild should clarify: package = library, tools = scripts that orchestrate library calls. The producer/decoder/scorer interfaces probably belong in the package as `animaloc.produce`, `animaloc.decode`, `animaloc.score`.

---

## Out of scope for this document

- Specific class hierarchies, file layouts, or migration steps — this is a vision doc, not an implementation plan.
- The training loop overhaul. The Trainer class has its own collection of issues (auto-LR, EMA bolted on optionally, validation interleaving) but that's a separate refactor — fix produce/decode/score first; the trainer cleanup follows naturally because the training-time validation path will be using the same boundaries as inference.
- The dataset overhaul. CSVDataset works fine for the FMO03 scale but the Albumentations coupling and the `end_transforms` (FIDT, PointsToMask) are entangled with current architecture assumptions. Redo when adding a non-heatmap producer.
- The loss-system overhaul. `LossWrapper` with `output_idx`/`target_idx` dispatch is clever but fragile. Probably wants to become per-architecture loss bundles that know what their producer emits.

These all need attention eventually, but the produce/decode/score split is the cut that unblocks everything else. Once those boundaries exist, refactoring the training loop or the dataset layer is a much smaller surgery because the connecting interfaces are stable.



## Other TODOs (graded)

A pragmatic punch list of additional improvements, with an honest take on each: what it buys, what to watch out for, and roughly where it sits relative to the produce/decode/score work above. Active learning is intentionally not in this list — it's handled in a separate model-agnostic repo and shouldn't get reimplemented here.

### Tier 1 — biggest leverage, do first

- **Full-size inference at a configurable interval during training, plus at end.** The single highest-leverage item. Per-epoch validation on the cropped val set has caused multiple wrong calls across Phases 1–5 (this doc, section E, and `phase5_seed_replication.md`). Wire it behind a `cfg.training_settings.full_size_val_every` knob (default 10), running on a small (1–2 frame) subset for cheap honesty plus full `V` at the end. Cost ~10 % of training wall-clock, immediate impact on every experiment afterwards.
- **Promote the Phase 9 / 11 / 12 analysis scripts into a real `animaloc.analysis` module.** Currently they're one-shot scripts in `/tmp`. Functions worth extracting:
  - `extract_errors(detections, gt, radius)` → FN/FP record list
  - `crop_errors(image_dir, errors, patch=256)` → annotated image patches
  - `plot_pr_curve(sweep_csv, ax=None)` and friends from `tools/plot_metrics.py`
  - `compare_runs(run_a, run_b)` → diff report
  Future error analysis becomes ~5 lines of code instead of 200. Big multiplier for any paper writing.
- **Better error metrics for counting.** Current MAE = 0.50 is great; per-frame % bias on small-N frames is misleading (`phase12_metrics_plots.md` shows DJI_0270 with GT=2, predicted=3 as "+50 %"). Add:
  - **SAPE** = `|pred − gt| / max(pred, gt)`, symmetric, bounded [0, 1]
  - **Density-stratified MAE** — bucket frames by GT count (e.g. quartiles), report per-bucket. Tells the user *"strong on dense frames, weak on sparse"*, which a plain MAE hides.
  - Skip naive per-frame % unless paired with the GT count — that one really does mislead.

### Tier 2 — speed up training so future experiments are cheaper

- **Drop Albumentations for Kornia.** The license claim is shaky (Albumentations is MIT, fine for commercial use), but the **speed argument is solid**: Phase-5 GPU was at 0 % util while CPU did augmentation. Kornia is GPU-native, MIT, and drop-in for most transforms we use; a few uncommon ones (`PlasmaShadow`, `ISONoise`) need shims or removal. `torchvision.transforms.v2` is a lighter alternative if Kornia's surface area feels too big. NVIDIA DALI is faster still but a heavy dependency that's hard to debug; skip unless throughput becomes the explicit bottleneck.
- **Optimise `ObjectAwareRandomCrop` — disambiguate two things first.** The TODO conflates:
  - *"Compute crops preemptively"* — only valid if you cache a *set* of valid crop coordinates per image (not the actual crops, which would lock in a fixed crop set and kill the diversity that makes the transform work). Move keypoint-constraint computation offline, sample coordinates at training time.
  - *"On GPU"* — straightforward with Kornia: do the actual tile slicing on the device. Pairs with the previous item.
  Combine both: precomputed coordinate cache + Kornia GPU slicing. Together they remove most of the CPU bottleneck.
- **Multi-GPU training.** Conditional value. Single 4080 + 33 M-param B4 currently runs fine. Pays off when:
  - Backbones grow (ConvNeXt-Base/Large)
  - Resolution training increases (down_ratio 4 → 2 to attack the Phase-11 hard-camouflage FN bucket)
  - Datasets grow past ~1000 frames (per the data-scaling experiment)
  Use `torch.nn.parallel.DistributedDataParallel` directly — ~30 lines of boilerplate. **Resist** introducing PyTorch Lightning just for this; it's a 200KLOC dependency for a 30-line problem. **Cheaper interim**: gradient accumulation for larger effective batch on a single GPU.

### Tier 3 — make it deployable for partners

These four are cousins; design once, share infrastructure.

- **`pip install herdnet` → fully usable package.** Scope it carefully: target *inference + light fine-tuning*, **not** training-from-scratch (training has too many local-environment assumptions — data layout, CUDA, wandb). Concretely needs: standalone inference path with no `data_fmo03/` references; canonical CLI; sample data + sample model auto-pulled from HF on first call; docs that don't reference repo paths.
- **Simple HTTP API.** Useful for ecologist partners and demos. Risk: if it grows beyond one `/predict` endpoint, it pulls in real ops work (auth, async, rate limits, observability). Decide upfront whether this is "research demo" (FastAPI + one route, done in a day) or "deployment service" (different beast). Also: **PyTorch Wildlife** already provides a similar service for wildlife detection — check if you can plug into theirs rather than building one.
- **Simple ssh-sync deployments.** Probably one helper script wrapping `rsync`: configs + data → server, weights/results → back. Don't over-engineer with state machines. Subsumes into the RunPod recipe below.
- **RunPod.io recipe.** Concrete pieces: a `Dockerfile` for the env (CUDA + PyTorch + animaloc), a `runpod_train.sh` doing pod startup → rsync data in → train → rsync results out → pod shutdown, and a spot-pod variant with checkpoint resume for cheaper preemptible training. Pairs with ssh-sync — the same script template covers both.

### Tier 4 — nice-to-have hygiene, easy wins

These are invisible improvements that pay off quietly the next time something goes wrong.

- **Reproducibility manifest per run.** Emit a `manifest.yaml` with `{git_commit, dataset_hash, hparams, seed, env_lock}` alongside `best_model.pth`. Bit-exact reruns on demand. Cheap to build (one hook); the next time someone asks "what produced this number?", you have an answer in five seconds.
- **Dataset versioning.** Annotations evolve (the FMO03 missing-iguana finds; cleaning passes for Phase 13's test set `T`). Hash the dataset CSV at training time, embed in checkpoint metadata, refuse to evaluate on a mismatching dataset version unless `--force`. Prevents silent train/eval-on-different-data mistakes.
- **Calibrated uncertainty / conformal prediction.** Field users want intervals (*"between 10 and 14 iguanas, 90 % CI"*), not point estimates with `adapt_ts` knobs. Conformal prediction needs one calibration set and is model-agnostic — ~50 lines on top of the existing ensemble. Turns the model output from a number into a decision-support signal that ecologists can actually act on.
- **Continuous evaluation / drift detection.** Once deployed in the Galapagos, track mean detection density per frame, confidence-distribution shape, tile-coverage stats. Alert when these drift from training-time baseline. Order of magnitude less work than usual drift detection because the existing 6-model ensemble disagreement is a free uncertainty signal.
- **Open-source the Phase-8 stack on HuggingFace.** README already references HF for the original Delplanque model. Push the 6-checkpoint production ensemble + a load-and-run snippet. Other wildlife groups working on small-object aerial detection benefit, and citations climb. Cost: an afternoon.

### Skip / explicitly not doing

- **Active learning.** Handled in a separate, model-agnostic repo on purpose; reimplementing here would create maintenance overhead and divergence. The data-scaling experiment uses random sampling intentionally for that reason.
- **Mobile / TFLite / on-device inference.** Tempting but premature — drone partners overwhelmingly do batch processing back at base, not real-time onboard. Don't optimise for an unconfirmed use case. Revisit only if a partner specifically asks.
- **PyTorch Lightning.** Resist. It's a large, opinionated dependency that buys very little for our specific shape (one architecture family, few callbacks, custom validation). Multi-GPU is solvable in 30 lines of DDP without it.

### Suggested order of work

If we were sequencing the TODOs against the produce/decode/score refactor:

1. **Validation alignment** (Tier 1, full-size every N epochs) — do *before* any new experiments to stop bad-signal selection.
2. **Analysis library** (Tier 1) and **better error metrics** (Tier 1) — multiplier for everything afterwards.
3. **The produce/decode/score split** itself (the body of this doc) — unblocks everything else.
4. **Albumentations → Kornia** (Tier 2) and **GPU ObjectAwareRandomCrop** (Tier 2) — landed together, cuts training time materially. Best done after the dataset-overhaul becomes part of the refactor.
5. **Reproducibility manifest** (Tier 4) and **dataset versioning** (Tier 4) — invisible but high-leverage hygiene; small enough to slip into any of the above PRs.
6. **Deployment track** (Tier 3 — pip-package, HTTP API, ssh-sync, RunPod) — group into one focused sprint when partners actually need it.
7. **Multi-GPU** (Tier 2) — defer until experiments grow to need it.
8. **Calibrated intervals**, **drift detection**, **HF release** (Tier 4) — pick up opportunistically as partners ask for production-grade artefacts.