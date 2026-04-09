---
name: training-insights
description: Architecture and training strategy insights from completed experiments. Use when planning new training runs, choosing models, or designing dataset pipelines.
---

# Training Insights

Accumulated knowledge from training experiments on the iguana detection task. Apply these insights when advising on model selection, dataset preparation, training configuration, or experiment design.

## Architecture Rankings (iguana detection, 512x512 crops)

1. **ConvNeXt Tiny** — Best overall. Consistently highest F1 (0.93-0.94). Stable convergence, minimal oscillation. Use as the default choice.
2. **Hybrid CNN+Transformer** — Near-identical to ConvNeXt on crops (0.93). Trains slower (40 epochs vs 30) and uses more VRAM. Choose only if transformer attention is specifically needed.
3. **DLA34** — Weakest at ~0.86. Erratic F1 oscillation during training (swings of 0.3-0.5 between validation epochs). Faster per-epoch due to smaller model but needs more epochs to stabilize. Not recommended unless VRAM is severely constrained (~2.4GB vs ~6.5GB for ConvNeXt).

## Dataset Strategy: Crops vs Hard Negatives vs ObjectAwareRandomCrop

### Pre-cropped patches
- Patcher with overlap (e.g. 512x512, overlap=250) produces good training data
- Each crop is a fixed 512x512 tile — no resizing, no information loss
- ~1430 annotated crops from 19 full-size images (FMO03 dataset)

### ObjectAwareRandomCrop from full-size images (equivalent when working correctly)
- Uses `ObjectAwareRandomCrop(height=512, width=512)` on full-size drone images (~5472x3648)
- Requires `augmentation_multiplier: 75` to match step count (`ceil(num_crops / num_images)`)
- **With the v2 fix applied:** achieves F1=0.94, matching pre-cropped patches
- **Empty probability sweep (0%, 5%, 10%, 15%):** all score ~0.94 — no significant effect
- Advantage over pre-cropping: each epoch sees different random crops, more augmentation diversity
- Set `augmentation_multiplier` to `ceil(num_crops / num_full_images)` for comparable epoch length

### Hard negatives (marginal impact)
- Adding empty crops as `hard_negative` class (label=2) with class weight 0.1
- **ConvNeXt:** Slightly hurts (0.94 -> 0.93) — already good at rejecting background
- **DLA34:** Slightly helps (0.86 -> 0.87) — weaker models benefit from explicit negatives
- **Hybrid:** No meaningful difference (0.93 -> 0.93)
- **Verdict:** Not worth the added complexity for strong models

## ObjectAwareRandomCrop Configuration

### Stitch-aware settings (recommended for stitcher inference)
```yaml
ObjectAwareRandomCrop:
  height: 512
  width: 512
  min_edge_distance: 0       # uniform keypoint placement (not biased to center)
  empty_probability: 0.05    # 5% pure background crops
  edge_probability: 0.2      # 20% of crops place keypoint in stitcher overlap zone
  edge_zone: 120             # match stitcher overlap (overlap=120)
  p: 1.0
```

- `min_edge_distance: 0` gives near-uniform spatial distribution of keypoints in crops (3.7% within 5px of edge vs 3.9% ideal). The old default of 10 was slightly center-biased.
- `edge_probability: 0.2` boosts keypoints in the overlap zone (0-120px from edge), matching what the model sees during tiled inference.
- `empty_probability`: has minimal effect on F1 (0.94 across 0-15%). Set to 0.05 for a small amount of background diversity.

### Critical bug fix: albumentations v2 compatibility
The `ObjectAwareRandomCrop` was **completely broken** in albumentations v2.0.8. The method `get_params_dependent_on_targets` (v1 API) was renamed to `get_params_dependent_on_data(self, params, data)` in v2. Without this fix:
- Crop coordinates were never computed (defaulted to crop_x=0, crop_y=0)
- Every crop was the top-left 512x512 corner of every image
- All keypoints fell outside the crop and were removed
- The model trained on the same top-left corner with zero annotations

This explained the 15-point F1 gap between pre-cropped and ObjCrop in Experiment 1.

## Augmentation Pipeline

### Recommended augmentations for point detection
The full `augplus` pipeline achieves the same F1 as the basic pipeline on the current val set, but provides better robustness for deployment across different conditions.

```yaml
# Geometric
HorizontalFlip(p=0.5), VerticalFlip(p=0.5), RandomRotate90(p=0.5)
ShiftScaleRotate(scale_limit=0.15, rotate_limit=45, p=0.5)
Perspective(scale=0.05, p=0.2)

# Resolution / altitude (OneOf, p=0.3)
Downscale(0.5-0.9) | GaussianBlur(3-7) | Sharpen(0.2-0.5)

# Lighting & shadows
RandomBrightnessContrast(p=0.3), RandomGamma(80-120, p=0.2)
PlasmaShadow | RandomShadow (SomeOf n=1, p=0.2)

# Color
HueSaturationValue(p=0.3), RGBShift(+-10, p=0.2), CLAHE(p=0.2)

# Noise & compression (SomeOf n=1, p=0.2)
GaussNoise | ISONoise | ImageCompression(60-90)

# Occlusion
CoarseDropout(3-8 holes, 8-25px, p=0.15)
```

### OneOf/SomeOf in Hydra configs
`_load_albu_transforms` now supports container transforms. Use suffix convention for duplicate keys:
```yaml
OneOf:
  transforms:
    GaussNoise: {p: 1.0}
    ISONoise: {p: 1.0}
  p: 0.2
SomeOf_shadows:        # suffix allows multiple SomeOf in same config
  transforms:
    PlasmaShadow: {p: 1.0}
    RandomShadow: {p: 1.0}
  n: 1
  p: 0.2
```

## Precision/Recall Tuning

### Model selection metric affects recall
The evaluator supports `validate_on: f1_score | f2_score | f5_score | recall | precision`.
- **F1 (default):** balanced precision/recall — tends to produce high-precision models
- **F2 (beta=2):** weights recall 2x over precision
- **F5 (beta=5):** strongly recall-biased — `recall_f5_ts15` achieved F5=0.961

### Detection threshold (`adapt_ts`) affects precision/recall tradeoff
- `adapt_ts=0.3` (default): conservative, high precision
- `adapt_ts=0.15`: catches more detections, higher recall
- `adapt_ts=0.10`: aggressive, may overdetect

### Strategy: train for recall, tune precision at inference
Train with `validate_on: f2_score` or `f5_score` and a lower `adapt_ts`. The saved model will favor recall. At inference time, raise the confidence threshold to control false positives. This is better than training a precision-biased model and trying to recover missed detections.

## Training Configuration Notes

### ConvNeXt Tiny
- `lr: 8e-5`, `batch_size: 4`, `epochs: 30`, `down_ratio: 4`
- `warmup_iters: 500`, `patience: 8`
- Converges by epoch 10, plateaus at epoch 20+
- VRAM: ~6.5GB

### DLA34
- `lr: 3e-4`, `batch_size: 8`, `epochs: 30`, `down_ratio: 2`
- `warmup_iters: 300`, `patience: 8`
- Higher learning rate needed due to simpler architecture
- VRAM: ~2.4GB — can run alongside ConvNeXt on a 16GB GPU

### Hybrid CNN+Transformer
- `lr: 8e-5`, `batch_size: 4`, `epochs: 40`, `down_ratio: 4`
- `warmup_iters: 600`, `patience: 10`
- Needs 10 more epochs than ConvNeXt to reach similar performance
- VRAM: similar to ConvNeXt

## Validation Set Notes

- Val crops must be pre-generated and stored alongside a `herdnet_format` CSV
- Use non-overlapping crops for validation (overlap=0) to avoid counting the same detection twice
- Val set should come from different transects/flights than training (FMO05 for val when training on FMO03)
- Always use **absolute paths** in dataset configs — `${oc.env:PWD}` breaks when Hydra changes the working directory

## Bug Fixes Log

### ObjectAwareRandomCrop v2 API (2026-04-08)
- `get_params_dependent_on_targets(self, params)` renamed to `get_params_dependent_on_data(self, params, data)`
- Also fixed in `ObjectAwareRandomCropEdgeBlackout` subclass
- Without fix: all ObjCrop training was broken (F1 0.63-0.80 instead of 0.94)

### ME metric accumulation (2026-04-08)
- `_sum_error` and `_agg_sum_error` were not reset in `Metrics.flush()` — ME was cumulative across validation images
- `_sum_error` was not aggregated in `Metrics.aggregate()` — post-aggregation ME was wrong
- `evaluators.py` summary used `iter_metrics.me()` (flushed) instead of `self.metrics.me()`

### OneOf/SomeOf config support (2026-04-08)
- `_load_albu_transforms` extended to handle container transforms (`OneOf`, `SomeOf`)
- Suffix convention (`SomeOf_noise`) allows duplicate container keys in Hydra YAML

## Experiment Design Recommendations

1. **Quick comparison:** Run ConvNeXt with ObjCrop + augplus — one run covers architecture + augmentation
2. **Architecture comparison:** Run all 3 models x clean crops (3 runs). ObjCrop and hard negatives don't change the ranking
3. **New dataset:** ObjCrop from full-size images works as well as pre-cropping (with the v2 fix). Choose based on convenience — pre-cropping is simpler, ObjCrop is more flexible
4. **Recall-biased model:** Use `validate_on: f5_score` + `adapt_ts: 0.15`, then tune confidence at inference
5. **Data local:** Copy training data to local disk (`data_fmo03/`) — avoid NFS latency during training
