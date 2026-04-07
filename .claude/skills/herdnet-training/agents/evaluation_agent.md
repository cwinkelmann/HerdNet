# Evaluation Agent

## Role
Guide users through interpreting HerdNetEvaluator output, selecting the correct matching_radius, understanding per-class metrics, and performing cross-run comparisons.

## HerdNetEvaluator Overview

Source: `animaloc/eval/evaluators.py`

The `HerdNetEvaluator` is a specialized evaluator for HerdNet models. It:
1. Runs inference on a validation/test dataset
2. Applies LMDS to get point detections from heatmaps
3. Matches detections to ground truth using Hungarian matching within a radius
4. Computes F1, precision, recall, MAE, RMSE, AP per class

```python
from animaloc.eval.evaluators import HerdNetEvaluator
from animaloc.eval.metrics import Metrics

# Metrics with matching radius
metrics = Metrics(
    threshold=75,           # MATCHING RADIUS in pixels -- CRITICAL
    num_classes=3            # Must match model's num_classes
)

evaluator = HerdNetEvaluator(
    model=wrapped_model,     # LossWrapper-wrapped model
    dataloader=val_loader,   # Validation DataLoader (batch_size=1)
    metrics=metrics,
    lmds_kwargs={
        'kernel_size': (5, 5),
        'adapt_ts': 0.5,
        'scale_factor': 1,
        'up': True           # True when no stitcher, False with stitcher
    },
    device_name='cuda',
    print_freq=100,
    stitcher=stitcher,       # Optional: for evaluating on large images
    work_dir='./eval_output',
    header='Validation'
)
```

## Matching Radius -- CRITICAL

The `threshold` parameter in `Metrics` defines the matching radius in pixels. A predicted point is considered a True Positive (TP) only if it falls within this radius of a ground truth point.

### Why 75px, Not 25px

The default matching radius in the original codebase is 25px, but this is **far too small for iguana detection**:

- Iguanas in the Galapagos orthomosaics can be 40-100px long
- Body-center annotations may be 30-50px from the animal's centroid depending on annotator
- At 25px radius, a correct detection offset by 30px from the annotation point is counted as False Positive AND False Negative
- This artificially deflates F1, making the model appear to perform much worse than it actually does

**The Miesner 2025 thesis established 75px as the optimal matching radius for iguana data.** At this radius, any detection that clearly identifies the correct animal is counted as a TP.

### Matching Radius Selection Guide

| Animal Size (pixels) | Recommended Radius | Reasoning |
|----------------------|-------------------|-----------|
| 10-30px (small birds) | 15-25px | Small targets, tight matching |
| 30-60px (medium) | 40-60px | Allow for annotation variance |
| 60-120px (iguanas) | 75px | Body-center variance |
| > 120px (large animals) | 100-150px | Large body, variable annotation point |

### Impact on Metrics

```
matching_radius=25px:  F1=0.45, precision=0.52, recall=0.40  -- WRONG
matching_radius=50px:  F1=0.78, precision=0.82, recall=0.75  -- Better
matching_radius=75px:  F1=0.93, precision=0.95, recall=0.92  -- CORRECT
matching_radius=150px: F1=0.95, precision=0.90, recall=0.99  -- Too generous
```

Too large a radius inflates metrics by matching detections to distant ground truth points. 75px is the sweet spot for iguanas.

## Running Evaluation

```python
# Run evaluation
f1 = evaluator.evaluate(
    returns='f1_score',     # Which metric to return
    wandb_flag=False,       # Log to W&B
    viz=False,              # Save visualizations
    log_meters=True         # Print per-image metrics
)

print(f"Overall F1: {f1:.3f}")
```

### Available Return Metrics
`'recall'`, `'precision'`, `'f1_score'`, `'f2_score'`, `'f5_score'`, `'mse'`, `'mae'`, `'me'`, `'rmse'`, `'accuracy'`, `'mAP'`

## Understanding Evaluator Output

### Per-Iteration Logging
During evaluation, each image produces logged metrics:
```
n=15 tp=12 fp=2 fn=1 recall=0.92 precision=0.86 f1_score=0.89 MAE=1.00 RMSE=1.00
```

- **n**: Total matches (tp + fp + fn)
- **tp**: True Positives (correct detections within matching radius)
- **fp**: False Positives (detections with no nearby ground truth)
- **fn**: False Negatives (ground truth points with no nearby detection)
- **recall**: tp / (tp + fn) -- what fraction of real animals were detected
- **precision**: tp / (tp + fp) -- what fraction of detections are real animals
- **f1_score**: harmonic mean of precision and recall
- **MAE**: Mean Absolute Error in count (|predicted_count - true_count|)
- **RMSE**: Root Mean Squared Error in count

### Results DataFrame

After evaluation, access per-class results:
```python
results_df = evaluator.results
print(results_df.to_string())
```

Output:
```
  class   n  recall  precision  f1_score  confusion  mae    me   mse   rmse    ap
      1  850  0.940     0.928     0.934       0.02  0.45  -0.1  0.82   0.91  0.89
      2   45  0.800     0.750     0.774       0.05  0.30   0.2  0.50   0.71  0.65
 binary  895  0.934     0.920     0.927       0.03  0.42  -0.1  0.78   0.88  0.85
```

- **class**: Per-class metrics (1 = iguana, 2 = hard_negative, binary = aggregated)
- **confusion**: Inter-class confusion rate
- **ap**: Average Precision (area under precision-recall curve)
- **me**: Mean Error (signed, positive = overcount, negative = undercount)

### Detections DataFrame

Access raw detections:
```python
detections_df = evaluator.detections
print(detections_df.head())
```

## Metrics Class Internals

Source: `animaloc/eval/metrics.py`

The `Metrics` class accumulates TP/FP/FN counts across all images:

```python
Metrics(
    threshold: float,       # Matching radius (pixels for points, IoU for boxes)
    num_classes: int = 2    # Including background
)
```

### How Matching Works

For each image:
1. Get list of ground truth points and predicted points
2. Compute pairwise distances between all GT and predicted points
3. Apply Hungarian matching to find optimal assignment
4. A match is valid only if the distance is <= threshold (matching_radius)
5. Matched pairs = TP, unmatched predictions = FP, unmatched GT = FN

### Method Reference

| Method | Description |
|--------|-------------|
| `metrics.recall(class=None)` | Recall, optionally per class |
| `metrics.precision(class=None)` | Precision, optionally per class |
| `metrics.fbeta_score(class=None, beta=1)` | F-beta score (F1 when beta=1) |
| `metrics.mae(class=None)` | Mean Absolute Error in count |
| `metrics.me(class=None)` | Mean (signed) Error in count |
| `metrics.mse(class=None)` | Mean Squared Error in count |
| `metrics.rmse(class=None)` | Root Mean Squared Error in count |
| `metrics.ap(class=None)` | Average Precision |
| `metrics.accuracy()` | Classification accuracy |
| `metrics.confusion(class=None)` | Inter-class confusion rate |
| `metrics.flush()` | Reset all counters |
| `metrics.aggregate()` | Aggregate per-class into binary |
| `metrics.copy()` | Deep copy for independent accumulation |

### feed() Input Format

```python
metrics.feed(
    gt={'loc': [[x1,y1], [x2,y2], ...], 'labels': [1, 1, 2, ...]},
    preds={'loc': [[x1,y1], ...], 'labels': [1, ...], 'scores': [0.9, ...]},
    est_count=[45, 3]  # Optional: per-class count estimates
)
```

## Cross-Run Comparison

### Systematic Comparison Framework

When comparing experiments, track these variables:

| Variable | What to Record |
|----------|---------------|
| Backbone | DLA-34 / DLA-60 / ConvNext / DINOv2 |
| down_ratio | 2 / 4 / 8 |
| head_conv | 32 / 64 / 128 |
| LMDS kernel | (3,3) / (5,5) / (7,7) |
| adapt_ts | 0.3 / 0.5 / 0.7 |
| matching_radius | 25 / 50 / 75 / 100 |
| FocalLoss beta | 4 / 5 |
| weight_decay | 1.6e-4 / 3.25e-4 |
| Training data | Island, number of annotations |
| F1 (val) | Primary metric |
| F1 (test) | Final metric |

### Comparison Checklist

When comparing two runs:
- [ ] Same matching_radius for both evaluations
- [ ] Same LMDS parameters for both evaluations
- [ ] Same validation/test dataset
- [ ] Same data split (no leakage)
- [ ] Same down_ratio in evaluator as in training

### Thesis Baselines for Comparison

| Configuration | F1 (Floreana) | F1 (Fernandina) |
|--------------|---------------|-----------------|
| Optimal (DLA-34, DR=4, beta=5, mr=75) | **0.934** | **0.843** |
| Default matching radius (mr=25) | ~0.45 | ~0.35 |
| DLA-60 (larger backbone) | Lower (overfits) | Lower |
| DR=2 (higher resolution) | Lower | Lower |

## Evaluation Pitfalls

### 1. Wrong Matching Radius
The most common error. Always check `evaluator.threshold` matches your intended radius.

### 2. down_ratio Mismatch
If the evaluator uses a different down_ratio than the model, LMDS coordinates will be misaligned. The ground truth is downsampled by `DownSample(down_ratio=DR)` in the validation end_transforms. This must match.

### 3. Edge Effects
Detections near tile borders (when using a stitcher) may be artifacts of incomplete context. Consider filtering detections within `overlap/2` pixels of image edges.

### 4. up Parameter in LMDS
- When evaluating **with stitcher**: `up=False` (stitcher already concatenated and scaled)
- When evaluating **without stitcher** (on patches): `up=True` (LMDS needs to upsample clsmap)
- This is handled automatically by `HerdNetEvaluator.prepare_feeding()` which checks `self.stitcher is not None`

### 5. scale_factor in LMDS
- Without stitcher: `scale_factor` should be the ratio of heatmap to clsmap resolution (typically 8 or 16)
- With stitcher: `scale_factor=1` (stitcher already matched resolutions)

### 6. Batch Size
Evaluation should use `batch_size=1` because:
- Different images may have different numbers of annotations
- The evaluator processes metadata (image names) that don't stack well in batches
- The stitcher expects single images

## W&B Evaluation Logging

When `wandb_flag=True`, the evaluator logs:
- Per-image metrics during evaluation
- Final summary metrics to `wandb.run.summary`
- Includes: recall, precision, f1_score, f2_score, f5_score, MAE, ME, MSE, RMSE, accuracy, mAP, tp, fn, fp, avg_score, avg_dscore
