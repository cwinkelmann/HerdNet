# Training Diagnostics Reference

## Quick Diagnosis Table

| Symptom | Likely Cause | Fix | Config Change |
|---------|-------------|-----|---------------|
| Loss NaN | LR too high | Reduce lr, add warmup | `lr: 1e-4`, `warmup_iters: 100` |
| Loss NaN at iter 1 | No warmup + high LR | Add warmup | `warmup_iters: 100` |
| Loss not decreasing | down_ratio mismatch | Check DR consistency | `${model.kwargs.down_ratio}` everywhere |
| Loss not decreasing | LR too low | Increase LR | `lr: 1e-3` |
| Loss oscillating | LR too high | Reduce LR | `lr: 1e-4` or add scheduler |
| F1 stuck at 0 | matching_radius too small | Use 75px, not 25px | `evaluator.threshold: 75` |
| F1 stuck at 0 | LMDS adapt_ts too high | Lower threshold | `adapt_ts: 0.3` |
| F1 stuck at 0 | LMDS neg_ts too high | Lower threshold | `neg_ts: 0.1` |
| F1 plateau early | LMDS kernel too small | Increase kernel | `kernel_size: [5, 5]` |
| F1 plateau early | Model capacity too low | Increase head_conv | `head_conv: 128` |
| Overfitting | Backbone too large | Use DLA-34 | `backbone: timm/dla34` |
| Overfitting | head_conv too large | Reduce to 64 | `head_conv: 64` |
| Overfitting | Insufficient regularization | Increase weight_decay | `weight_decay: 3.25e-4` |
| Overfitting | Too many epochs | Reduce or add early stopping | `epochs: 15`, `early_stopping: True` |
| Memory overflow (OOM) | DR=2 too expensive | Use DR=4 | `down_ratio: 4` |
| Memory overflow (OOM) | Batch size too large | Reduce | `batch_size: 1` |
| Memory overflow (OOM) | Backbone too large | Use DLA-34 | `backbone: timm/dla34` |
| Slow convergence | No warmup | Add warmup | `warmup_iters: 100` |
| Slow convergence | backbone_lr = head_lr | Differentiate | `backbone_lr: 1e-6`, `lr: 1e-4` |
| Slow convergence | No pretrained weights | Enable | `pretrained: True` |
| Poor cross-island | Domain gap | Train per island | Separate configs |
| RuntimeError: size mismatch | num_classes wrong | Fix count | Include background in count |
| RuntimeError: weight tensor size | CE weight list wrong | Fix length | Length = num_classes |
| Blank stitcher output | up=True with reduction='mean' | Set up=False | `stitcher.kwargs.up: False` |
| Double-counting at borders | Small overlap or sum reduction | Fix stitcher | `overlap: 120`, `reduction: mean` |

## Detailed Diagnosis Flows

### Flow 1: Loss is NaN

```
Loss is NaN
    |
    +-- At which iteration?
    |     |
    |     +-- Iteration 1-5: Learning rate too high
    |     |   Fix: lr: 1e-4, warmup_iters: 100
    |     |
    |     +-- After 50+ iterations: Numerical instability
    |     |   Fix: Add gradient clipping (max_norm=1.0)
    |     |
    |     +-- Random iteration: Data issue
    |         Check: Are there corrupted images in the dataset?
    |         Check: Are FIDT targets all valid? (no NaN/Inf)
    |
    +-- Only in one loss?
          |
          +-- FocalLoss NaN: Check alpha/beta params, check heatmap output range
          +-- CE Loss NaN: Check class weights, check if any class has 0 samples
```

### Flow 2: F1 Score is 0

```
F1 = 0 for all epochs
    |
    +-- Is the loss decreasing?
    |     |
    |     +-- No: Model isn't learning at all
    |     |   Check: data loading, transforms, annotations
    |     |
    |     +-- Yes: Evaluation issue (not training issue!)
    |           |
    |           +-- Check matching_radius
    |           |   Default 25px is almost always too small
    |           |   Try: threshold: 75 in evaluator config
    |           |
    |           +-- Check LMDS parameters
    |           |   adapt_ts too high -> no peaks survive
    |           |   kernel_size too small -> wrong peaks selected
    |           |   Try: adapt_ts: 0.3, kernel_size: [5,5]
    |           |
    |           +-- Check down_ratio in evaluator
    |               Must match model's down_ratio
    |               DownSample in val end_transforms must use same DR
```

### Flow 3: Overfitting

```
Training loss decreasing, validation F1 peaks then drops
    |
    +-- What backbone?
    |     |
    |     +-- DLA-60/102/169: Too large for dataset size
    |     |   Fix: Switch to DLA-34
    |     |
    |     +-- DLA-34: Still overfitting
    |           |
    |           +-- What head_conv?
    |           |     |
    |           |     +-- 128 or higher: Reduce to 64
    |           |
    |           +-- What weight_decay?
    |           |     |
    |           |     +-- < 3e-4: Increase to 3.25e-4
    |           |
    |           +-- How many augmentations?
    |           |     |
    |           |     +-- Only Normalize: Add ObjectAwareRandomCrop, flips
    |           |
    |           +-- How much data?
    |                 |
    |                 +-- < 1000 annotations: May need more data
    |                 +-- > 2500: Should be sufficient (learning curve plateau)
```

### Flow 4: Memory Issues

```
CUDA Out of Memory
    |
    +-- Current down_ratio?
    |     |
    |     +-- DR=2: Switch to DR=4 (halves memory for heatmap)
    |     +-- DR=4 already: Reduce batch_size
    |
    +-- Current batch_size?
    |     |
    |     +-- > 2: Reduce to 1 or 2
    |     +-- Already 1: Reduce image size or backbone
    |
    +-- Current backbone?
    |     |
    |     +-- DLA-60+: Switch to DLA-34
    |     +-- DINOv2: Enable gradient checkpointing
    |
    +-- Still OOM with BS=1, DR=4, DLA-34?
          |
          +-- Check image size: 512x512 should fit in 4GB
          +-- Check num_workers: High values use more CPU memory
          +-- Check for memory leaks: Are you accumulating tensors?
```

## Learning Rate Guidance

### Symptoms by LR Range

| LR | Typical Behavior |
|-----|-----------------|
| 1e-2 | NaN loss within first epoch |
| 1e-3 | May work but unstable, oscillating loss |
| **1e-4** | **Optimal for head, stable training** |
| 1e-5 | Slow convergence, may not reach optimum |
| 1e-6 | Good for backbone (pretrained features), too slow for heads |

### Differential Learning Rate Guide

```yaml
# Recommended setup:
backbone_lr: 1e-6    # Barely touch pretrained features
head_lr: 1e-4        # Heads need to learn from scratch
lr: 1e-4             # Global (applied to non-backbone params)
```

Why differential LR works:
- Backbone has already learned useful feature extraction from ImageNet
- Small backbone_lr preserves these features while allowing fine-tuning
- Heads are randomly initialized and need larger LR to learn

## Epoch Count Guidance

| Dataset Size | Expected Convergence | Max Epochs |
|-------------|---------------------|------------|
| < 500 annotations | 5-8 epochs | 15 |
| 500-2000 | 8-11 epochs | 20 |
| 2000-5000 | 10-15 epochs | 25 |
| > 5000 | 15-20 epochs | 30 |

Convergence is defined as: validation F1 has not improved by > 0.005 for 3+ epochs.

## Monitoring Checklist

During training, monitor these signals:

### Healthy Training
- [ ] Total loss decreasing smoothly
- [ ] FocalLoss and CE loss both decreasing
- [ ] F1 increasing on validation set
- [ ] Recall and precision both increasing (not just one)
- [ ] No NaN values in any metric
- [ ] Training and validation metrics are close (no large gap)

### Warning Signs
- [ ] Loss decreasing but F1 not improving -> evaluation parameter issue
- [ ] Training F1 >> validation F1 -> overfitting
- [ ] Precision >> recall -> too conservative (high adapt_ts or few predictions)
- [ ] Recall >> precision -> too many false positives (low adapt_ts or wrong threshold)
- [ ] Loss plateaus very early -> LR too low or wrong FIDT parameters
- [ ] MAE/RMSE not decreasing -> counting error not improving

## W&B Dashboard Setup

Recommended W&B panel layout for monitoring:

1. **Loss panel**: total_loss, focal_loss, ce_loss (per epoch)
2. **Detection panel**: f1_score, recall, precision (per epoch)
3. **Counting panel**: MAE, RMSE (per epoch)
4. **Learning rate panel**: lr (per iteration, should show warmup ramp)
5. **Confusion panel**: tp, fp, fn (per epoch)

## Post-Training Validation

After training completes, before deploying the model:

1. **Check best epoch**: Is it near the end (good) or early (overfitting)?
2. **Run evaluation on test set**: Use held-out data never seen during training
3. **Check per-class metrics**: Is one class much worse than others?
4. **Visualize predictions**: Overlay detections on sample images
5. **Compare to thesis benchmarks**: F1=0.934 (Floreana), F1=0.843 (Fernandina)
6. **Test on different orthomosaic quality**: Pix4D vs DroneDeploy (~0.07 F1 delta)
