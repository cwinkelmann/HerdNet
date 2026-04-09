# Training Round 4: Pushing Past F1=0.95

## Current Best

| Model | F1 | Precision | Recall | Config |
|-------|-----|-----------|--------|--------|
| ConvNeXt Tiny | 0.952 | 0.979 | 0.926 | fmo03_convnext_opt |
| Hybrid (CNN+Transformer) | 0.927 | 0.962 | 0.895 | fmo03_hybrid |
| DLA34 | 0.908 | 0.935 | 0.882 | fmo03_dla34_opt |

Error analysis shows ~12 FPs are real iguanas (missing GT) → true F1 ≈ 0.96.
47 FNs break down as: 25 duplicates in overlapping crops, 14 below-threshold, 8 at edges.

## Bottleneck Analysis

| Bottleneck | Impact | Fix |
|-----------|--------|-----|
| Small training set (1003 crops) | Model hasn't seen enough variation | Add iguana_hn data (6753 annotations) |
| Single site (FMO03/FMO05 only) | Poor generalization | Add Floreana FMO01/FMO06 from iguana_hn |
| Annotation noise (~20 wrong labels) | 3% of error is GT, not model | Can't fix without correction loop |
| Conservative threshold (adapt_ts=0.6) | Misses 14 real detections | Already identified optimal: 0.50 |
| Limited augmentation diversity | 1x multiplier per epoch | Use augmentation_multiplier=3 |

## Experiments

### A: Combined Dataset (highest expected impact)

Train ConvNeXt on FMO03 crops + iguana_hn full-res images together.

The iguana_hn dataset has 57 full-resolution drone images with 6753 iguana annotations (3.5x more data). These are full-resolution images requiring `ObjectAwareRandomCrop` to extract 512x512 training patches.

```yaml
# configs/demo/datasets/combined_local.yaml
# Merges FMO03 crops + iguana_hn full images
# Uses ObjectAwareRandomCrop for full images, standard augmentation for crops
```

- Model: ConvNeXt Tiny, pretrained, down_ratio=4
- Epochs: 40, lr: 8e-5, batch: 4, warmup: 500
- Heavy augmentation
- Val: FMO03 val only (fair comparison)

**Why this helps:** More diverse training data → better feature learning for camouflaged iguanas. Different sites (Fernandina vs Floreana) expose the model to more background variation.

### B: Fine-tune Combined → FMO03

Two-phase curriculum learning:
1. Phase 1 (20 epochs, lr=1e-4): Train on combined dataset — learn general "iguana from above" features
2. Phase 2 (20 epochs, lr=2e-5): Fine-tune on FMO03 only — specialize to target site

**Why this helps:** Phase 1 builds robust features from diverse data. Phase 2 adapts to the specific site characteristics without forgetting.

### C: Augmentation Multiplier

Same best ConvNeXt config but with `augmentation_multiplier: 3` — each image appears 3x per epoch with different augmentations. Effectively triples dataset size without new data.

- Model: ConvNeXt Tiny (same as R2 opt)
- augmentation_multiplier: 3
- Epochs: 15 (= 45 effective epochs)
- Otherwise identical to fmo03_convnext_opt

**Why this helps:** More augmentation diversity per epoch means the model sees more varied views of each iguana before the LR scheduler reduces the learning rate.

### D: Hybrid with Longer Training on Combined Data

The Hybrid model (CNN stem + 4 Transformer blocks) reached F1=0.927 in 30 epochs. The Transformer blocks started from random init and may need more time, especially with more data.

- Model: HerdNetHybrid, ConvNeXt Tiny stem
- Train: Combined dataset
- Epochs: 60, lr: 5e-5, warmup: 800
- valid_freq: 3 (faster training)

**Why this helps:** Transformer blocks need more data and epochs than CNNs. Combined dataset provides 3.5x more data; 60 epochs gives 2x more training time.

## Execution

```bash
# 1. Create configs (already prepared by Claude)
# 2. Sequential execution (~2 hours total with AMP)

/train fmo03_combined_convnext    # Exp A: ~30 min
/train fmo03_augmult_convnext     # Exp C: ~20 min
/train fmo03_combined_hybrid      # Exp D: ~45 min
/train fmo03_finetune_convnext    # Exp B: ~20 min (if A improves)

# 3. Evaluate all with hparam search
python tools/inference_hparam_search.py
```

## Success Criteria

- Beat current best F1=0.952 on FMO03 val (noisy GT)
- Or beat corrected F1=0.961 (after accounting for GT errors)
- Recall ≥ 0.95 at precision ≥ 0.95
