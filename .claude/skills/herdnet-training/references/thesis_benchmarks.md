# Thesis Benchmarks Reference (Miesner 2025)

## Overview

This document contains the verified experimental results from the Miesner 2025 thesis on marine iguana detection using HerdNet. These serve as the baseline for students to compare their own results against.

## Best Results

### Per-Island Performance

| Island | F1 Score | Configuration |
|--------|---------|---------------|
| **Floreana** | **0.934** | Optimal parameters (see below) |
| **Fernandina** | **0.843** | Same optimal parameters |

Fernandina has lower F1 due to harder terrain, more variable iguana density, and different background complexity.

## Optimal Parameter Set

The following parameters were found to be optimal through systematic experimentation:

### Model Parameters
| Parameter | Optimal Value | Default Value | Impact |
|-----------|--------------|---------------|--------|
| Backbone | DLA-34 (`timm/dla34`) | DLA-34 | All larger backbones overfit |
| down_ratio | 4 | 2 | Less memory, better generalization |
| head_conv | 64 | 64 (some configs: 128) | Smaller helps generalization |
| num_classes | 3 | - | background + iguana + hard_negative |
| pretrained | True | True | ImageNet pretraining essential |

### Loss Parameters
| Parameter | Optimal Value | Default Value | Impact |
|-----------|--------------|---------------|--------|
| FocalLoss alpha | 2 | 2 | Standard value |
| FocalLoss beta | **5** | 4 | Penalizes background more heavily |
| CE background weight | 0.1 | 0.1 | Low weight for dominant background |
| CE iguana weight | 4-5 | 1.0 | Compensates class imbalance |
| CE hard_neg weight | 0.1-1.0 | 1.0 | Depends on hard negative frequency |

### Training Parameters
| Parameter | Optimal Value | Default Value | Impact |
|-----------|--------------|---------------|--------|
| lr (head) | 1e-4 | 1e-3 | Lower for stability |
| backbone_lr | 1e-6 | 1e-5 | Preserve pretrained features |
| weight_decay | **3.25e-4** | 1.6e-4 | Stronger regularization |
| warmup_iters | 100 | 0 | Prevents early NaN |
| optimizer | AdamW | - | Standard choice |
| batch_size | 2-5 | - | Depends on GPU memory |

### Evaluation Parameters
| Parameter | Optimal Value | Default Value | Impact |
|-----------|--------------|---------------|--------|
| matching_radius | **75px** | 25px | CRITICAL for correct F1 |
| LMDS kernel_size | **(5, 5)** | (3, 3) | Better for iguana spacing |
| LMDS adapt_ts | **0.5** | 0.3 | Higher reduces false positives |
| LMDS neg_ts | 0.1 | 0.1 | Standard value |

## Key Experimental Findings

### 1. Body-Center vs Head Annotations
- **Body-center annotations outperform head annotations by F1 = +0.10**
- Head annotations have higher variance because the head is small and mobile
- Body center is more consistent across annotators
- Recommendation: Always annotate body center

### 2. Backbone Comparison
| Backbone | Parameters | Overfits? | Relative F1 |
|----------|-----------|-----------|-------------|
| **DLA-34** | ~15.7M | **No** | **Best** |
| DLA-60 | ~22.0M | Yes | Lower |
| DLA-102 | ~33.0M | Yes | Lower |
| DLA-169 | ~53.4M | Yes | Lower |

All larger DLA variants overfit on the iguana dataset (~2000-2500 annotations). This is a clear example of the bias-variance tradeoff: larger models have too much capacity for the available data.

### 3. Down Ratio Comparison
| down_ratio | Heatmap Size (512 input) | Memory | F1 |
|-----------|------------------------|---------|----|
| 2 | 256x256 | High | Lower |
| **4** | **128x128** | **Medium** | **Best** |
| 8 | 64x64 | Low | Not tested |

DR=4 outperforms DR=2 despite lower resolution. This may be because:
- Lower resolution provides implicit regularization
- Point annotation precision doesn't warrant pixel-level heatmaps
- Reduced memory allows larger batch sizes

### 4. Learning Curve Saturation
- Performance improves rapidly up to ~1000 annotations
- Plateau begins around **2000-2500 annotations**
- Adding more annotations beyond 2500 provides diminishing returns
- Implication: Focus on annotation quality over quantity

### 5. Cross-Island Generalization
- **Training per island produces the best results**
- Cross-island training (training on Floreana, testing on Fernandina) degrades performance
- Domain gap includes: terrain type, vegetation, lighting, iguana density patterns
- Recommendation: If deploying to a new island, collect at least some local training data

### 6. Orthomosaic Quality
- **Pix4D orthomosaics > DroneDeploy orthomosaics** (F1 delta ~0.07)
- Pix4D produces sharper, more geometrically accurate orthomosaics
- DroneDeploy orthomosaics have more stitching artifacts and blurriness
- Recommendation: Use Pix4D for photogrammetric processing when possible

### 7. Convergence Behavior
- Training typically converges at **8-11 epochs**
- Further training does not improve and may cause slight overfitting
- Early stopping with patience=10 is recommended
- Learning rate scheduling (ReduceLROnPlateau) helps in the final epochs

### 8. Matching Radius Sensitivity
| Radius (px) | F1 (approximate) | Note |
|------------|-------------------|------|
| 10 | ~0.20 | Far too strict |
| 25 (default) | ~0.45 | Still too strict |
| 50 | ~0.78 | Approaching correct |
| **75** | **~0.93** | **Optimal** |
| 100 | ~0.94 | Slightly generous |
| 150 | ~0.95 | Too generous |

The correct matching radius depends on animal size and annotation precision. For iguanas at typical GSD (0.5-1.0 cm/px), 75px captures the realistic annotation variance.

## Comparison Framework for Students

When comparing your results to these benchmarks, ensure:

1. **Same matching_radius** (75px): This is the #1 source of apparent discrepancy
2. **Same LMDS parameters**: kernel_size=(5,5), adapt_ts=0.5
3. **Same evaluation set**: Use the same validation/test split
4. **Same annotation style**: Body-center, not head
5. **Per-island evaluation**: Don't mix islands in evaluation

### Expected Student Results

| Level | Expected F1 (Floreana) | Notes |
|-------|----------------------|-------|
| First attempt | 0.40-0.65 | Common issues: wrong matching_radius, default params |
| After fixing eval params | 0.75-0.85 | Correct matching_radius and LMDS |
| Optimized | 0.88-0.93 | Tuned LR, weight_decay, backbone |
| Thesis reproduction | 0.93-0.94 | Full optimal parameter set |

### If Your Results are Much Lower

| Your F1 | Most Likely Issue |
|---------|------------------|
| ~0.0 | Wrong matching_radius or LMDS returns no detections |
| ~0.3-0.5 | matching_radius = 25px (default), switch to 75px |
| ~0.6-0.7 | Suboptimal LMDS params or wrong down_ratio in evaluator |
| ~0.7-0.8 | Missing optimal training params (weight_decay, FocalLoss beta) |
| ~0.8-0.9 | Close to optimal, fine-tune LMDS and training params |

### If Your Results are Much Higher
- Check for data leakage (patches from same orthomosaic in train and val)
- Check matching_radius (> 100px is too generous)
- Verify you're evaluating on the correct dataset
