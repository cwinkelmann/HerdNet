# HerdNet Evaluation Report

## Experiment Metadata

| Property | Value |
|----------|-------|
| **Date** | YYYY-MM-DD |
| **Experiment Name** | _e.g., floreana_optimal_v3_ |
| **Model** | HerdNetTimmDLA |
| **Backbone** | DLA-34 (`timm/dla34`) |
| **down_ratio** | 4 |
| **head_conv** | 64 |
| **num_classes** | 3 (background + iguana + hard_negative) |
| **Checkpoint** | _path/to/best_model.pth_ |
| **Training Epochs** | _e.g., 15 (best at epoch 11)_ |
| **Training Set** | _e.g., Floreana, 1850 annotations, 312 images_ |
| **Validation Set** | _e.g., Floreana, 420 annotations, 78 images_ |
| **Test Set** | _e.g., Floreana, 380 annotations, 65 images_ |

### Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate (head) | 1e-4 |
| Learning rate (backbone) | 1e-6 |
| Weight decay | 3.25e-4 |
| Warmup iterations | 100 |
| Optimizer | AdamW |
| Batch size | 2 |
| FocalLoss beta | 5 |
| CE weights | [0.1, 4.0, 1.0] |

### Evaluation Parameters

| Parameter | Value |
|-----------|-------|
| Matching radius | 75 px |
| LMDS kernel_size | (5, 5) |
| LMDS adapt_ts | 0.5 |
| LMDS neg_ts | 0.1 |
| Stitcher overlap | 120 px |
| Stitcher reduction | mean |

---

## Per-Class Results

| Class | N (GT) | TP | FP | FN | Precision | Recall | F1 | MAE | RMSE | AP |
|-------|--------|-----|-----|-----|-----------|--------|-----|-----|------|-----|
| iguana (1) | _NNN_ | _NNN_ | _NNN_ | _NNN_ | _0.XXX_ | _0.XXX_ | _0.XXX_ | _X.XX_ | _X.XX_ | _0.XXX_ |
| hard_neg (2) | _NNN_ | _NNN_ | _NNN_ | _NNN_ | _0.XXX_ | _0.XXX_ | _0.XXX_ | _X.XX_ | _X.XX_ | _0.XXX_ |
| **binary (agg)** | **_NNN_** | **_NNN_** | **_NNN_** | **_NNN_** | **_0.XXX_** | **_0.XXX_** | **_0.XXX_** | **_X.XX_** | **_X.XX_** | **_0.XXX_** |

---

## Comparison to Thesis Benchmarks (Miesner 2025)

| Metric | This Experiment | Thesis (Floreana) | Thesis (Fernandina) | Delta |
|--------|----------------|-------------------|---------------------|-------|
| F1 | _0.XXX_ | 0.934 | 0.843 | _+/- X.XXX_ |
| Precision | _0.XXX_ | -- | -- | -- |
| Recall | _0.XXX_ | -- | -- | -- |
| MAE | _X.XX_ | -- | -- | -- |

### Interpretation

_Describe how your results compare to the thesis benchmarks. If lower, hypothesize why (e.g., different data split, fewer annotations, different island). If higher, verify there's no data leakage._

---

## Confusion Analysis

### Error Types

| Error Type | Count | Example |
|-----------|-------|---------|
| True Positive (correct detection) | _NNN_ | _Detection within 75px of GT_ |
| False Positive (phantom detection) | _NNN_ | _Detection with no nearby GT_ |
| False Negative (missed animal) | _NNN_ | _GT with no nearby detection_ |
| Class Confusion (iguana <-> hard_neg) | _NNN_ | _Right location, wrong class_ |

### FP Analysis
_What are the false positives? Common sources:_
- Rocks resembling iguanas
- Shadows
- Stitching artifacts at tile borders
- Other wildlife

### FN Analysis
_What are the false negatives? Common sources:_
- Partially occluded iguanas
- Very small / distant iguanas
- Unusual pose or color
- Dense aggregations

---

## Parameter Sensitivity Analysis

### Matching Radius Sweep

_If you ran a matching radius sweep, report results here:_

| Radius (px) | F1 | Precision | Recall |
|------------|-----|-----------|--------|
| 25 | _0.XXX_ | _0.XXX_ | _0.XXX_ |
| 50 | _0.XXX_ | _0.XXX_ | _0.XXX_ |
| 75 | _0.XXX_ | _0.XXX_ | _0.XXX_ |
| 100 | _0.XXX_ | _0.XXX_ | _0.XXX_ |

### LMDS Parameter Sweep

_If you swept LMDS parameters:_

| kernel_size | adapt_ts | F1 | Precision | Recall |
|------------|----------|-----|-----------|--------|
| (3,3) | 0.3 | _0.XXX_ | _0.XXX_ | _0.XXX_ |
| (3,3) | 0.5 | _0.XXX_ | _0.XXX_ | _0.XXX_ |
| (5,5) | 0.3 | _0.XXX_ | _0.XXX_ | _0.XXX_ |
| (5,5) | 0.5 | _0.XXX_ | _0.XXX_ | _0.XXX_ |
| (7,7) | 0.5 | _0.XXX_ | _0.XXX_ | _0.XXX_ |

---

## Training Curves

_Include or reference W&B plots for:_

1. **Loss curve**: total_loss vs epoch (train and val)
2. **F1 curve**: f1_score vs epoch (validation)
3. **Precision/Recall curves**: precision and recall vs epoch
4. **Learning rate**: lr vs iteration (should show warmup ramp)

### Key Observations
- _At which epoch did the model converge?_
- _Was there any overfitting (train loss dropping while val F1 drops)?_
- _Did learning rate scheduling activate?_

---

## Qualitative Examples

### Good Detections
_Include 2-3 example images showing correct detections. Note what the model handles well._

### Failure Cases
_Include 2-3 example images showing errors. Categorize the error type._

---

## Recommendations for Improvement

Based on this evaluation:

1. **If F1 < 0.85**: _Consider [specific improvement]_
2. **If precision << recall**: _The model over-detects. Try increasing adapt_ts or FocalLoss beta._
3. **If recall << precision**: _The model under-detects. Try decreasing adapt_ts or increasing iguana class weight._
4. **If MAE > 2.0**: _Counting error is high. Check for systematic over/under-counting._

### Next Steps
- [ ] _Action item 1_
- [ ] _Action item 2_
- [ ] _Action item 3_

---

## Reproducibility Notes

### Environment
- Python version: _X.X_
- PyTorch version: _X.X_
- CUDA version: _X.X_
- GPU: _e.g., NVIDIA A100 40GB_
- animaloc version: _X.X_

### Config Files
_List or attach the exact Hydra config files used for this experiment._

### Random Seed
- Seed: _42_
- Deterministic: _Yes/No_

---

_Report generated on: YYYY-MM-DD_
_Model evaluated by: [Name]_
