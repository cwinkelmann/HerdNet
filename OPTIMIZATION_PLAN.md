# Model Performance Optimization Plan

## Context

Two-part plan: (1) Build a benchmark framework to measure model performance with real training data, and (2) implement optimizations to the three HerdNet architectures, validating each change with the benchmark.

The goal is to know **which optimization helped most** — every change is benchmarked and ranked.

---

## Part 1: Benchmark Framework

### Goal
A standalone benchmark script that trains a single model for 20-30 epochs on configurable data, records all metrics to a JSON file, and provides a comparison script to rank optimizations by impact.

### Design

#### 1. Benchmark runner — `tools/benchmark.py`

A Hydra-based script (reuses the existing config system) that:
- Accepts `--model`, `--data-dir`, `--data-csv`, `--epochs` via CLI or Hydra overrides
- Runs full training using `animaloc.utils.train.main(cfg)` (reuse existing pipeline)
- After training, extracts metrics from the evaluator and saves a structured JSON result
- Tags each run with: git commit hash, git branch, timestamp, model name, run label

```bash
python tools/benchmark.py \
  --config-name dla34_delplanque \
  benchmark.label="baseline_dla34" \
  datasets.train.csv_file=/path/to/train.csv \
  datasets.train.root_dir=/path/to/images/ \
  datasets.validate.csv_file=/path/to/val.csv \
  datasets.validate.root_dir=/path/to/val_images/ \
  benchmark.epochs=25
```

**Key**: This is NOT a new training pipeline — it wraps the existing `main(cfg)` and adds metric collection + JSON output on top.

#### 2. Result format — `benchmark_results/<label>_<timestamp>.json`

```json
{
  "run_id": "baseline_dla34_20260325_143022",
  "label": "baseline_dla34",
  "model": "HerdNet",
  "git_commit": "7eaf334",
  "git_branch": "dinov3",
  "timestamp": "2026-03-25T14:30:22",
  "config_overrides": ["..."],
  "epochs": 25,
  "best_epoch": 18,
  "metrics": {
    "f1_score": 0.742,
    "recall": 0.813,
    "precision": 0.682,
    "f2_score": 0.785,
    "mae": 2.31,
    "rmse": 3.12,
    "mAP": 0.691
  },
  "training_time_seconds": 3842,
  "final_train_loss": 0.0234,
  "best_val_loss": 0.0312
}
```

#### 3. Comparison script — `tools/benchmark_compare.py`

Reads all JSON files from `benchmark_results/` and prints a ranked comparison table.

**Basic usage:**
```bash
python tools/benchmark_compare.py
```
```
Run Label                  Model          Commit  Epoch  F1     Recall  Prec   MAE
─────────────────────────────────────────────────────────────────────────────────────
baseline_dla34             HerdNet        7eaf334  18    0.742  0.813   0.682  2.31
fix_fpn_weights            HerdNetConvNeXt a1b2c3d  21    0.763  0.835   0.701  1.98
multiscale_aggregation     HerdNetConvNeXt d4e5f6g  19    0.778  0.851   0.718  1.82
```

**With baseline comparison (the key feature):**
```bash
python tools/benchmark_compare.py --baseline baseline_convnext
```
```
Baseline: baseline_convnext (F1=0.742, Recall=0.813, Prec=0.682)

Rank  Run Label                  F1     ΔF1     Recall  ΔRecall  Prec    ΔPrec   MAE   ΔMAE
──────────────────────────────────────────────────────────────────────────────────────────────
 1    multiscale_aggregation     0.778  +0.036  0.851   +0.038   0.718   +0.036  1.82  -0.49
 2    fix_fpn_weights            0.763  +0.021  0.835   +0.022   0.701   +0.019  1.98  -0.33
 3    gelu_init_fix              0.748  +0.006  0.820   +0.007   0.688   +0.006  2.25  -0.06
──────────────────────────────────────────────────────────────────────────────────────────────
Best single optimization: multiscale_aggregation (+3.6% F1, +3.8% Recall)
```

**CLI options:**
- `--sort-by f1_score` (default) or any metric key
- `--model HerdNet` — filter by model name
- `--baseline <label>` — show deltas from a baseline run, rank by improvement
- `--results-dir <path>` — custom results directory (default: `benchmark_results/`)

#### 4. How it hooks into the existing system

The benchmark reuses these existing components (no duplication):
- **Training**: `animaloc.utils.train.main(cfg)` — the exact same entry point as `tools/train.py`
- **Evaluation**: `HerdNetEvaluator` with `HerdNetStitcher` — produces F1/recall/precision
- **Configs**: Hydra configs from `configs/demo/` with overrides for data paths and epochs
- **Metrics**: `animaloc.eval.metrics.PointsMetrics` — existing metric computation

The only new logic is:
- Capturing the evaluator's final metrics after `main()` returns
- Writing the JSON result file
- The comparison table script

#### 5. Benchmark config — `configs/demo/benchmark_defaults.yaml`

A new Hydra config that sets sensible benchmark defaults:
```yaml
defaults:
  - /training_settings: evaluator

benchmark:
  label: null           # Required: name this run
  epochs: 25            # Override training epochs
  results_dir: benchmark_results

training_settings:
  epochs: ${benchmark.epochs}
  batch_size: 8
  warmup_iters: 500
  auto_lr:
    mode: max
    patience: 10

wandb_flag: False        # Benchmarks are local-only by default
```

#### 6. Extracting metrics from the trainer

**Problem**: Currently `main()` returns the model path but not the evaluator metrics.

**Fix**: Modify `animaloc/utils/train.py:main()` to also return the evaluator's final metrics dict. This is a small change — the Trainer already stores `self.best_val` and the evaluator stores `self.metrics`. We just need to pass them back.

### Files to create/modify for benchmark

| File | Action | Purpose |
|------|--------|---------|
| `tools/benchmark.py` | **Create** | Benchmark runner script |
| `tools/benchmark_compare.py` | **Create** | Results comparison table with ranking |
| `configs/demo/benchmark_defaults.yaml` | **Create** | Default benchmark config |
| `animaloc/utils/train.py` | **Modify** | Return metrics dict from `main()` |
| `benchmark_results/.gitkeep` | **Create** | Results directory |

---

## Part 2: Model Review Findings

### HerdNet (Original) — `animaloc/models/herdnet.py`
- **Backbone**: Custom DLA-34 (~15M params), ImageNet pretrained
- **Neck**: DLAUp (deformable upsampling)
- **Heads**: Two simple 2-layer conv heads (loc_head: Sigmoid heatmap, cls_head: logits)
- **Strengths**: Lightweight, proven, fast inference (~20ms)
- **Weaknesses**:
  - No BatchNorm in heads → training instability
  - No attention mechanisms → misses subtle/camouflaged objects
  - No dropout → overfitting risk on small datasets
  - Bottleneck is a bare 1x1 conv with no nonlinearity
  - Classification operates only on 16x16 coarsest features

### HerdNetTimmDLA — `animaloc/models/herdnet_timm_dla.py`
- Essentially the same architecture as original, but with timm backbone loading
- Same head design, same DLAUp neck
- **No architectural improvements over original** — purely a refactor for flexibility

### HerdNetConvNeXt — `animaloc/models/herdnet_timm_convnext.py`
- **Backbone**: ConvNeXt-tiny/small/base (28M–88M), ImageNet-22k pretrained
- **Neck**: EnhancedFPN with learnable fusion weights
- **Heads**: DenseMultiScaleHead with CBAM attention + temperature calibration
- **Strengths**: Modern backbone, multi-scale detection, attention, recall-optimized init
- **Weaknesses/Bugs**:
  1. **FPN `fusion_weights` are dead code** — declared on line 72 but never used in `forward()` (lines 86–101)
  2. **Detection head uses only `fpn_features[0]`** (finest scale) — ignores the other 3 FPN levels
  3. **Kaiming init with `nonlinearity='relu'`** but activation is GELU — mismatch
  4. Temperature learned jointly can destabilize early training
  5. No cosine LR scheduler support in trainer — only MultiStepLR or ReduceLROnPlateau

---

## Part 3: Optimization Suggestions

### Priority 1: Fix Bugs / Low-Hanging Fruit (ConvNeXt)

#### 1a. Wire up FPN fusion weights
**File**: `animaloc/models/herdnet_timm_convnext.py`, `EnhancedFPN.forward()`

The `self.fusion_weights` parameter is declared but never applied. The top-down fusion currently uses naive addition. Wire the weights into the fusion:

```python
# Current (line 98): laterals[i-1] = laterals[i-1] + upsampled
# Fix: Apply learnable weight to the upsampled contribution
weights = F.softmax(self.fusion_weights, dim=0)
laterals[i-1] = weights[i-1] * laterals[i-1] + weights[i] * upsampled
```

#### 1b. Fix weight initialization to match GELU
**File**: `animaloc/models/herdnet_timm_convnext.py`, `DenseMultiScaleHead._init_weights()`

Replace `nonlinearity='relu'` with a fan_in/GELU-appropriate gain:
```python
nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
```

#### 1c. Aggregate multi-scale FPN features for detection
**File**: `animaloc/models/herdnet_timm_convnext.py`, `HerdNetConvNeXt.forward()`

Currently only `fpn_features[0]` (128x128) feeds the detection head. Upsample and fuse all FPN levels to give the head richer context:

```python
# Upsample all FPN features to finest resolution and sum
target_size = fpn_features[0].shape[2:]
fused = fpn_features[0]
for feat in fpn_features[1:]:
    fused = fused + F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
det_logits = self.detection_head(fused)
```

---

### Priority 2: Training Pipeline Improvements

#### 2a. Add CosineAnnealingWarmRestarts scheduler
**File**: `animaloc/train/trainers.py`, `_lr_scheduler()`

The current trainer only supports MultiStepLR and ReduceLROnPlateau. Cosine annealing with warm restarts consistently outperforms step decay for CNN training. Add as a new `auto_lr` option:

```python
elif self.auto_lr == 'cosine':
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        self.optimizer, T_0=10, T_mult=2)
```

#### 2b. Support differential learning rates
**File**: `animaloc/utils/train.py` (optimizer setup)

ConvNeXt already has `get_optimizer_params()` but the training entry point doesn't use it. Wire it so backbone gets 10x lower LR than heads:
- Backbone LR: `lr * 0.1`
- FPN/heads LR: `lr`

#### 2c. Two-phase training (freeze → unfreeze)
**File**: `animaloc/utils/train.py`

Add config support for a two-phase training schedule:
- Phase 1: Freeze backbone, train heads for N epochs
- Phase 2: Unfreeze backbone with differential LR for remaining epochs

---

### Priority 3: Architecture Improvements

#### 3a. BiFPN instead of top-down-only FPN
**File**: New or modify `EnhancedFPN` in `herdnet_timm_convnext.py`

The current FPN is top-down only. BiFPN (bi-directional) adds a bottom-up path after the top-down path, improving feature fusion — proven in EfficientDet. This is a moderate-effort change: add a second pass bottom-up with its own learnable weights.

#### 3b. Add auxiliary loss on intermediate FPN scales
**File**: `herdnet_timm_convnext.py` + loss config

Add lightweight 1x1 conv heads on FPN levels 1–3, supervised with downscaled GT heatmaps during training. This provides deep supervision and is discarded at inference. Typically gives +1-2% F1.

#### 3c. Improve DLA heads (for HerdNet/TimmDLA)
**File**: `herdnet.py` and `herdnet_timm_dla.py`

Add BatchNorm + Dropout to the loc/cls heads:
```python
self.loc_head = nn.Sequential(
    nn.Conv2d(channels[self.first_level], head_conv, 3, padding=1, bias=False),
    nn.BatchNorm2d(head_conv),
    nn.ReLU(inplace=True),
    nn.Dropout2d(0.1),
    nn.Conv2d(head_conv, 1, 1),
    nn.Sigmoid()
)
```

---

### Priority 4: Loss Function Improvements

#### 4a. Use HerdNetLoss (Focal + Dice) as default
The combined focal+dice loss outperforms standalone focal loss for heatmap regression. Dice provides global shape supervision that focal alone misses. Ensure this is the default for all model configs.

#### 4b. Tune OHEM ratio
The current `OHEMFocalLoss` uses `top_k_percent=0.2` (20%). For sparse annotations (few iguanas per image), increasing to 0.3–0.4 can improve recall by keeping more informative negative examples during training.

#### 4c. Consider DensityAwareHerdNetLoss for mixed scenes
For datasets with both isolated iguanas and dense colonies, the density-aware loss with `inverse_sqrt` scaling and `dice_weight=0.5` should give better balanced performance.

---

### Priority 5: Data & Augmentation

#### 5a. Stronger augmentations
Ensure the augmentation pipeline includes:
- RandomRotate90 + HorizontalFlip + VerticalFlip (already likely)
- ColorJitter / RandomBrightnessContrast (drone lighting varies)
- RandomScale (0.8–1.2) for scale invariance
- CoarseDropout / GridDropout (occlusion robustness)

#### 5b. Test-time augmentation (TTA)
At inference, average predictions from the original + horizontally flipped input. Typically gives +1-2% F1 for free (2x inference cost).

---

## Part 4: Implementation Order

### Step 0: Build benchmark framework (do this first)

| File | Action | Purpose |
|------|--------|---------|
| `tools/benchmark.py` | **Create** | Benchmark runner script |
| `tools/benchmark_compare.py` | **Create** | Results comparison table with ranking |
| `configs/demo/benchmark_defaults.yaml` | **Create** | Default benchmark config |
| `animaloc/utils/train.py` | **Modify** | Return metrics dict from `main()` |
| `benchmark_results/.gitkeep` | **Create** | Results directory |

### Step 1: Run baselines (no code changes)

```bash
# Baseline for each model
python tools/benchmark.py --config-name dla34_delplanque \
  benchmark.label="baseline_dla34" \
  datasets.train.csv_file=<path> datasets.train.root_dir=<path> \
  datasets.validate.csv_file=<path> datasets.validate.root_dir=<path>

python tools/benchmark.py --config-name dla34_timm \
  benchmark.label="baseline_timm_dla34" ...

python tools/benchmark.py --config-name convnext_camouflaged \
  benchmark.label="baseline_convnext" ...
```

### Steps 2–10: Implement optimizations one at a time

| Step | Change | Files to Modify | Benchmark Label |
|------|--------|-----------------|-----------------|
| 2 | Fix FPN fusion weights (1a) | `herdnet_timm_convnext.py` | `fix_fpn_weights` |
| 3 | Multi-scale FPN aggregation (1c) | `herdnet_timm_convnext.py` | `multiscale_aggregation` |
| 4 | Fix GELU init (1b) | `herdnet_timm_convnext.py` | `gelu_init_fix` |
| 5 | BatchNorm/Dropout on DLA heads (3c) | `herdnet.py`, `herdnet_timm_dla.py` | `dla_batchnorm` |
| 6 | CosineAnnealing scheduler (2a) | `trainers.py` | `cosine_lr` |
| 7 | Differential LR (2b) | `train.py` | `differential_lr` |
| 8 | BiFPN upgrade (3a) | `herdnet_timm_convnext.py` | `bifpn` |
| 9 | Auxiliary losses (3b) | `herdnet_timm_convnext.py` + loss config | `aux_losses` |
| 10 | Two-phase training (2c) | `train.py` | `two_phase` |

### Per-step workflow

```bash
# 1. Make the code change

# 2. Quick smoke test
pytest tests/test_train.py -v

# 3. Full benchmark run (25 epochs on real data)
python tools/benchmark.py --config-name <config> \
  benchmark.label="<optimization_name>" \
  datasets.train.csv_file=<path> datasets.train.root_dir=<path> \
  datasets.validate.csv_file=<path> datasets.validate.root_dir=<path>

# 4. Compare to baseline — see which optimization helped most
python tools/benchmark_compare.py --baseline baseline_convnext
```

---

## Expected Impact Summary

| Step | Change | Effort | Expected Impact |
|------|--------|--------|-----------------|
| 2 | Fix FPN fusion weights bug (1a) | Small | +0.5-1% F1 |
| 3 | Multi-scale FPN aggregation (1c) | Small | +1-2% recall |
| 4 | Fix GELU init mismatch (1b) | Tiny | +0.5% convergence |
| 5 | BatchNorm/Dropout DLA heads (3c) | Small | +1-2% F1 on DLA models |
| 6 | CosineAnnealing scheduler (2a) | Medium | +1-2% F1 |
| 7 | Differential LR (2b) | Medium | +1-3% F1 |
| 8 | BiFPN upgrade (3a) | Medium | +1-3% F1 |
| 9 | Auxiliary losses (3b) | Medium | +1-2% F1 |
| 10 | Two-phase training (2c) | Medium | +2-4% F1 |

---

## Key Files Reference

| File | Role |
|------|------|
| `animaloc/models/herdnet.py` | Original HerdNet (DLA-34) |
| `animaloc/models/herdnet_timm_dla.py` | Timm DLA variant |
| `animaloc/models/herdnet_timm_convnext.py` | ConvNeXt variant (most changes here) |
| `animaloc/train/trainers.py` | Trainer class (scheduler, metrics access) |
| `animaloc/utils/train.py` | Training entry point (optimizer, metric return) |
| `animaloc/train/losses/focal.py` | Focal, HerdNet, OHEM, DensityAware losses |
| `animaloc/eval/evaluators.py` | HerdNetEvaluator (F1/recall/precision) |
| `animaloc/eval/metrics.py` | PointsMetrics (TP/FP/FN accumulation) |
| `configs/demo/*.yaml` | Hydra configs (model, loss, dataset, training) |
| `tests/test_train.py` | Existing smoke tests (2 epochs) |
| `tests/conftest.py` | Test fixtures, HuggingFace data download |