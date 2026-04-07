# Hydra Config Agent

## Role
Construct valid 4-level Hydra YAML config trees with correct interpolation dependencies. Validate configs for common errors before training. Explain the config system and override syntax.

## Config Tree Structure

HerdNet uses [Hydra](https://hydra.cc/) for hierarchical configuration management. The config is organized as a 4-level tree:

```
configs/
    main.yaml                           # Top-level: defaults, wandb settings, overrides
    model/
        HerdNetTimmDLA34.yaml           # Model architecture config
    datasets/
        floreana_dataset.yaml           # Dataset paths, transforms, class definitions
    losses/
        herdnet_iguana.yaml             # Loss functions and weights
    training_settings/
        evaluator.yaml                  # Training params, optimizer, evaluator, stitcher
```

## Main Config (top level)

```yaml
# main.yaml
wandb_notes: |
  Description of this experiment

defaults:
  - /losses: herdnet_iguana
  - /datasets: floreana_dataset
  - /training_settings: evaluator
  - /model: HerdNetTimmDLA34
  - _self_                              # Include this config's own content last

wandb_flag: True
wandb_project: 'herdnet'
wandb_entity: 'your_entity'
wandb_run: 'experiment_name'
wandb_tags: ["dla34", "floreana"]
seed: 1
device_name: null                       # null = auto-select least occupied GPU

# Override values from sub-configs:
training_settings:
  lr: 1.0e-4
  epochs: 20
  batch_size: 2
  evaluator:
    kwargs:
      lmds_kwargs:
        adapt_ts: 0.5

model:
  load_from: null                       # path to pretrained checkpoint for fine-tuning
  kwargs:
    down_ratio: 4
    head_conv: 64

hydra:
  run:
    dir: ./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

### defaults Section
The `defaults` list tells Hydra which sub-configs to load. The leading `/` means absolute path from the config root. The `_self_` entry means this config's own values are applied last (can override sub-configs).

## Model Config

```yaml
# model/HerdNetTimmDLA34.yaml
name: 'HerdNetTimmDLA'
from_torchvision: False
load_from: null                         # Path to pretrained .pth checkpoint
resume_from: null                       # Path to resume training (includes optimizer state)
kwargs:
  backbone: 'timm/dla34'
  pretrained: True                      # Use ImageNet pretrained weights
  down_ratio: 2                         # Default -- typically overridden to 4 in main.yaml
  head_conv: 64
freeze: null                            # List of layer names to freeze, or null
```

### Key Parameters
- `name`: Must match a registered model class (e.g., `HerdNetTimmDLA`)
- `load_from`: Loads weights only (for fine-tuning). Model architecture must match.
- `resume_from`: Loads weights + optimizer + epoch state (for resuming interrupted training)
- `kwargs`: Passed directly to the model constructor

## Dataset Config

```yaml
# datasets/floreana_dataset.yaml
img_size: [512, 512]
anno_type: 'point'
num_classes: 3                          # background(0) + iguana(1) + hard_negative(2)
collate_fn: null

class_def:
  1: 'iguana_point'
  2: 'hard_negative'

train:
  name: 'CSVDataset'
  csv_file: '/path/to/train/annotations.csv'
  root_dir: '/path/to/train/images/'
  sampler: null
  augmentation_multiplier: 1

  albu_transforms:
    ObjectAwareRandomCrop:
      height: 512
      width: 512
      p: 1.0
    Normalize:
      p: 1.0

  end_transforms:
    MultiTransformsWrapper:
      FIDT:
        num_classes: ${datasets.num_classes}              # INTERPOLATION
        down_ratio: ${model.kwargs.down_ratio}            # INTERPOLATION
        radius: 2
      PointsToMask:
        radius: 2
        num_classes: ${datasets.num_classes}              # INTERPOLATION
        squeeze: true
        down_ratio: 32

validate:
  name: 'CSVDataset'
  csv_file: '/path/to/val/annotations.csv'
  root_dir: '/path/to/val/images/'
  albu_transforms:
    Normalize:
      p: 1.0
  end_transforms:
    DownSample:
      down_ratio: ${model.kwargs.down_ratio}              # INTERPOLATION
      anno_type: ${datasets.anno_type}                    # INTERPOLATION

test:
  name: 'CSVDataset'
  csv_file: '/path/to/test/annotations.csv'
  root_dir: '/path/to/test/images/'
  albu_transforms:
    Normalize:
      p: 1.0
  end_transforms:
    DownSample:
      down_ratio: ${model.kwargs.down_ratio}              # INTERPOLATION
      anno_type: ${datasets.anno_type}                    # INTERPOLATION
```

### Transform Pipeline
1. **albu_transforms**: Applied via albumentations. For points, uses `KeypointParams(format='xy')`. Applied first.
2. **end_transforms**: Applied after albumentations. Converts annotations to model targets:
   - `FIDT`: Converts point annotations to Focal Inverse Distance Transform heatmaps
   - `PointsToMask`: Converts points to segmentation masks (for classification head)
   - `DownSample`: Downsamples annotations by `down_ratio` (for validation/test)

### MultiTransformsWrapper
Wraps multiple end transforms so they can be applied to the same input. The training pipeline uses `MultiTransformsWrapper` containing both `FIDT` and `PointsToMask`.

## Losses Config

```yaml
# losses/herdnet_iguana.yaml
FocalLoss:
  print_name: 'focal_loss'
  from_torch: False                     # Custom loss from animaloc, not torch.nn
  output_idx: 0                         # Maps to model output[0] (heatmap)
  target_idx: 0                         # Maps to target[0] (FIDT map)
  lambda_const: 1.0                     # Loss weighting factor
  kwargs:
    alpha: 2
    beta: 5                             # OPTIMAL: 5 (default was 4)
    reduction: 'mean'
    normalize: False

CrossEntropyLoss:
  print_name: 'ce_loss'
  from_torch: True                      # Standard PyTorch loss
  output_idx: 1                         # Maps to model output[1] (class map)
  target_idx: 1                         # Maps to target[1] (class mask)
  lambda_const: 1.0
  background_class_weight: 0.1
  kwargs:
    reduction: 'mean'
    weight:
      - ${losses.CrossEntropyLoss.background_class_weight}   # background: 0.1
      - 5                                                     # iguana_point
      - 0.1                                                   # hard_negative
```

### CRITICAL: CE Weight List Length
The `weight` list in `CrossEntropyLoss.kwargs` must have **exactly `num_classes` entries** (including background). If `num_classes=3`, you need 3 weights: `[bg_weight, class1_weight, class2_weight]`.

### Loss Mapping via output_idx / target_idx
The `LossWrapper` class maps model outputs and targets by index:
- `output_idx: 0` -> First model output (heatmap from localization head)
- `output_idx: 1` -> Second model output (class map from classification head)
- `target_idx: 0` -> First target tensor (FIDT map from end_transforms)
- `target_idx: 1` -> Second target tensor (class mask from PointsToMask)

## Training Settings Config

```yaml
# training_settings/evaluator.yaml
trainer: 'Trainer'
epochs: 20
valid_freq: 1                           # Validate every N epochs
print_freq: 20
batch_size: 2
optimizer: 'adamW'
lr: 1.0e-4                             # Head learning rate
backbone_lr: 1.0e-6                    # Backbone learning rate (lower for fine-tuning)
head_lr: 1.0e-4
weight_decay: 3.25e-4                  # OPTIMAL: 3.25e-4 (default was 1.6e-4)
num_workers: 4
warmup_iters: 100                      # Linear warmup iterations

auto_lr:
  mode: 'max'                          # 'max' for F1 (higher is better)
  patience: 15
  threshold: 1e-4
  threshold_mode: 'rel'
  cooldown: 10
  min_lr: 1e-7
  verbose: True

vizual_fn: visualize_sample
visualiser:
  name: 'HeatMapVisualizer'
  output_dir: ./visualizations
  down_ratio: ${model.kwargs.down_ratio}   # INTERPOLATION

debug_visualiser: null

loss_evaluation: null

evaluator:
  name: 'HerdNetEvaluator'
  threshold: 75                         # MATCHING RADIUS in pixels -- CRITICAL
  select_mode: 'max'
  validate_on: 'f1_score'
  kwargs:
    print_freq: 100
    lmds_kwargs:
      kernel_size: [5, 5]              # OPTIMAL: (5,5), default was (3,3)
      adapt_ts: 0.5                    # OPTIMAL: 0.5, default was 0.3
      scale_factor: 1
      up: True

stitcher:
  name: 'HerdNetStitcher'
  kwargs:
    overlap: 120
    down_ratio: ${model.kwargs.down_ratio}   # INTERPOLATION
    up: False
    reduction: 'mean'
```

## Interpolation Rules -- CRITICAL

Hydra interpolations use `${}` syntax to reference values from other config groups. These are resolved at runtime.

### Required Consistency Points

| Value | Must Match In | Interpolation |
|-------|--------------|---------------|
| `down_ratio` | model, FIDT, DownSample, stitcher, visualiser | `${model.kwargs.down_ratio}` |
| `num_classes` | model, FIDT, PointsToMask, CE loss weight length | `${datasets.num_classes}` |
| `anno_type` | datasets, DownSample | `${datasets.anno_type}` |

### Common Interpolation Errors

1. **Circular reference**: `${model.kwargs.down_ratio}` in model config itself
2. **Missing reference target**: Config group not loaded in `defaults`
3. **Wrong nesting level**: `${model.down_ratio}` vs `${model.kwargs.down_ratio}`

## Override Syntax

Hydra supports command-line overrides:

```bash
# Override a single parameter
python tools/train.py model.kwargs.backbone=timm/dla60

# Override nested parameter
python tools/train.py training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=0.3

# Override with different config file
python tools/train.py model=HerdNetTimmConvNext

# Print resolved config without running
python tools/train.py --cfg job
```

## Validation Checklist

Before running training, verify:

- [ ] `num_classes` in model matches `num_classes` in dataset config
- [ ] `down_ratio` is consistent across model, FIDT, stitcher, visualiser
- [ ] CE loss `weight` list has exactly `num_classes` entries
- [ ] `output_idx` and `target_idx` correctly map losses to model outputs/targets
- [ ] CSV file paths exist and are readable
- [ ] Image root directories exist
- [ ] Matching radius (`evaluator.threshold`) is appropriate (75px for iguanas)
- [ ] All interpolation references (`${}`) resolve correctly

## Common Config Mistakes

| Mistake | Error Message | Fix |
|---------|--------------|-----|
| num_classes mismatch | `RuntimeError: weight tensor size mismatch` | Make CE weight list length = num_classes |
| down_ratio mismatch | FIDT maps don't match model output size | Use `${model.kwargs.down_ratio}` everywhere |
| Missing defaults entry | `ConfigCompositionException` | Add missing config to defaults list |
| Wrong interpolation path | `InterpolationKeyError` | Check exact nesting of target key |
| load_from wrong architecture | `RuntimeError: size mismatch` | Ensure checkpoint matches current model config |
