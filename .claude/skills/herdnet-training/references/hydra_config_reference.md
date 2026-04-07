# Hydra Config Reference

## Overview

HerdNet uses [Hydra](https://hydra.cc/) for hierarchical configuration management. The config is organized as a 4-level tree that gets composed at runtime into a single flat configuration.

Source: `configs/demo/` -- Example config tree in the HerdNet repository

## Config Tree Structure

```
configs/
    main.yaml                              # Top-level entry point
    model/
        HerdNetTimmDLA34.yaml              # Model architecture
        HerdNetTimmConvNext_Camouflaged.yaml
    datasets/
        floreana_dataset.yaml              # Dataset configuration
        general_dataset.yaml
    losses/
        herdnet_iguana.yaml                # Loss functions
        herdnet_general_dataset.yaml
    training_settings/
        evaluator.yaml                     # Training & evaluation params
```

## Composition via defaults

The `defaults` list in `main.yaml` tells Hydra which sub-configs to load:

```yaml
defaults:
  - /losses: herdnet_iguana              # Load configs/losses/herdnet_iguana.yaml
  - /datasets: floreana_dataset          # Load configs/datasets/floreana_dataset.yaml
  - /training_settings: evaluator        # Load configs/training_settings/evaluator.yaml
  - /model: HerdNetTimmDLA34            # Load configs/model/HerdNetTimmDLA34.yaml
  - _self_                               # Apply this config's own values last
```

The leading `/` means absolute path from config root. `_self_` means the current file's values are applied last and can override sub-config values.

## Interpolation Syntax

Hydra uses OmegaConf's interpolation syntax to reference values across config groups:

```yaml
# Reference a value from another config group
down_ratio: ${model.kwargs.down_ratio}

# Reference within the same config
background_weight: ${losses.CrossEntropyLoss.background_class_weight}

# Reference from parent
num_classes: ${datasets.num_classes}
```

### Interpolation Resolution

Interpolations are resolved at runtime when the config is accessed. The full config tree is composed first, then interpolations are resolved.

```yaml
# In datasets/my_dataset.yaml
end_transforms:
  FIDT:
    down_ratio: ${model.kwargs.down_ratio}    # -> resolves to value from model config
    num_classes: ${datasets.num_classes}       # -> resolves to value from this dataset config
```

### Common Interpolation Patterns

| Pattern | From | To |
|---------|------|-----|
| `${model.kwargs.down_ratio}` | Dataset/Evaluator/Visualiser | Model config |
| `${datasets.num_classes}` | FIDT/PointsToMask | Dataset config |
| `${datasets.anno_type}` | DownSample transform | Dataset config |
| `${losses.CrossEntropyLoss.background_class_weight}` | CE weight list | Loss config |

## Consistency Rules -- CRITICAL

### Rule 1: down_ratio

`down_ratio` must be identical across all of these locations:

| Location | Config Path | Example |
|----------|------------|---------|
| Model | `model.kwargs.down_ratio` | `4` (source of truth) |
| FIDT transform | `datasets.train.end_transforms.*.FIDT.down_ratio` | `${model.kwargs.down_ratio}` |
| Validation DownSample | `datasets.validate.end_transforms.DownSample.down_ratio` | `${model.kwargs.down_ratio}` |
| Test DownSample | `datasets.test.end_transforms.DownSample.down_ratio` | `${model.kwargs.down_ratio}` |
| Stitcher | `training_settings.stitcher.kwargs.down_ratio` | `${model.kwargs.down_ratio}` |
| Visualiser | `training_settings.visualiser.down_ratio` | `${model.kwargs.down_ratio}` |

If any of these are inconsistent, training will either crash or produce silently wrong results.

### Rule 2: num_classes

`num_classes` must be consistent across:

| Location | Config Path | Constraint |
|----------|------------|------------|
| Model | `model.kwargs.num_classes` | Must include background |
| Dataset | `datasets.num_classes` | Must match model |
| FIDT transform | FIDT's `num_classes` param | `${datasets.num_classes}` |
| PointsToMask | PointsToMask's `num_classes` | `${datasets.num_classes}` |
| CE Loss weights | `losses.CrossEntropyLoss.kwargs.weight` | List length = num_classes |
| Metrics | `training_settings.evaluator.metrics.num_classes` | Must match model |

### Rule 3: patch_size / img_size

Patch size should be consistent between training and inference:

| Location | Config Path |
|----------|------------|
| Training | `datasets.img_size` |
| Stitcher | `training_settings.stitcher.kwargs.size` |
| Patcher | `patcher --height --width` |

## Command-Line Overrides

Hydra supports overriding any config value from the command line:

```bash
# Override scalar value
python tools/train.py model.kwargs.backbone=timm/dla60

# Override nested value
python tools/train.py training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=0.3

# Override with a different config file
python tools/train.py model=HerdNetTimmConvNext

# Multiple overrides
python tools/train.py model.kwargs.down_ratio=4 training_settings.lr=0.0001 training_settings.epochs=20

# Print resolved config without running
python tools/train.py --cfg job

# Print defaults list
python tools/train.py --info defaults
```

## Complete Config Templates

### Template: Iguana Detection (Optimal)

**main.yaml**:
```yaml
defaults:
  - /losses: herdnet_iguana
  - /datasets: floreana_dataset
  - /training_settings: evaluator
  - /model: HerdNetTimmDLA34
  - _self_

wandb_flag: True
wandb_project: 'herdnet'
wandb_entity: 'your_team'
wandb_run: 'floreana_optimal'
wandb_tags: ["dla34", "dr4", "optimal"]
seed: 42
device_name: null

model:
  kwargs:
    down_ratio: 4
    head_conv: 64

training_settings:
  lr: 1.0e-4
  backbone_lr: 1.0e-6
  weight_decay: 3.25e-4
  warmup_iters: 100
  epochs: 20
  batch_size: 2
  evaluator:
    threshold: 75
    kwargs:
      lmds_kwargs:
        kernel_size: [5, 5]
        adapt_ts: 0.5

hydra:
  run:
    dir: ./outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

### Template: New Species (Generic)

```yaml
defaults:
  - /losses: herdnet_general
  - /datasets: general_dataset
  - /training_settings: evaluator
  - /model: HerdNetTimmDLA34
  - _self_

wandb_flag: False
seed: 42
device_name: null

# EDIT THESE for your species:
datasets:
  num_classes: 2           # 1 species + background
  class_def:
    1: 'my_species'

model:
  kwargs:
    down_ratio: 4
    head_conv: 64

training_settings:
  epochs: 30
  evaluator:
    threshold: 50          # Adjust based on animal size in pixels
```

## Debugging Config Issues

### Print Resolved Config
```bash
python tools/train.py --cfg job
```
This prints the fully resolved config (all interpolations expanded) without running training.

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `InterpolationKeyError` | Referenced key doesn't exist | Check interpolation path matches actual config structure |
| `ConfigCompositionException` | Missing config file in defaults | Verify file exists at referenced path |
| `MissingMandatoryValue` | Used `???` without providing a value | Supply the value via command line or config |
| `OmegaConf.resolve` error | Circular interpolation | Remove circular reference |
| `RuntimeError: size mismatch` | num_classes or down_ratio inconsistent | Check all consistency rules above |

### Validation Script

```python
from omegaconf import OmegaConf, DictConfig

def validate_config(cfg: DictConfig):
    errors = []

    # Check down_ratio consistency
    model_dr = cfg.model.kwargs.down_ratio
    if hasattr(cfg.datasets.train, 'end_transforms'):
        # Check FIDT down_ratio
        fidt_dr = cfg.datasets.train.end_transforms.MultiTransformsWrapper.FIDT.down_ratio
        if fidt_dr != model_dr:
            errors.append(f"FIDT down_ratio ({fidt_dr}) != model down_ratio ({model_dr})")

    # Check num_classes consistency
    model_nc = cfg.model.kwargs.num_classes if 'num_classes' in cfg.model.kwargs else None
    dataset_nc = cfg.datasets.num_classes
    if model_nc and model_nc != dataset_nc:
        errors.append(f"Model num_classes ({model_nc}) != dataset num_classes ({dataset_nc})")

    # Check CE loss weight length
    if hasattr(cfg.losses, 'CrossEntropyLoss'):
        ce_weights = list(cfg.losses.CrossEntropyLoss.kwargs.weight)
        if len(ce_weights) != dataset_nc:
            errors.append(f"CE weight length ({len(ce_weights)}) != num_classes ({dataset_nc})")

    return errors
```
