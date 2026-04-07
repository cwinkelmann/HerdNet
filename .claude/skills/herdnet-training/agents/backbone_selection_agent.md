# Backbone Selection Agent

## Role
Compare DLA-34/60/102/169, ConvNext, and DINOv2 backbones for HerdNet point detection. Provide evidence-based recommendations grounded in the Miesner 2025 thesis benchmarks and the actual codebase implementation.

## Architecture Overview

HerdNet is a **two-headed CNN** built on a shared backbone:

```
Input Image [B, 3, H, W]
       |
  [Backbone] -- timm model with features_only=True
       |
  Multi-scale features (e.g., 5 levels for DLA-34)
       |
  [DLAUp] -- Feature Pyramid Upsampling
       |
       +---> [Localization Head] --> Heatmap [B, 1, H/DR, W/DR]
       |       Conv2d -> ReLU -> Conv2d -> Sigmoid
       |
       +---> [Classification Head] --> Class Map [B, C, H/16, W/16]
               Conv2d -> ReLU -> Conv2d
```

Source: `animaloc/models/herdnet_timm_dla.py` -- `HerdNetTimmDLA` class

### Key Architecture Details
- **Localization head** outputs a single-channel heatmap via Sigmoid activation, trained with FocalLoss against FIDT ground truth maps
- **Classification head** outputs `num_classes` channels (including background), trained with CrossEntropyLoss
- **head_conv** controls the intermediate channel count in both heads: `Conv2d(in, head_conv, 3) -> ReLU -> Conv2d(head_conv, out, 1)`
- **down_ratio** determines the output resolution of the localization heatmap: `output_size = input_size / down_ratio`
- The classification head always operates at the deepest feature level (stride 16 relative to input)

## HerdNetTimmDLA Constructor

```python
HerdNetTimmDLA(
    num_classes: int = 2,         # includes background
    pretrained: bool = True,      # ImageNet pretrained backbone
    down_ratio: int = 2,          # output downsampling factor (2, 4, 8, or 16)
    head_conv: int = 64,          # intermediate channels in heads
    pretrained_path: str = None,  # path to custom pretrained weights
    debug: bool = True,           # print architecture info
    backbone: str = 'timm/dla34'  # timm model name
)
```

### Backbone Name Formats
The `backbone` parameter supports three families with different `first_level` calculation logic:

| Family | Example Names | Min down_ratio | first_level Formula |
|--------|--------------|----------------|---------------------|
| DLA | `timm/dla34`, `timm/dla60`, `timm/dla102`, `timm/dla169` | 2 | `log2(DR) - 1` |
| ConvNext | `convnext_tiny`, `convnext_small`, `convnext_base` | 4 | `log2(DR) - 2` |
| EfficientNet | `efficientnet_b0`, `efficientnet_b3` | 2 | `log2(DR) - 1` |

### Model Creation Internals
```python
# The backbone is created via timm with features_only=True
base = timm.create_model(backbone, pretrained=pretrained, features_only=True)

# features_only=True returns multi-scale feature maps instead of classification logits
# For DLA-34, this gives 6 feature levels at strides [1, 2, 4, 8, 16, 32]
# first_level selects which features to use based on down_ratio
```

## Backbone Comparison: Thesis Results (Miesner 2025)

### DLA Variants

| Backbone | Parameters | F1 (Floreana) | F1 (Fernandina) | Overfitting? | Recommendation |
|----------|-----------|---------------|-----------------|--------------|----------------|
| **DLA-34** | ~15.7M | **0.934** | **0.843** | No | **BEST -- use this** |
| DLA-60 | ~22.0M | Lower | Lower | Yes | Avoid |
| DLA-102 | ~33.0M | Lower | Lower | Yes | Avoid |
| DLA-169 | ~53.4M | Lower | Lower | Yes | Avoid |

**Key finding**: All larger DLA variants overfit on the iguana dataset. DLA-34 provides the best balance of capacity and generalization.

### Why Larger Backbones Overfit
The iguana detection dataset is relatively small (2000-2500 annotations at the learning curve plateau). Larger backbones have more parameters than the data can constrain, leading to:
- Training loss continues decreasing while validation loss increases
- High training F1 but lower validation F1
- Poor cross-island generalization

### DINOv2 Backbone
- Available via `HerdNetDINOv21` class in `animaloc/models/herdnet_dino_v2.py`
- Uses Vision Transformer (ViT) features from DINOv2 self-supervised pretraining
- Requires significantly more GPU memory than DLA-34
- Was not fully benchmarked for HerdNet point detection at the time of the thesis
- May be promising for future work with larger datasets or different species
- Requires `gradient_checkpointing` for training on consumer GPUs

### ConvNext Backbone
- Available via `HerdNetTimmDLA` with `backbone='convnext_tiny'` (or `small`, `base`)
- Requires `down_ratio >= 4` due to different feature stride structure
- Modern ConvNet architecture with competitive ImageNet performance
- Not benchmarked in the thesis for iguana detection

## down_ratio Selection

The `down_ratio` (DR) parameter controls the resolution of the localization heatmap output:

| down_ratio | Output Size (512 input) | Memory | F1 | Notes |
|-----------|------------------------|--------|-----|-------|
| 2 | 256x256 | High | Lower | More memory, does NOT improve results |
| **4** | **128x128** | **Medium** | **Best** | **Optimal for iguana data** |
| 8 | 64x64 | Low | Lower | Too coarse for densely packed iguanas |
| 16 | 32x32 | Very Low | Poor | Only for very sparse distributions |

**Recommendation**: Always use `down_ratio=4` unless working with very different animal densities.

### Consistency Requirement
`down_ratio` must be set consistently across the entire pipeline:
1. `model.kwargs.down_ratio` -- model config
2. `datasets.train.end_transforms.FIDT.down_ratio` -- FIDT target generation (use `${model.kwargs.down_ratio}`)
3. `training_settings.stitcher.kwargs.down_ratio` -- inference stitcher
4. `training_settings.visualiser.down_ratio` -- visualization

## head_conv Selection

| head_conv | Parameters Added | Effect |
|-----------|-----------------|--------|
| 256 | More | Overfitting risk, more memory |
| 128 | Default in some configs | Adequate for large datasets |
| **64** | Less | **Optimal for iguana data**, better generalization |
| 32 | Fewer | May underfit on complex multi-species tasks |

**Recommendation**: Use `head_conv=64` for iguana detection and similar small-to-medium datasets.

## Model Methods

### freeze_backbone_completely()
```python
model.freeze_backbone_completely()
# Freezes all backbone parameters -- only heads are trained
# Useful for initial head-only training when fine-tuning
```

### reshape_classes(num_classes)
```python
model.reshape_classes(num_classes=4)
# Rebuilds the classification head for a new number of classes
# Used when adapting a pretrained model to a different species set
# Only affects cls_head[-1] (final Conv2d layer)
```

### check_trainable_parameters()
```python
model.check_trainable_parameters()
# Prints total, trainable, and percentage of trainable parameters
# Useful to verify freeze/unfreeze worked correctly
```

## Decision Flow

```
Start
  |
  +-- How much training data?
  |     |
  |     +-- < 2000 annotations --> DLA-34 with head_conv=64
  |     +-- 2000-10000 annotations --> DLA-34 with head_conv=64 or 128
  |     +-- > 10000 annotations --> Consider DLA-60 or ConvNext
  |
  +-- What GPU memory?
  |     |
  |     +-- < 8 GB --> DLA-34, DR=4, batch_size=1-2
  |     +-- 8-16 GB --> DLA-34, DR=4, batch_size=2-4
  |     +-- > 16 GB --> DLA-34, DR=4, batch_size=4-8 (or experiment with DINOv2)
  |
  +-- Multi-species or single?
        |
        +-- Single species --> num_classes=2 (or 3 with hard negatives)
        +-- Multiple species --> num_classes=N+1, consider head_conv=128
```

## Config Example for Recommended Setup

```yaml
# model/HerdNetTimmDLA34.yaml
name: 'HerdNetTimmDLA'
from_torchvision: False
load_from: null
resume_from: null
kwargs:
  backbone: 'timm/dla34'
  pretrained: True
  down_ratio: 4
  head_conv: 64
freeze: null
```
