# HerdNet Architecture Reference

## Overview

HerdNet is a two-headed convolutional neural network designed for simultaneous animal localization and classification in aerial imagery. It produces two outputs:
1. A **localization heatmap** (FIDT-style) for detecting where animals are
2. A **classification map** for identifying which species each detection belongs to

Source: `animaloc/models/herdnet_timm_dla.py` -- `HerdNetTimmDLA` class

## Architecture Diagram

```
Input: [B, 3, H, W]  (e.g., [B, 3, 512, 512])
         |
    ┌────┴────┐
    │ Backbone │  timm model (DLA-34, ConvNext, EfficientNet)
    │          │  features_only=True -> multi-scale features
    └────┬────┘
         |
    Feature levels: [f0, f1, f2, f3, f4, f5]
    (for DLA-34: strides [1, 2, 4, 8, 16, 32])
         |
    Select from first_level based on down_ratio:
    DR=2: first_level=0 -> [f0, f1, f2, f3, f4, f5]
    DR=4: first_level=1 -> [f1, f2, f3, f4, f5]
    DR=8: first_level=2 -> [f2, f3, f4, f5]
         |
    ┌────┴────┐
    │Bottleneck│  Conv2d(channels[-1], channels[-1], 1x1)
    │   Conv   │  Applied to deepest feature before DLAUp
    └────┬────┘
         |
    ┌────┴────┐
    │  DLAUp  │  Feature Pyramid Upsampling
    │         │  Fuses multi-scale features to first_level resolution
    └────┬────┘
         |
    ┌────┴────────────────────┐
    │                          │
┌───┴───┐              ┌──────┴──────┐
│Loc Head│              │  Cls Head   │
│        │              │             │
│Conv2d  │              │Conv2d       │
│(in→64) │              │(deep→64)    │
│ReLU    │              │ReLU         │
│Conv2d  │              │Conv2d       │
│(64→1)  │              │(64→C)       │
│Sigmoid │              │(raw logits) │
└───┬───┘              └──────┬──────┘
    │                          │
 Heatmap                   Class Map
[B,1,H/DR,W/DR]         [B,C,H/16,W/16]
```

## Constructor Parameters

```python
HerdNetTimmDLA(
    num_classes: int = 2,            # Total classes INCLUDING background
    pretrained: bool = True,         # Load ImageNet pretrained backbone
    down_ratio: int = 2,             # Heatmap downsampling factor (2, 4, 8, 16)
    head_conv: int = 64,             # Intermediate channels in both heads
    pretrained_path: str = None,     # Path to custom pretrained backbone weights
    debug: bool = True,              # Print architecture info and test forward pass
    backbone: str = 'timm/dla34'     # timm model identifier
)
```

## Backbone: timm Integration

The backbone is created via the `timm` library with `features_only=True`:

```python
base = timm.create_model(backbone, pretrained=pretrained, features_only=True)
```

This returns a model that outputs a list of feature maps at different spatial resolutions instead of a single classification logit. For DLA-34 with a 512x512 input:

| Level | Stride | Output Size | Channels |
|-------|--------|-------------|----------|
| 0 | 1 | 512x512 | 16 |
| 1 | 2 | 256x256 | 32 |
| 2 | 4 | 128x128 | 64 |
| 3 | 8 | 64x64 | 128 |
| 4 | 16 | 32x32 | 256 |
| 5 | 32 | 16x16 | 512 |

### first_level Calculation

The `first_level` determines which features are used, based on `down_ratio`:

```python
# For DLA backbones:
first_level = int(np.log2(down_ratio)) - 1
# DR=2 -> first_level=0 (use all 6 levels)
# DR=4 -> first_level=1 (use levels 1-5)
# DR=8 -> first_level=2 (use levels 2-5)

# For ConvNext backbones (only 4 feature levels):
first_level = int(np.log2(down_ratio)) - 2
# Minimum DR=4 for ConvNext

```

## DLAUp: Feature Fusion

Source: `animaloc/models/dla.py` -- `DLAUp` class

DLAUp progressively upsamples and fuses deeper features with shallower features:

```python
selected_channels = feature_channels[first_level:]
scales = [2**i for i in range(len(selected_channels))]
self.dla_up = DLAUp(selected_channels, scales=scales)
```

The output has the same spatial resolution and channel count as `features[first_level]`.

## Bottleneck Convolution

A 1x1 convolution applied to the deepest feature map before feeding into DLAUp:

```python
self.bottleneck_conv = nn.Conv2d(
    channels[-1], channels[-1],
    kernel_size=1, stride=1, padding=0, bias=True
)
```

This acts as a channel mixer/reducer at the deepest level before multi-scale fusion.

## Localization Head

Predicts the FIDT heatmap. Output is a single-channel probability map via Sigmoid:

```python
self.loc_head = nn.Sequential(
    nn.Conv2d(channels[first_level], head_conv, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(head_conv, 1, kernel_size=1),
    nn.Sigmoid()
)
```

- Input: DLAUp output at `first_level` resolution `[B, C_first, H/DR, W/DR]`
- Output: `[B, 1, H/DR, W/DR]` -- values in [0, 1]
- Trained with: FocalLoss against FIDT ground truth maps
- Bias initialized to 0.0

## Classification Head

Predicts per-pixel class logits from the deepest feature level:

```python
self.cls_head = nn.Sequential(
    nn.Conv2d(channels[-1], head_conv, kernel_size=3, padding=1),
    nn.ReLU(inplace=True),
    nn.Conv2d(head_conv, num_classes, kernel_size=1)
)
```

- Input: Deepest feature map (after bottleneck) `[B, C_deep, H/16, W/16]`
- Output: `[B, num_classes, H/16, W/16]` -- raw logits (not softmax-ed)
- Trained with: CrossEntropyLoss against PointsToMask ground truth
- Bias initialized to 0.0
- Note: Always operates at stride 16 regardless of `down_ratio`

## Forward Pass

```python
def forward(self, x):
    feats = self.backbone(x)                    # All feature levels
    selected_feats = feats[self.first_level:]    # Subset by down_ratio

    bottlenecked = self.bottleneck_conv(feats[-1])
    selected_feats[-1] = bottlenecked           # Replace deepest with bottlenecked

    upsampled = self.dla_up(selected_feats)     # Fuse and upsample

    heatmap = self.loc_head(upsampled)          # [B, 1, H/DR, W/DR]
    clsmap = self.cls_head(selected_feats[-1])  # [B, C, H/16, W/16]

    return heatmap, clsmap
```

### Output Shapes (512x512 input)

| down_ratio | Heatmap Shape | Class Map Shape |
|-----------|--------------|-----------------|
| 2 | [B, 1, 256, 256] | [B, C, 32, 32] |
| 4 | [B, 1, 128, 128] | [B, C, 32, 32] |
| 8 | [B, 1, 64, 64] | [B, C, 32, 32] |
| 16 | [B, 1, 32, 32] | [B, C, 32, 32] |

Note: The class map is always at H/16 x W/16 because it always uses the deepest feature level at stride 16.

## Key Methods

### freeze_backbone_completely()
```python
def freeze_backbone_completely(self):
    for param in self.backbone.parameters():
        param.requires_grad = False
```
Freezes all backbone parameters. Only the heads and DLAUp are trainable. Useful for initial fine-tuning phases.

### reshape_classes(num_classes)
```python
def reshape_classes(self, num_classes: int):
    self.cls_head[-1] = nn.Conv2d(
        self.head_conv, num_classes,
        kernel_size=1, stride=1, padding=0, bias=True
    )
    self.cls_head[-1].bias.data.fill_(0.0)
    self.num_classes = num_classes
```
Replaces the final classification convolution for a new number of classes. Only the last Conv2d layer is replaced -- all other learned weights are preserved.

### check_trainable_parameters()
Reports total, trainable, and percentage of trainable parameters. Useful for verifying that freeze/unfreeze operations worked correctly.

### freeze(layers)
```python
def freeze(self, layers: list):
    # Freeze specific layer groups by name
    # e.g., model.freeze(['backbone', 'dla_up'])
```

## Related Model Variants

| Model Class | File | Backbone | Notes |
|------------|------|----------|-------|
| `HerdNetTimmDLA` | `herdnet_timm_dla.py` | DLA/ConvNext/EfficientNet via timm | Primary model |
| `HerdNetDINOv21` | `herdnet_dino_v2.py` | DINOv2 ViT | Vision Transformer backbone |
| `HerdNet` | `herdnet.py` | Original DLA (non-timm) | Legacy implementation |
| `HerdNetP2P` | `herdnet_p2p.py` | Point-to-Point variant | Experimental |

## Memory Footprint (approximate, 512x512 input, batch_size=1)

| Configuration | GPU Memory |
|--------------|------------|
| DLA-34, DR=2 | ~4 GB |
| DLA-34, DR=4 | ~2.5 GB |
| DLA-60, DR=4 | ~3.5 GB |
| DLA-169, DR=4 | ~5 GB |
| ConvNext-Tiny, DR=4 | ~3 GB |
