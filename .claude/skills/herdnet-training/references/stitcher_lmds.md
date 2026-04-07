# Stitcher and LMDS Reference

## Overview

The inference pipeline for large orthomosaics consists of two components:
1. **HerdNetStitcher**: Splits images into overlapping tiles, runs model inference on each, and reassembles the output using Hann window weighting
2. **HerdNetLMDS**: Detects local maxima in the assembled heatmap and assigns class labels

Source files:
- `animaloc/eval/stitchers.py` -- Stitcher and HerdNetStitcher
- `animaloc/eval/lmds.py` -- LMDS and HerdNetLMDS
- `animaloc/eval/utils.py` -- HannWindow2D

## HerdNetStitcher

### Class Hierarchy

```
ImageToPatches
    └── Stitcher
            └── HerdNetStitcher
```

`Stitcher` extends `ImageToPatches` (which handles tiling) and adds inference + reassembly. `HerdNetStitcher` overrides `_inference()` to handle HerdNet's two-headed output.

### Constructor

```python
HerdNetStitcher(
    model: nn.Module,           # LossWrapper-wrapped model
    size: tuple,                # Patch size (height, width), e.g., (512, 512)
    overlap: int = 100,         # Overlap between tiles in pixels
    batch_size: int = 1,        # Tiles per inference batch
    down_ratio: int = 1,        # Must match model's down_ratio
    up: bool = False,           # Upsample after stitching
    reduction: str = 'sum',     # How to combine overlapping regions
    device_name: str = 'cuda'
)
```

### How Stitching Works

#### Step 1: Tile the image
```
Original image: [C, H, W]
                ┌──────────┐
                │  Tile 0,0 │ overlap │ Tile 0,1 │ overlap │ Tile 0,2 │
                │           ├─────────┤           ├─────────┤           │
                │           │         │           │         │           │
                ├───────────┤         ├───────────┤         │           │
                │  overlap  │         │  overlap  │         │           │
                ├───────────┤         ├───────────┤         │           │
                │  Tile 1,0 │ overlap │ Tile 1,1 │ overlap │ Tile 1,2 │
                └──────────┘
```

The `ImageToPatches` base class handles dividing the image. Tiles are `size[0] x size[1]` pixels with `overlap` pixels of shared border.

#### Step 2: Run inference on each tile

`HerdNetStitcher._inference()` processes tiles through the model:

```python
@torch.no_grad()
def _inference(self, patches):
    maps = []
    for patch in dataloader:
        outputs = self.model(patch)[0]         # Get model output
        heatmap = outputs[0]                    # [B, 1, H/DR, W/DR]
        clsmap = outputs[1]                     # [B, C, H/16, W/16]

        # Upsample clsmap to match heatmap resolution
        scale_factor = heatmap.size(-1) // clsmap.size(-1)
        clsmap = F.interpolate(clsmap, scale_factor=scale_factor, mode='nearest')

        # Concatenate heatmap + clsmap
        outmaps = torch.cat([heatmap, clsmap], dim=1)  # [B, 1+C, H/DR, W/DR]
        maps.append(outmaps)
    return maps
```

Key detail: The classification map is upsampled to match the heatmap resolution via nearest-neighbor interpolation before concatenation. This means the stitched output has `1 + num_classes` channels.

#### Step 3: Reassemble using F.fold

The base `Stitcher._patch_maps()` uses `torch.nn.functional.fold` to reassemble tile outputs into the full image coordinate system. This operation is essentially the inverse of `unfold`.

#### Step 4: Reduce overlapping regions

```python
def _reduce(self, map):
    if self.reduction == 'mean':
        # Create a normalization map by counting how many tiles cover each pixel
        # Divide the stitched output by this count
        norm_map = self._compute_normalization_map()
        return map / norm_map
    elif self.reduction == 'sum':
        return map  # Keep the sum (may cause double-counting)
    elif self.reduction == 'max':
        # Handled separately via _max_fold
        return map
```

#### Step 5: Optional upsampling

If `up=True`, the output is bilinearly upsampled by `down_ratio`:
```python
if self.up:
    patched_map = F.interpolate(patched_map, scale_factor=self.down_ratio,
                                mode='bilinear', align_corners=True)
```

**Recommendation**: Set `up=False` when using `reduction='mean'`. The coordinate conversion is handled downstream by multiplying LMDS coordinates by `down_ratio`.

### Recommended Parameters

| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| size | (512, 512) | Match training patch size |
| overlap | 120 | Sufficient to avoid missing detections at borders |
| down_ratio | 4 | Match model's down_ratio |
| up | False | Let LMDS handle coordinate conversion |
| reduction | 'mean' | Smooth blending, prevents double-counting |
| batch_size | 1 | Conservative; increase if GPU memory allows |

### Output Format

```python
output = stitcher(image_tensor)
# output shape: [1, 1+C, H/DR, W/DR]
# where C = num_classes (including background)
#
# Channel 0: localization heatmap (Sigmoid-activated, 0-1)
# Channels 1..C: classification logits (upsampled from H/16 to H/DR)
```

## Hann Window

Source: `animaloc/eval/utils.py` -- `HannWindow2D`

A 2D Hann (raised cosine) window is applied to each tile's output before stitching. This ensures smooth blending at tile borders:

```
Weight profile across a tile:
1.0  ___________
    /           \
   /             \
  /               \
0.0 ___/         \___
   edge  center  edge
```

### Window Types by Tile Position

The `DensityMapStitcher._make_hann_matrix()` generates different windows based on tile position in the grid:

| Position | Window Type | Edges with Hann Taper |
|----------|------------|----------------------|
| Top-left corner | `corner, up_left` | Right edge, bottom edge |
| Top edge | `edge, up` | Bottom edge |
| Top-right corner | `corner, up_right` | Left edge, bottom edge |
| Left edge | `edge, left` | Right edge |
| Interior | `original, up` | All four edges |
| Right edge | `edge, right` | Left edge |
| Bottom-left corner | `corner, down_left` | Right edge, top edge |
| Bottom edge | `edge, down` | Top edge |
| Bottom-right corner | `corner, down_right` | Left edge, top edge |

Corner and edge tiles do NOT have Hann taper on the image boundary sides (since there's no overlap there).

Note: `HerdNetStitcher` does not use Hann windowing (it is only used in `DensityMapStitcher`). HerdNet instead relies on `reduction='mean'` for blending.

## HerdNetLMDS

### Class Hierarchy

```
LMDS
    └── HerdNetLMDS
```

`LMDS` handles basic local maxima detection. `HerdNetLMDS` adds handling for HerdNet's two-output architecture.

### LMDS Constructor

```python
LMDS(
    kernel_size: tuple = (3, 3),         # Local maxima search window
    adapt_ts: float = 100.0/255.0,       # Adaptive threshold (~0.392)
    neg_ts: float = 0.1,                 # Negative sample threshold
    score_threshold: float = 0.3         # Absolute score threshold
)
```

### HerdNetLMDS Constructor

```python
HerdNetLMDS(
    up: bool = True,                     # Upsample clsmap to match heatmap
    kernel_size: tuple = (3, 3),         # Local maxima search window
    adapt_ts: float = 0.3,              # Adaptive threshold (default)
    neg_ts: float = 0.1,                # Negative sample threshold
    scale_factor: int = 16              # Clsmap upsample factor
)
```

### Optimal Parameters (Miesner 2025)

| Parameter | Optimal | Default | Effect |
|-----------|---------|---------|--------|
| kernel_size | (5, 5) | (3, 3) | Larger kernel better for iguana spacing |
| adapt_ts | 0.5 | 0.3 | Higher threshold reduces false positives |
| neg_ts | 0.1 | 0.1 | Keep at default |
| up | context-dependent | True | False with stitcher, True without |
| scale_factor | 1 (with stitcher) | 16 | 1 when stitcher already scaled |

### How LMDS Works

#### Step 1: Local Maxima Detection
```python
# Max pooling with kernel_size, stride=1, padding=kernel_size//2
# Keeps only pixels that are the maximum in their local neighborhood
keep = F.max_pool2d(est_map, kernel_size=self.kernel_size, stride=1, padding=pad)
keep = (keep == est_map).float()
est_map = keep * est_map
```

#### Step 2: Adaptive Thresholding
```python
# Only keep peaks above adapt_ts * global_maximum
est_map_max = torch.max(est_map).item()
est_map[est_map < self.adapt_ts * est_map_max] = 0
```

#### Step 3: Negative Sample Check
```python
# If global max is below neg_ts, treat as empty (no detections)
if est_map_max < self.neg_ts:
    est_map = est_map * 0
```

#### Step 4: Extract Locations and Scores
```python
# Find all non-zero pixels -> these are the detections
locs = numpy.argwhere(est_map == 1)  # (row, col) tuples
scores = est_map[locs]                # Detection confidence
```

### HerdNetLMDS Specifics

`HerdNetLMDS.__call__()` receives the full HerdNet output `[heatmap, clsmap]`:

1. **Upsample clsmap** (if `up=True`):
   ```python
   clsmap = F.interpolate(clsmap, scale_factor=self.scale_factor, mode='nearest')
   ```

2. **Softmax on class scores**:
   ```python
   cls_scores = torch.softmax(clsmap, dim=1)[:, 1:, :, :]  # Exclude background
   ```

3. **Run LMDS on heatmap** to get detection locations

4. **Assign class labels** by looking up the class with highest score at each detection location:
   ```python
   cls_idx = torch.argmax(clsmap[:, 1:, :, :], dim=0)
   labels = cls_idx[detection_rows, detection_cols] + 1  # 1-indexed
   ```

5. **Return**: `(counts, locs, labels, cls_scores, det_scores)`
   - `counts`: List of per-class counts, e.g., `[42, 3]`
   - `locs`: List of `(row, col)` tuples in heatmap coordinates
   - `labels`: List of class labels (1-indexed)
   - `cls_scores`: Classification confidence (from softmax)
   - `det_scores`: Detection confidence (from heatmap value)

### Coordinate Systems

LMDS returns coordinates in the **heatmap coordinate system** (downsampled by `down_ratio`):

```
Heatmap coords: (row, col) in [0, H/DR) x [0, W/DR)
Image coords:   (x, y) = (col * DR, row * DR)
```

To convert to image coordinates:
```python
for (row, col) in locs:
    image_x = col * down_ratio
    image_y = row * down_ratio
```

### up Parameter Decision Table

| Context | up | scale_factor | Reasoning |
|---------|-----|-------------|-----------|
| With HerdNetStitcher | False | 1 | Stitcher already concatenated and scaled |
| Without stitcher (direct model output) | True | 8 or 16 | Need to upsample clsmap to heatmap res |
| With DensityMapStitcher | N/A | N/A | LMDS not used (density map has different pipeline) |

The `HerdNetEvaluator.prepare_feeding()` handles this automatically:
```python
if self.stitcher is not None:
    lmds = HerdNetLMDS(up=False, **self.lmds_kwargs)
else:
    lmds = HerdNetLMDS(up=True, **self.lmds_kwargs)
```
