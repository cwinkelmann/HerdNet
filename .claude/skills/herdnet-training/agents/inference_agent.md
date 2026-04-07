# Inference Agent

## Role
Guide users through the full HerdNet inference pipeline: loading a trained checkpoint, using HerdNetStitcher for overlapping tile inference on large orthomosaics, applying HerdNetLMDS for peak detection, and exporting results.

## Inference Pipeline Overview

```
Trained Model (.pth)
    |
    v
Load checkpoint --> HerdNetTimmDLA
    |
    v
Wrap with LossWrapper (mode='preds_only' or 'module')
    |
    v
HerdNetStitcher
    |-- Splits orthomosaic into overlapping tiles
    |-- Runs model on each tile
    |-- Stitches output heatmaps using Hann windowing
    |
    v
Stitched output: [1, 1+C, H/DR, W/DR]
    |-- Channel 0: localization heatmap
    |-- Channels 1..C: classification scores (softmax-ed)
    |
    v
HerdNetLMDS
    |-- Finds local maxima in heatmap
    |-- Assigns class labels from classification channels
    |-- Returns: counts, locations, labels, class_scores, detection_scores
    |
    v
Export to CSV / Overlay on image
```

## Step 1: Load Trained Model

```python
import torch
from animaloc.models import HerdNetTimmDLA
from animaloc.models.utils import LossWrapper

# Device selection
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')

# Create model with SAME architecture as training
model = HerdNetTimmDLA(
    backbone='timm/dla34',
    num_classes=3,          # Must match training config
    down_ratio=4,           # Must match training config
    head_conv=64,           # Must match training config
    pretrained=False,       # Don't need ImageNet weights, we have our own
    debug=False
)

# Load checkpoint
checkpoint = torch.load('best_model.pth', map_location=device)
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Handle LossWrapper prefix in state dict keys
clean_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('model.', '', 1) if k.startswith('model.') else k
    clean_state_dict[new_key] = v

model.load_state_dict(clean_state_dict, strict=False)
model.to(device)
model.eval()
```

### Important: If checkpoint was saved from LossWrapper
When the Trainer saves checkpoints, it saves the full `LossWrapper` state dict. The keys will have a `model.` prefix (e.g., `model.backbone.conv1.weight`). You need to strip this prefix when loading into the bare model.

## Step 2: HerdNetStitcher

Source: `animaloc/eval/stitchers.py`

The `HerdNetStitcher` handles overlapping tile inference on large images.

```python
from animaloc.eval.stitchers import HerdNetStitcher

stitcher = HerdNetStitcher(
    model=wrapped_model,         # LossWrapper-wrapped model
    size=(512, 512),             # Tile size (must match training patch size)
    overlap=120,                 # Overlap between tiles in pixels
    batch_size=1,                # Inference batch size per tile
    down_ratio=4,                # Must match model's down_ratio
    up=False,                    # Don't upsample -- stitcher handles it
    reduction='mean',            # Average overlapping regions
    device_name='cuda'
)
```

### Parameters Explained

**size**: Tile dimensions `(height, width)`. Should match the patch size used during training (typically 512x512).

**overlap**: Pixel overlap between adjacent tiles. 120px is recommended. Too little overlap risks missing detections at tile borders. Too much overlap increases computation time.

**down_ratio**: The stitcher needs to know the model's downsampling ratio to correctly compute output map sizes. Must match `model.down_ratio`.

**up**: Set to `False` when using `reduction='mean'`. The stitcher handles coordinate mapping internally. Setting `up=True` with `reduction='mean'` would apply bilinear upsampling after stitching, which is unnecessary and can introduce artifacts.

**reduction**: How overlapping regions are combined:
- `'mean'`: Average of overlapping predictions (recommended). Smooth, reduces border artifacts.
- `'sum'`: Sum of overlapping predictions. Can cause double-counting.
- `'max'`: Maximum of overlapping predictions. Preserves strong detections but can be noisy.

### Hann Window Weighting

The `HerdNetStitcher` uses 2D Hann (raised cosine) windows to weight each tile's contribution during stitching. This ensures:
- Center of each tile contributes most (weight = 1.0)
- Edges of tiles contribute less (weight tapers to ~0)
- Overlapping regions blend smoothly

The Hann window matrix is constructed differently for different tile positions:
- **Corner tiles**: Hann window applied on 2 edges
- **Edge tiles**: Hann window applied on 1 edge
- **Interior tiles**: Full Hann window on all sides

Source: `animaloc/eval/utils.py` -- `HannWindow2D` class

### Running the Stitcher

```python
import torchvision
from PIL import Image

# Load orthomosaic
image = Image.open('orthomosaic.tif').convert('RGB')
image_tensor = torchvision.transforms.ToTensor()(image)  # [C, H, W]

# Normalize (must match training normalization)
normalize = torchvision.transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
image_tensor = normalize(image_tensor)

# Run stitcher
with torch.no_grad():
    output = stitcher(image_tensor)
# output shape: [1, 1+num_classes-1, H/DR, W/DR]
# Channel 0: localization heatmap
# Channels 1..N: class scores (already softmax-ed by HerdNetStitcher)
```

### HerdNetStitcher._inference() Details

The `HerdNetStitcher` overrides the base `Stitcher._inference()` to handle HerdNet's two-headed output:
1. Gets `(heatmap, clsmap)` from the model
2. Upsamples `clsmap` to match `heatmap` resolution via nearest-neighbor interpolation
3. Concatenates: `[heatmap, clsmap]` along channel dimension
4. Returns concatenated maps for stitching

This means the stitched output contains both heatmap and class information in a single tensor.

## Step 3: HerdNetLMDS

Source: `animaloc/eval/lmds.py`

Local Maxima Detection and Suppression converts the stitched heatmap into discrete point detections.

```python
from animaloc.eval.lmds import HerdNetLMDS

lmds = HerdNetLMDS(
    up=False,                    # False when using with stitcher output
    kernel_size=(5, 5),          # OPTIMAL: (5,5), default was (3,3)
    adapt_ts=0.5,                # OPTIMAL: 0.5, default was 0.3
    neg_ts=0.1,                  # Negative sample threshold
    scale_factor=1               # 1 when post-stitcher (already at correct scale)
)
```

### Parameters Explained

**kernel_size**: Size of the local maximum search window. Larger kernels suppress more closely-spaced detections. `(5,5)` is optimal for iguana spacing; `(3,3)` may produce duplicate detections on large animals.

**adapt_ts**: Adaptive threshold relative to the global maximum in the heatmap. A peak must be `>= adapt_ts * max_value` to be retained. Higher values reduce false positives. `0.5` is optimal for iguanas.

**neg_ts**: If the global maximum of the heatmap is below `neg_ts`, the entire image is treated as a negative (no detections). Prevents spurious detections on empty images.

**up**: Controls whether the classification map is upsampled to match the heatmap resolution. Set to `False` when the input is already from the stitcher (which already handled upsampling). Set to `True` when using directly with model output.

**scale_factor**: Upsampling factor for the classification map. Set to `1` when post-stitcher.

### Running LMDS

```python
# Split stitched output into heatmap and class map
heatmap = output[:, :1, :, :]           # [1, 1, H/DR, W/DR]
clsmap = output[:, 1:, :, :]            # [1, C-1, H/DR, W/DR]

# Run LMDS
counts, locs, labels, cls_scores, det_scores = lmds([heatmap, clsmap])

# counts: list of per-class counts, e.g., [45, 3] for 45 iguanas, 3 hard negatives
# locs: list of (row, col) tuples in heatmap coordinates
# labels: list of class labels (1-indexed)
# cls_scores: classification confidence scores
# det_scores: detection confidence scores (from heatmap values)
```

### Converting LMDS Coordinates to Image Coordinates

LMDS returns coordinates in the heatmap coordinate system (downsampled by `down_ratio`). To get original image coordinates:

```python
down_ratio = 4
image_coords = []
for loc in locs[0]:  # First (only) batch
    row, col = loc
    img_y = int(row * down_ratio)
    img_x = int(col * down_ratio)
    image_coords.append((img_x, img_y))
```

## Step 4: Export Results

### Export to CSV
```python
import pandas as pd

results = []
for (row, col), label, cls_score, det_score in zip(
    locs[0], labels[0], cls_scores[0], det_scores[0]
):
    results.append({
        'x': int(col * down_ratio),
        'y': int(row * down_ratio),
        'label': label,
        'class_score': round(cls_score, 4),
        'detection_score': round(det_score, 4),
    })

df = pd.DataFrame(results)
df.to_csv('detections.csv', index=False)
```

### Overlay on Image
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches

fig, ax = plt.subplots(1, 1, figsize=(20, 20))
ax.imshow(image)

colors = {1: 'lime', 2: 'red'}  # iguana=green, hard_neg=red
for (row, col), label, score in zip(locs[0], labels[0], det_scores[0]):
    x = col * down_ratio
    y = row * down_ratio
    circle = plt.Circle((x, y), radius=10, color=colors.get(label, 'blue'),
                        fill=False, linewidth=2)
    ax.add_patch(circle)
    ax.text(x + 12, y, f'{score:.2f}', color=colors.get(label, 'blue'),
            fontsize=8)

ax.set_title(f'Detections: {sum(counts[0])} total')
plt.savefig('detections_overlay.png', dpi=150, bbox_inches='tight')
```

## Full Inference Script

```python
import torch
import torchvision
from PIL import Image
import pandas as pd
from animaloc.models import HerdNetTimmDLA
from animaloc.models.utils import LossWrapper
from animaloc.eval.stitchers import HerdNetStitcher
from animaloc.eval.lmds import HerdNetLMDS

# Config -- must match training
NUM_CLASSES = 3
DOWN_RATIO = 4
PATCH_SIZE = (512, 512)
OVERLAP = 120
CHECKPOINT = 'best_model.pth'
IMAGE_PATH = 'orthomosaic.tif'

# Device
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')

# Model
model = HerdNetTimmDLA(backbone='timm/dla34', num_classes=NUM_CLASSES,
                       down_ratio=DOWN_RATIO, head_conv=64,
                       pretrained=False, debug=False)
ckpt = torch.load(CHECKPOINT, map_location=device)
sd = ckpt.get('state_dict', ckpt)
sd = {k.replace('model.', '', 1): v for k, v in sd.items()}
model.load_state_dict(sd, strict=False)
model.to(device).eval()

# Wrap for stitcher (needs LossWrapper interface)
wrapped = LossWrapper(model, losses=[], mode='preds_only')

# Stitcher
stitcher = HerdNetStitcher(
    model=wrapped, size=PATCH_SIZE, overlap=OVERLAP,
    down_ratio=DOWN_RATIO, up=False, reduction='mean',
    device_name=str(device)
)

# Load and normalize image
img = Image.open(IMAGE_PATH).convert('RGB')
tensor = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])(img)

# Stitch
with torch.no_grad():
    output = stitcher(tensor)

# LMDS
heatmap = output[:, :1, :, :]
clsmap = output[:, 1:, :, :]
lmds = HerdNetLMDS(up=False, kernel_size=(5, 5), adapt_ts=0.5, neg_ts=0.1,
                   scale_factor=1)
counts, locs, labels, cls_scores, det_scores = lmds([heatmap, clsmap])

# Export
results = [{'x': int(c * DOWN_RATIO), 'y': int(r * DOWN_RATIO),
            'label': l, 'cls_score': round(s, 4), 'det_score': round(d, 4)}
           for (r, c), l, s, d in zip(locs[0], labels[0], cls_scores[0], det_scores[0])]
pd.DataFrame(results).to_csv('detections.csv', index=False)
print(f"Detected {sum(counts[0])} animals: {counts[0]}")
```

## Common Inference Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| Stitcher output is blank | `up=True` with `reduction='mean'` | Set `up=False` |
| Double-counting at borders | Overlap too small or `reduction='sum'` | Use `overlap=120`, `reduction='mean'` |
| No detections from LMDS | `adapt_ts` too high | Lower `adapt_ts` (start with 0.3) |
| Too many false positives | `adapt_ts` too low | Increase `adapt_ts` (try 0.5-0.7) |
| Wrong coordinates | Forgot down_ratio scaling | Multiply LMDS coords by `down_ratio` |
| State dict key mismatch | LossWrapper prefix | Strip `model.` prefix from keys |
| Memory error on large image | Image too large for single pass | Use stitcher (it processes tiles) |
