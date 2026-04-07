# FIDT Maps Reference

## Focal Inverse Distance Transform

The FIDT (Focal Inverse Distance Transform) converts sparse point annotations into continuous heatmaps suitable for training convolutional neural networks. Originally proposed by Liang et al. (2021) for crowd localization and counting.

Source: `animaloc/data/transforms.py` -- `FIDT` class (lines 289-425)

## Mathematical Formula

Given a binary mask where annotation points are marked as 0 and background as 1:

```
d = distance_transform_edt(mask)        # Euclidean distance to nearest annotation
fidt(d) = 1 / (d^(alpha * d + beta) + c)
```

Where:
- `d` = Euclidean distance from each pixel to the nearest annotation point
- `alpha` = 0.02 (controls how quickly peak sharpness increases with distance)
- `beta` = 0.75 (base exponent, controls overall peak width)
- `c` = 1.0 (prevents division by zero, shifts the baseline)

### Parameter Effects

| Parameter | Lower Value | Higher Value |
|-----------|------------|-------------|
| alpha | Broader, more uniform peaks | Sharper peaks that drop off faster |
| beta | Wider spread around each point | Narrower, more focused peaks |
| c | Higher peak values (approaches 1.0) | Lower peak values, more compressed |

### Comparison with Gaussian Heatmaps

| Property | FIDT | Gaussian |
|----------|------|----------|
| Peak shape | Sharp, focal | Smooth, bell-shaped |
| Crowded scenes | Better separation of nearby peaks | Peaks merge when close |
| Background | Near-zero far from annotations | Non-zero tail extends far |
| Parameter sensitivity | alpha, beta interact | Single sigma parameter |
| Computational cost | Requires distance transform | Direct convolution |

FIDT peaks are sharper than Gaussian peaks, which makes them better for distinguishing closely-spaced animals (e.g., iguanas basking in groups).

## Implementation Details

### FIDT Class Constructor

```python
FIDT(
    alpha: float = 0.02,       # Distance exponent multiplier
    beta: float = 0.75,        # Base exponent
    c: float = 1.0,            # Baseline constant
    radius: int = 1,           # Pixel radius for point annotation in mask
    num_classes: int = 2,      # Background included. 2=binary, >2=multi-class
    add_bg: bool = False,      # Add explicit background channel
    down_ratio: int = None     # Downsample output by this factor
)
```

### How It Works Step by Step

1. **Input**: Image tensor `[C, H, W]` and target dict with `'points'` and `'labels'`

2. **Downsampling** (if `down_ratio` is set):
   ```python
   img_height = img_height // down_ratio
   img_width = img_width // down_ratio
   # Points are also downsampled via DownSample transform
   ```

3. **Create binary mask**: For each class, create a `[H, W]` mask filled with 1s, then set 0 at each annotation point (with buffer of `radius` pixels)

4. **Distance transform**: Apply `scipy.ndimage.distance_transform_edt()` to the binary mask. This computes the Euclidean distance from each pixel to the nearest 0 (annotation).

5. **Apply FIDT formula**:
   ```python
   dist_map = 1 / (torch.pow(dist_map, alpha * dist_map + beta) + c)
   dist_map = torch.where(dist_map < 0.01, 0., dist_map)  # Zero out noise
   ```

6. **Output**: Tensor of shape `[num_classes-1, H, W]` (or `[num_classes, H, W]` if `add_bg=True`)

### Multi-Class Handling

- `num_classes=2` (binary): Single-channel output `[1, H, W]` regardless of label values
- `num_classes>2` (multi-class): One channel per foreground class `[C-1, H, W]`
  - Channel 0 = class 1 FIDT map
  - Channel 1 = class 2 FIDT map
  - etc.
- Internal `self.num_classes` is set to `num_classes - 1` (background is excluded)

### Background Channel

When `add_bg=True`:
```python
background = 1.0 - sum(all_foreground_channels)
output = concat([background, foreground_channels])  # [C, H, W]
```

## down_ratio Interaction

The `down_ratio` parameter in FIDT **must match** the model's `down_ratio`:
- Model with `down_ratio=4` produces heatmap at 1/4 resolution
- FIDT with `down_ratio=4` produces training targets at 1/4 resolution
- Loss function compares model output to FIDT target -- dimensions must match

In Hydra config, use interpolation to enforce consistency:
```yaml
end_transforms:
  FIDT:
    down_ratio: ${model.kwargs.down_ratio}
```

## Visualization Code

```python
import torch
import matplotlib.pyplot as plt
from animaloc.data.transforms import FIDT

# Create sample data
image = torch.rand(3, 512, 512)
target = {
    'points': [(100, 150), (250, 200), (300, 350), (400, 100)],
    'labels': [1, 1, 1, 1]
}

# Generate FIDT map
fidt = FIDT(alpha=0.02, beta=0.75, c=1.0, num_classes=2, down_ratio=4)
_, fidt_map = fidt(image, target)

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Original image with annotation points
axes[0].imshow(image.permute(1, 2, 0))
for pt in target['points']:
    axes[0].plot(pt[0], pt[1], 'r+', markersize=15, markeredgewidth=2)
axes[0].set_title('Image with Annotations')

# FIDT heatmap
axes[1].imshow(fidt_map[0].numpy(), cmap='hot', interpolation='nearest')
axes[1].set_title(f'FIDT Map (DR={fidt.down_ratio})')
axes[1].set_xlabel(f'Resolution: {fidt_map.shape[1]}x{fidt_map.shape[2]}')

plt.tight_layout()
plt.savefig('fidt_visualization.png', dpi=150)
```

## FIDT vs Training Loss

The FocalLoss is applied to the model's heatmap output (post-Sigmoid) against the FIDT target:
- Peak locations (annotation points) have high FIDT values (~1.0) -- model should predict high confidence here
- Background has FIDT value = 0.0 -- model should predict low confidence
- The `beta` parameter in FocalLoss (not FIDT's beta) controls how much the loss focuses on hard examples (misclassified pixels)

## Key References

- Liang, D., Chen, X., Xu, W., Zhou, Y., & Bai, X. (2021). Focal Inverse Distance Transform Maps for Crowd Localization and Counting in Dense Crowd. *arXiv:2102.07925*
- Delplanque, A., Foucher, S., Lejeune, P., & Yiwen, Z. (2022). From crowd to herd counting: How to precisely detect and count African mammals using aerial imagery and deep learning. *ISPRS Journal of Photogrammetry and Remote Sensing*
