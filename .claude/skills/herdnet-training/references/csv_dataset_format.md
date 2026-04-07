# CSVDataset Format Reference

## Overview

The `CSVDataset` class provides a PyTorch Dataset that loads images and annotations from CSV files. It auto-detects the annotation type from column headers and handles data augmentation via albumentations.

Source: `animaloc/datasets/csv.py` -- `CSVDataset` class

## Annotation Types

CSVDataset automatically detects the annotation type from CSV column headers:

### Point Annotations
**Required columns**: `images, x, y, labels`

```csv
images,x,y,labels
patch_001.jpg,234,156,1
patch_001.jpg,412,89,1
patch_001.jpg,300,400,2
patch_002.jpg,0,0,0
```

- Used for HerdNet (FIDT-based detection)
- Albumentations uses `KeypointParams(format='xy')` for augmentation

### Bounding Box Annotations
**Required columns**: `images, x_min, y_min, x_max, y_max, labels`

```csv
images,x_min,y_min,x_max,y_max,labels
patch_001.jpg,200,130,268,182,1
patch_001.jpg,350,200,420,260,1
```

- Used for Faster R-CNN models
- Albumentations uses `BboxParams(format='pascal_voc')` for augmentation

### Segmentation Mask Annotations
**Required columns**: `images, mask_path, labels` or `images, masks, labels`

```csv
images,mask_path,labels
image_001.jpg,masks/mask_001.png,1
```

- Used for segmentation models
- Masks are loaded as grayscale images where pixel values = class indices

## Label Convention

### 1-Indexed Labels
Labels in the CSV must be **1-indexed integers**:
- `0` = background (implicit, used only to mark empty images)
- `1` = first species (e.g., iguana)
- `2` = second species/class (e.g., hard negative)
- etc.

### num_classes
The `num_classes` parameter throughout the pipeline **includes background**:
- Binary detection: `num_classes = 2` (background + 1 species)
- Binary with hard negatives: `num_classes = 3` (background + iguana + hard_negative)
- Multi-species: `num_classes = N + 1` (background + N species)

This is the single most common configuration error in HerdNet.

### Class Definition in Config
```yaml
num_classes: 3
class_def:
  1: 'iguana_point'
  2: 'hard_negative'
```

### CE Loss Weight Alignment
The `CrossEntropyLoss.weight` list must have exactly `num_classes` entries:
```yaml
weight:
  - 0.1    # index 0: background weight
  - 5.0    # index 1: iguana weight
  - 0.1    # index 2: hard_negative weight
```

## Empty Images

Images with no annotations should have a single row with zero coordinates:

```csv
images,x,y,labels
empty_patch.jpg,0,0,0
```

This tells the dataset that the image exists but contains no objects. Important for training -- the model needs to see negative examples.

## Multiple Annotations Per Image

Multiple annotations on the same image appear as multiple rows:

```csv
images,x,y,labels
patch_001.jpg,100,200,1
patch_001.jpg,300,150,1
patch_001.jpg,450,380,2
```

The CSVDataset groups annotations by the `images` column internally. Each unique value in `images` represents one training sample.

## Images Column

The `images` column contains **relative paths from `root_dir`**:

```python
dataset = CSVDataset(
    csv_file='/data/annotations/train.csv',
    root_dir='/data/images/train/'
)
# CSV entry: 'subfolder/patch_001.jpg'
# Loaded from: '/data/images/train/subfolder/patch_001.jpg'
```

## Constructor

```python
CSVDataset(
    csv_file: str,                        # Path to CSV file (or pandas DataFrame)
    root_dir: str,                        # Root directory for images
    albu_transforms: list = None,         # Albumentations transform list
    end_transforms: list = None,          # Post-albumentations transforms (FIDT, etc.)
    augmentation_multiplier: int = 1      # How many times to multiply dataset size
)
```

### augmentation_multiplier
Multiplies the effective dataset size for data augmentation:
```python
def __len__(self):
    return len(self._img_names) * self.augmentation_multiplier
```

When `augmentation_multiplier=3`, each image appears 3 times per epoch with different random augmentations (if stochastic transforms are configured).

## Transform Pipeline

### Execution Order
1. Load image (PIL.Image, converted to RGB)
2. Load target (dict with annotations)
3. Apply `albu_transforms` (albumentations, with keypoint/bbox awareness)
4. Convert to tensor via `SampleToTensor`
5. Apply `end_transforms` (FIDT, PointsToMask, DownSample)

### albu_transforms Example
```yaml
albu_transforms:
  ObjectAwareRandomCrop:
    height: 512
    width: 512
    p: 1.0
  HorizontalFlip:
    p: 0.5
  VerticalFlip:
    p: 0.5
  RandomBrightnessContrast:
    brightness_limit: 0.2
    contrast_limit: 0.2
    p: 0.3
  Normalize:
    p: 1.0
```

Albumentations automatically handles keypoint/bbox coordinate transformations during spatial augmentations.

### end_transforms Example
```yaml
end_transforms:
  MultiTransformsWrapper:
    FIDT:
      num_classes: ${datasets.num_classes}
      down_ratio: ${model.kwargs.down_ratio}
      radius: 2
    PointsToMask:
      radius: 2
      num_classes: ${datasets.num_classes}
      squeeze: true
      down_ratio: 32
```

## Dataset Output Format

`dataset[i]` returns `(image_tensor, target_dict)`:

```python
image, target = dataset[0]
# image: torch.Tensor [C, H, W], normalized
# target: dict with keys depending on end_transforms

# After FIDT + PointsToMask:
# target[0] = FIDT map [num_classes-1, H/DR, W/DR]
# target[1] = class mask [H/32, W/32]
```

## Common Issues

| Issue | Symptom | Solution |
|-------|---------|----------|
| Wrong column names | `KeyError` or wrong annotation type detected | Use exact headers: `images,x,y,labels` |
| 0-indexed labels | Wrong class mapping, silent errors | Use 1-indexed labels (1, 2, 3...) |
| Spaces in column names | Columns not matched | No spaces: `images`, not ` images` |
| num_classes off by 1 | CE loss dimension mismatch | Count: 1 (bg) + N (species) |
| Missing image file | FileNotFoundError | Check root_dir + images path |
| Coordinates out of bounds | Augmentation crash | Validate all x < width, y < height |
| No empty image rows | Model never sees negatives | Add rows: x=0, y=0, labels=0 |
| Tab/semicolon separator | Parsing failure | Use comma separator |
| Non-RGB images | Unexpected channel count | CSVDataset calls `.convert('RGB')` but check source |

## Data Leakage Warning

When creating train/val/test splits from tiled orthomosaics:

**WRONG**: Randomly split patches across sets
- Patches from the same orthomosaic can appear in both train and val
- Overlapping context means the model has seen near-identical data
- Inflated validation metrics

**CORRECT**: Split by source orthomosaic, then tile
- All patches from orthomosaic A go to training
- All patches from orthomosaic B go to validation
- No shared visual context between sets
