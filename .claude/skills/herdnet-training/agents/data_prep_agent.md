# Data Preparation Agent

## Role
Guide users through preparing their data for HerdNet training: CSV annotation format, tiling large orthomosaics with `patcher.py`, annotation validation, and data splitting best practices.

## CSVDataset Format

The `CSVDataset` class (in `animaloc/datasets/csv.py`) auto-detects annotation type from CSV column headers:

### Point Annotations (used for HerdNet)
```
images,x,y,labels
patch_001.jpg,234,156,1
patch_001.jpg,412,89,1
patch_001.jpg,300,400,2
patch_002.jpg,0,0,0
```

### Box Annotations (used for Faster R-CNN)
```
images,x_min,y_min,x_max,y_max,labels
patch_001.jpg,200,130,268,182,1
```

### Label Rules
- Labels are **1-indexed integers**. Background is implicitly 0.
- For iguana detection with hard negatives: `1 = iguana_point`, `2 = hard_negative`
- `num_classes` in the model config **includes background**. So binary detection = `num_classes=2`, binary + hard_neg = `num_classes=3`.
- Empty images (no annotations) should have a single row with `x=0, y=0, labels=0`.

### Multiple Annotations Per Image
Multiple rows with the same `images` value represent multiple annotations on the same image. The CSVDataset groups by image name internally via `AnnotationsFromCSV`.

### Images Column
The `images` column contains the **relative path from `root_dir`**. Example:
```python
CSVDataset(
    csv_file='annotations.csv',
    root_dir='/data/patches/'
)
# images column: 'patch_001.jpg' -> loads '/data/patches/patch_001.jpg'
```

## Tiling with patcher.py

Source: `tools/patcher.py`

The patcher cuts large orthomosaics into training-size patches and transfers annotations.

### Command-Line Usage
```bash
python tools/patcher.py \
    /path/to/orthomosaic/images \   # root: directory containing source images
    512 \                            # height: patch height in pixels
    512 \                            # width: patch width in pixels
    120 \                            # overlap: overlap between patches in pixels
    /path/to/output/patches \        # dest: output directory
    -csv /path/to/annotations.csv \  # optional: CSV with annotations to transfer
    -min 0.1 \                       # optional: min fraction of annotation area to keep (for boxes)
    -all False                       # optional: True to save all patches, not only annotated ones
```

### How It Works
1. Reads each image from `root` directory
2. Uses `ImageToPatches` to split into overlapping tiles of size `(height, width)` with specified overlap
3. If `-csv` is provided, uses `PatchesBuffer` to:
   - Determine which annotations fall in each patch
   - Clip annotations to patch boundaries
   - Apply minimum visibility filter (for bounding boxes)
4. Saves patches as images and exports transferred annotations to `gt.csv` in `dest`

### Output Structure
```
/path/to/output/patches/
    orthomosaic1_x0_y0.jpg
    orthomosaic1_x392_y0.jpg
    orthomosaic1_x0_y392.jpg
    ...
    gt.csv    # transferred annotations
```

### Key Parameters
- **patch_size (height, width)**: 512x512 is standard for HerdNet training
- **overlap**: 120px recommended. Ensures no annotations are missed at tile borders during training. During inference, the `HerdNetStitcher` handles overlap with Hann windowing.
- **-min**: Only relevant for bounding box annotations. For point annotations, a point is either inside the patch or not.
- **-all**: Set to `True` to include patches with no annotations (useful for training with negatives).

## Data Splitting -- CRITICAL

### The Data Leakage Problem
When tiling an orthomosaic into patches, **all patches from the same source orthomosaic MUST go in the same split** (train, val, or test). If patches from the same orthomosaic appear in both training and validation sets, the model will see near-duplicate data, leading to inflated validation metrics that do not reflect real-world performance.

### Correct Splitting Procedure
1. List all unique source orthomosaics
2. Assign each orthomosaic to train, val, or test (e.g., 70/15/15)
3. Tile each orthomosaic
4. All patches from a train orthomosaic go to training set, etc.

### Recommended Split Sizes
- Training: 70-80% of source images
- Validation: 10-15% of source images
- Test: 10-15% of source images (held out entirely until final evaluation)

## ObjectAwareRandomCrop

Used as a training-time augmentation in the `albu_transforms` list:
```yaml
albu_transforms:
  ObjectAwareRandomCrop:
    height: 512
    width: 512
    p: 1.0
```

This augmentation randomly crops a 512x512 region from the input patch while ensuring that at least some annotations remain within the cropped area. This is important because random cropping could otherwise produce many empty patches.

## Annotation Validation Checklist

Before training, validate your CSV annotations:

1. **Column names match exactly**: `images,x,y,labels` (case-sensitive, no spaces)
2. **Labels are 1-indexed**: Background is implicit (0), species start at 1
3. **Coordinates are within image bounds**: `0 <= x < image_width`, `0 <= y < image_height`
4. **No duplicate annotations**: Same (x, y) point should not appear twice for the same image
5. **Labels match num_classes**: If `num_classes=3`, valid labels are 0, 1, 2
6. **Image files exist**: Every path in the `images` column must resolve to a file in `root_dir`
7. **Images are RGB**: CSVDataset calls `.convert('RGB')` but source images should be valid

### Validation Script
```python
import pandas as pd
from PIL import Image
import os

def validate_annotations(csv_path, root_dir, num_classes):
    df = pd.read_csv(csv_path)

    # Check columns
    required = {'images', 'x', 'y', 'labels'}
    assert required.issubset(df.columns), f"Missing columns: {required - set(df.columns)}"

    errors = []
    for img_name in df['images'].unique():
        img_path = os.path.join(root_dir, img_name)
        if not os.path.exists(img_path):
            errors.append(f"Missing image: {img_path}")
            continue

        img = Image.open(img_path)
        w, h = img.size

        img_annos = df[df['images'] == img_name]
        for _, row in img_annos.iterrows():
            if row['labels'] == 0:
                continue  # empty image marker
            if not (0 <= row['x'] < w):
                errors.append(f"{img_name}: x={row['x']} out of bounds (width={w})")
            if not (0 <= row['y'] < h):
                errors.append(f"{img_name}: y={row['y']} out of bounds (height={h})")
            if row['labels'] < 0 or row['labels'] >= num_classes:
                errors.append(f"{img_name}: label={row['labels']} invalid for num_classes={num_classes}")

    return errors
```

## Annotation Type: Body-Center vs Head

The Miesner 2025 thesis found that **body-center annotations outperform head annotations by approximately F1 = +0.10**. When creating new annotations, instruct annotators to click the center of the animal's body, not the head.

## Common Data Preparation Errors

| Error | Symptom | Fix |
|-------|---------|-----|
| Labels are 0-indexed | Model trains but F1 is wrong | Shift all labels by +1 |
| num_classes wrong | CrossEntropyLoss dimension mismatch | Count: background + all species |
| Data leakage across splits | Inflated validation metrics | Split by source orthomosaic, not patch |
| Coordinates out of bounds | Training crashes on transform | Validate all annotations before training |
| Missing empty image rows | Model never sees negatives | Add rows with x=0, y=0, labels=0 for empty images |
| Wrong CSV separator | Columns not parsed | Use comma separator, not semicolon or tab |
