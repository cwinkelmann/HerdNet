# Patcher Usage Reference

## Overview

The patcher (`tools/patcher.py`) tiles large orthomosaics into training-size patches and transfers annotations to the tiled coordinate system. This is essential because HerdNet processes fixed-size patches (typically 512x512), but drone orthomosaics can be tens of thousands of pixels wide.

Source: `tools/patcher.py`

## Command-Line Usage

```bash
python tools/patcher.py \
    ROOT \            # Path to directory containing source images
    HEIGHT \          # Patch height in pixels (e.g., 512)
    WIDTH \           # Patch width in pixels (e.g., 512)
    OVERLAP \         # Overlap between adjacent patches in pixels (e.g., 120)
    DEST \            # Destination directory for patches
    [-csv CSV] \      # Optional: path to annotations CSV
    [-min MIN] \      # Optional: minimum visibility fraction (default 0.1)
    [-all ALL]        # Optional: save all patches, not just annotated (default False)
```

### Example

```bash
# Tile orthomosaics with annotation transfer
python tools/patcher.py \
    /data/orthomosaics/ \
    512 512 120 \
    /data/patches/ \
    -csv /data/annotations.csv \
    -all False

# Tile orthomosaics without annotations (just images)
python tools/patcher.py \
    /data/orthomosaics/ \
    512 512 120 \
    /data/patches/
```

## Parameters

### ROOT
Directory containing source orthomosaic images. All non-CSV files in this directory are treated as images.

### HEIGHT and WIDTH
Dimensions of output patches in pixels. Standard for HerdNet training is **512x512**.

Must match the `img_size` in the dataset config and the `size` parameter of `HerdNetStitcher` during inference.

### OVERLAP
Number of shared pixels between adjacent patches. Recommended: **120 pixels**.

Purpose:
- Ensures annotations near patch borders are captured by at least one patch
- Prevents missing detections at tile boundaries
- During inference, the stitcher uses the same overlap concept with Hann windowing

### DEST
Output directory. Created if it doesn't exist. Will contain:
- Patch images (JPEG)
- `gt.csv` (if `-csv` was specified)

### -csv
Path to a CSV file with annotations in CSVDataset format (`images, x, y, labels` for points or `images, x_min, y_min, x_max, y_max, labels` for boxes).

When provided, the patcher:
1. Uses `PatchesBuffer` to determine which annotations fall in each patch
2. Transforms annotation coordinates to patch-local coordinates
3. Writes `gt.csv` with transformed annotations

### -min
Minimum fraction of an annotation's area that must be visible in a patch for it to be kept. Default: 0.1 (10%).

Primarily relevant for **bounding box** annotations. For **point** annotations, a point is either inside the patch or not (binary decision).

### -all
- `False` (default): Only save patches that contain at least one annotation. Useful when you want annotated patches only.
- `True`: Save all patches, including empty ones. Useful when you need negative examples for training.

## How Annotation Transfer Works

### Point Annotations
For each patch with corners `(x_min, y_min, x_max, y_max)`:
- A point `(x, y)` is included if `x_min <= x < x_max` and `y_min <= y < y_max`
- The point's coordinates are transformed: `(x - x_min, y - y_min)`

### Box Annotations
For each patch:
- A box is included if it overlaps with the patch by at least `min` fraction of its area
- Box coordinates are clipped to patch boundaries
- Transformed to patch-local coordinates

## Output Structure

```
/data/patches/
    orthomosaic1_x0_y0.jpg              # Patch at position (0, 0)
    orthomosaic1_x392_y0.jpg            # Patch at position (392, 0), overlap=120
    orthomosaic1_x784_y0.jpg
    orthomosaic1_x0_y392.jpg
    orthomosaic1_x392_y392.jpg
    ...
    gt.csv                               # Transferred annotations
```

### Patch Naming Convention
Patches are named: `{original_name}_x{col}_y{row}.{ext}`

Where `x{col}` and `y{row}` indicate the top-left corner of the patch in the original image coordinate system.

### gt.csv Format
The output CSV has the same format as the input but with:
- `images`: patch filename (not original image)
- Coordinates transformed to patch-local system

```csv
images,x,y,labels
orthomosaic1_x0_y0.jpg,156,234,1
orthomosaic1_x0_y0.jpg,300,100,1
orthomosaic1_x392_y0.jpg,50,180,2
```

## Edge Patches

When the image dimensions are not evenly divisible by (patch_size - overlap), edge patches may be smaller than the full patch size. The patcher handles this by:
- Using `PadIfNeeded` from albumentations to pad edge patches to the target size
- Padding mode: `BORDER_CONSTANT` with value 0 (black padding)
- Pad position: `TOP_LEFT` (padding added to bottom/right)

This means edge patches may have black padding at the bottom or right edges. The model should be robust to this, but excessive padding can affect training quality.

## Data Splitting -- CRITICAL

### The Fundamental Rule
**All patches from the same source orthomosaic MUST go in the same data split** (train, val, or test).

### Why This Matters
Patches from the same orthomosaic share:
- Overlapping image content (due to overlap parameter)
- Same lighting conditions, camera angle, GSD
- Same background vegetation and terrain
- Potentially the same individual animals visible in multiple patches

If these patches are split across train and validation, the model "cheats" by recognizing familiar context, producing inflated validation metrics.

### Correct Workflow

```
Step 1: Assign orthomosaics to splits
  ortho_A, ortho_B, ortho_C -> train
  ortho_D -> validation
  ortho_E -> test

Step 2: Tile each split independently
  python tools/patcher.py train_orthos/ 512 512 120 train_patches/ -csv train_annos.csv
  python tools/patcher.py val_orthos/ 512 512 120 val_patches/ -csv val_annos.csv
  python tools/patcher.py test_orthos/ 512 512 120 test_patches/ -csv test_annos.csv

Step 3: Point each config to the correct split
  datasets:
    train:
      csv_file: train_patches/gt.csv
      root_dir: train_patches/
    validate:
      csv_file: val_patches/gt.csv
      root_dir: val_patches/
    test:
      csv_file: test_patches/gt.csv
      root_dir: test_patches/
```

### Wrong Workflow (Data Leakage)

```
# DO NOT DO THIS:
# 1. Tile all orthomosaics together
python tools/patcher.py all_orthos/ 512 512 120 all_patches/ -csv all_annos.csv

# 2. Randomly split the resulting patches
# This causes data leakage because adjacent patches share content!
```

## Overlap Selection Guide

| Overlap | Pros | Cons |
|---------|------|------|
| 0 | Fastest, least patches | Annotations at borders may be missed |
| 60 | Good balance | Some border issues remain |
| **120** | **Recommended** | **Annotations reliably captured** |
| 200 | Very safe | Many redundant patches, slower training |
| 256 | Half a patch | Excessive redundancy |

For inference, the `HerdNetStitcher` uses the same overlap concept. Using the same overlap for both tiling and inference ensures consistency.

## Integration with PatchesBuffer

The `PatchesBuffer` class (from `animaloc/data`) handles the annotation-to-patch mapping:

```python
from animaloc.data import PatchesBuffer

buffer = PatchesBuffer(
    csv_file='annotations.csv',
    root_dir='/path/to/images/',
    size=(512, 512),               # Patch dimensions
    overlap=120,                    # Overlap in pixels
    min_visibility=0.1             # Minimum visibility for boxes
)

# Access the buffer DataFrame
df = buffer.buffer
# Columns: images (patch name), base_images (source image), x, y, labels, limits
```

## Practical Tips

1. **Verify annotation counts**: After patching, the total annotations across all patches should roughly equal the original (some may be duplicated in overlapping regions, some edge annotations may be lost).

2. **Check for empty patches**: If `-all False`, only annotated patches are saved. This can create an unbalanced dataset. Consider adding some empty patches manually.

3. **Consistent patch size**: Use the same patch size throughout the pipeline (patcher, training, stitcher). Mismatched sizes cause silent errors.

4. **Large orthomosaics**: Very large orthomosaics (> 30,000 px per side) can take significant time and memory to tile. Consider processing one at a time.

5. **File format**: The patcher saves patches in the same format as the source. For training, JPEG is typically fine. For maximum quality, consider PNG.
