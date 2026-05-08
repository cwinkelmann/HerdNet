# Best Models — FMO03

> **2026-05-08 update**: the production stack is the **Phase 8 cross-architecture ensemble** of 6 checkpoints, wrapped at [`phase8/`](phase8/README.md). F1 = 0.972, MAE = 0.50 on full-size stitched val. Use those for any new work.
>
> The earlier `fmo03_new_*` checkpoints below (from the April 2026 9-run study) reflect the codebase before Phase 1's findings — at that time the per-epoch ranking suggested "ObjCrop underperforms by 10–25 F1 points", which Phase 1's full-size eval falsified. They are kept here for reproducibility and historical reference but should **not** be used as a production baseline. See [`phase8/`](phase8/) for the current best.

---

# Best Models — FMO03 New Dataset (2026-04-07)

9-run experiment comparing 3 architectures × 3 dataset strategies on the FMO03 iguana detection dataset.

## Results

| Model | Clean (crops only) | Full (+ hard negatives) | ObjCrop (full-size) |
|-------|-------------------|------------------------|---------------------|
| **ConvNeXt Tiny** | **0.9425** (ep22) | 0.9307 (ep18) | 0.7988 (ep4) |
| **Hybrid** | 0.9329 (ep28) | 0.9326 (ep16) | 0.7784 (ep8) |
| **DLA34** | 0.8564 (ep18) | 0.8691 (ep28) | 0.6276 (ep6) |

## Dataset

- **Train:** 19 FMO03 images → 1430 annotated crops (512x512, overlap=250), 2682 annotations
- **Val:** 12 FMO05 images → 196 crops (512x512, overlap=0), 179 annotations
- **Source:** `/data/mnt/storage/Iguanas_From_Above/training_data/2026_04_07/2026_04_07_FMO03_02_05/`

## Dataset Strategies

- **Clean:** Only annotated crops (1430 images, 2 classes: background + iguana)
- **Full:** All crops including empty ones as hard negatives (2860 images, 3 classes: background + iguana + hard_negative)
- **ObjCrop:** Full-size images with `ObjectAwareRandomCrop(512, 512)` and `augmentation_multiplier: 75`

## Directory Structure

Each subdirectory contains:
- `best_model.pth` — Best checkpoint by F1 score (includes model state dict + config)
- `config.yaml` — Full resolved Hydra config used for training
- `overrides.yaml` — CLI overrides (if any)

## Reproduction

```bash
# Retrain any model:
python tools/train.py --config-path ../configs/demo --config-name <config_name>

# Available config names:
#   fmo03_new_clean_convnext    fmo03_new_full_convnext    fmo03_new_objcrop_convnext
#   fmo03_new_clean_dla34       fmo03_new_full_dla34       fmo03_new_objcrop_dla34
#   fmo03_new_clean_hybrid      fmo03_new_full_hybrid      fmo03_new_objcrop_hybrid
```

## Loading a Model

```python
import torch
import animaloc

checkpoint = torch.load('best_models/fmo03_new_clean_convnext/best_model.pth', map_location='cpu', weights_only=False)
cfg = checkpoint['config']

model = animaloc.models.__dict__[cfg['model']['name']](**cfg['model'].get('kwargs', {}))
model.reshape_classes(cfg['datasets']['num_classes'])

state_dict = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
              for k, v in checkpoint['model_state_dict'].items()}
model.load_state_dict(state_dict, strict=False)
model.eval()
```

## Key Findings

1. **ConvNeXt clean is the best** — simple pre-cropped patches without hard negatives
2. **Hard negatives don't help strong models** — slight regression for ConvNeXt, marginal help for DLA34
3. **ObjectAwareRandomCrop underperforms** by 10-25 F1 points — too few source images (19) for sufficient diversity
4. **DLA34 is unstable** — F1 oscillates wildly between validation epochs
