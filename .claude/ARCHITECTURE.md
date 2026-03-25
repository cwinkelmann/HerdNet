# Architecture

## Package Structure (`animaloc/`)

- **`models/`** — Model architectures. All models register via `@MODELS.register()` decorator and can be instantiated by name from config. Key variants: HerdNet (original DLA), HerdNetTimmDLA, HerdNetTimmConvNeXt, CamouflageHerdNetConvNeXt, HerdNetDINOv2, HerdNetTimmDINOv3, and more.
- **`datasets/`** — Dataset classes (CSVDataset is primary). Registered via `@DATASETS.register()`.
- **`train/`** — `Trainer` class (main training loop), `LossWrapper`, and loss functions (FocalLoss, P2PL, TverskyLoss, SSIMLoss). Losses registered via `@LOSSES.register()`.
- **`eval/`** — `HerdNetEvaluator` (validation during training), `HerdNetStitcher` (tiled inference on large images with overlap), `PointsMetrics` (F1, MAE, RMSE), LMDS (local maxima detection).
- **`data/`** — Transforms (FIDT, PointsToMask, DownSample, MultiTransformsWrapper), augmentation pipeline via Albumentations.
- **`utils/`** — `train.main(cfg)` is the main training entry point called by `tools/train.py`. Also contains inference, seeding, and logging utilities.
- **`vizual/`** — Plotting and visualization helpers.

## Key Design Patterns

**Registry pattern**: Models, datasets, and losses all use `Registry` (from `animaloc.utils.registry`) for string-based instantiation from Hydra configs. Register with `@MODELS.register()`, instantiate with `MODELS[name](**kwargs)`.

**Hydra configuration**: Configs are modular YAML files composed via `defaults` lists. A typical config composes `datasets/`, `model/`, `losses/`, and `training_settings/` sub-configs. Override any value from the command line.

**Dual-head output**: HerdNet models produce two outputs — a localization heatmap (sigmoid-activated, where objects are) and a classification map (class logits). `LossWrapper` maps each output head to its loss via `idx`/`idy` indices.

**Tiled inference**: `HerdNetStitcher` handles large aerial images by splitting into overlapping patches, running inference on each, and stitching results with overlap averaging to reduce boundary artifacts.

## Dataset Format

CSV with header `images,x,y,labels` (points) or `images,x_min,y_min,x_max,y_max,labels` (bounding boxes). One row per annotation; an image with N objects spans N rows.

## Training Pipeline Flow

```
Hydra config
  -> tools/train.py
  -> animaloc.utils.train.main(cfg)
  -> builds datasets, DataLoaders, model+LossWrapper, optimizer
  -> HerdNetEvaluator with HerdNetStitcher
  -> Trainer.start() with warmup, checkpointing on best metric (typically f1_score)
```

## Experiment Tracking

Uses Weights & Biases (wandb). Configured in YAML via `wandb_flag`, `wandb_project`, `wandb_entity`, `wandb_run`, `wandb_tags`. Wandb is an optional dependency — install with `pip install -e ".[tracking]"`.
