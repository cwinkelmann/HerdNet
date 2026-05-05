# Phase 1 — augmentation × crop-strategy ablation

**Run date**: 2026-05-04 (training start 21:02, finish 23:07).
**Branch / commits**: `convnext_extension` @ `48575f0` (after centre-bias fix, wandb_flag guard, and warning silence).
**Hardware**: single RTX 4080 SUPER (16 GiB), GPU shared with an unrelated COLMAP job.
**Script**: [`run_phase1.sh`](../../run_phase1.sh).

## Setup

Fixed: `HerdNetTimmConvNext` (backbone=`tiny`, 32.6 M params), `herdnet_fmo03` loss (Focal + weighted CE), 30 epochs, `lr=8e-5`, `batch_size=4`, seed=42, `valid_freq=2`.

Varied: dataset config only.

| Tag | Dataset config | Crop source | Augmentations |
|---|---|---|---|
| **A1** | `fmo03_new_clean` | pre-baked 512×512 patches (stride 250) | minimal (HFlip/VFlip/Rotate90, MotionBlur, Brightness, ShiftScale, HSV, CLAHE) |
| **A2** | `fmo03_new_clean_augplus` | same pre-baked patches | augplus (A1 + Perspective, Downscale/Blur/Sharpen, Plasma/RandomShadow, RGBShift, GaussNoise/ISONoise/Compress, CoarseDropout) |
| **A3** | `fmo03_new_objcrop_augplus` | full-size images + `ObjectAwareRandomCrop` | augplus (no pre-baked patches; crops sampled at every step with `min_edge_distance=0`, `empty_probability=0.05`, `edge_probability=0.2`, `edge_zone=120`) |

Per-epoch validation: `data_fmo03/val/herdnet_format_512_0_crops.csv` (98 cropped patches).
Final full-size validation: 12 full-size frames in `data_fmo03/val/Default/`, 181 GT iguanas, stitched at overlap=120.

A1 → A2 isolates the augmentation effect (same crop source).
A2 → A3 isolates the crop-strategy effect (same augmentations).

## Per-epoch metrics (small cropped val)

| Tag | best F1 | F2 | precision | recall | MAE |
|---|---|---|---|---|---|
| A1 | 0.9155 | 0.8920 | 0.9573 | 0.8771 | 0.19 |
| A2 | **0.9298** | 0.9044 | 0.9755 | 0.8883 | 0.18 |
| A3 | 0.9107 | 0.8937 | 0.9405 | 0.8827 | 0.21 |

Cropped-val ranking: A2 > A1 > A3. **This signal is misleading** — see below.

## Full-size stitched validation (HerdNetStitcher, overlap 120)

| Tag | F1 | precision | recall | MAE | RMSE | AP | detections |
|---|---|---|---|---|---|---|---|
| A1 | 0.7648 | 0.6350 | 0.9613 | 7.75 | 8.86 | 0.833 | — |
| A2 | 0.8000 | 0.7029 | 0.9282 | 4.83 | 5.64 | 0.851 | — |
| A3 | **0.9056** | **0.9106** | 0.9006 | **1.50** | **1.87** | **0.894** | 179 |

GT count: 181 iguanas across 12 frames.

## Effect sizes

- **Augmentation** (A1 → A2, both pre-cropped): F1 +0.035, MAE −38 %, RMSE −36 %. Modest but real.
- **Crop strategy** (A2 → A3, both augplus): F1 +0.106, MAE −69 %, RMSE −67 %, precision +0.21, recall −0.03. Dominant lever.
- **Combined** (A1 → A3): F1 +0.141, MAE −81 % (7.75 → 1.50), AP +0.06.

A3's MAE of 1.5 means the average per-frame count error is about one and a half iguanas on frames that have ~15 each. A1's MAE of 7.75 is unusable for census.

## Why per-epoch metrics lie here

The per-epoch val set is 98 pre-cut 512×512 patches. The trained model only ever sees one tile at a time, and a model that overfits to the patch grid will look great on those patches and bad on the original frames once `HerdNetStitcher` re-tiles them with stride 392 and merges via local-max suppression.

A3's `ObjectAwareRandomCrop` deliberately samples crops with keypoints anywhere in the tile (not always centred), with 5 % pure-background and 20 % edge-zone crops. That's the same distribution the stitcher produces at inference time — so A3's training distribution matches its evaluation distribution, while A1/A2 train on a fixed patch grid that the inference tiling doesn't reproduce.

This same pattern showed up earlier today on DLA-34 (full-size F1: ObjCrop 0.833 vs pre-cropped 0.702, MAE 3.67 vs 11.83). Phase 1 confirms it generalises to ConvNeXt and isolates the augmentation vs crop contributions.

## Recommendations

1. **Lock `fmo03_new_objcrop_augplus` for downstream phases.** The pre-cropped variants underperform on the only metric that reflects real inference behaviour.
2. **Don't ship per-epoch F1 as a model selection signal** in this codebase. Use it for convergence tracking only; gate selection on the full-size eval.
3. **Keep the augplus pipeline.** It contributes a smaller but consistent gain and should be the default.

## Artifacts

- Best checkpoints: `output/phase1_{A1,A2,A3}_convnext_baseline/2026-05-04/<HH-MM-SS>/best_model.pth`
- Full-size metrics CSVs: `output/phase1_fullval_20260504_210213/{A1,A2,A3}/metrics_results.csv`
- Per-run training logs: `/tmp/phase1_20260504_210213/{1_A1,2_A2,3_A3}_{train,fullval}.log`
- Wrapper log: `/tmp/phase1_20260504_210213_wrapper.log`
- WandB project: `hn_phase1` (runs `phase1_A1_*`, `phase1_A2_*`, `phase1_A3_*`)
