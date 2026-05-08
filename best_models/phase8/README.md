# Phase 8 production checkpoints — wrapped after benchmark Phases 1–12

Six checkpoints that make up the production stack as of Phase 8 (cross-architecture ensemble, F1 = 0.972, MAE = 0.50). Saved here so the production recipe is reproducible without dipping back into `output/`.

The files are **symlinks** to the canonical locations in `output/` — no disk duplication. If you ever clean `output/`, dereference these into real copies first (`cp -L`).

## Single-checkpoint metrics (full-size stitched val, best-tuned per checkpoint)

| Checkpoint | Architecture | Seed | best F1 @ ts | best MAE @ ts | Source |
|---|---|---|---|---|---|
| **b4_seed42** ⭐ | ConvNeXt-T + BiFPN + DeformConv | 42 | 0.955 @ 0.50 | **0.75** @ 0.35 | Phase 3 |
| b3_seed123 | ConvNeXt-V2-T + Gabor | 123 | **0.958** @ 0.40 | 1.17 @ 0.50 | Phase 5 |
| b3_seed42 | ConvNeXt-V2-T + Gabor | 42 | 0.952 @ 0.40 | 1.00 @ 0.35 | Phase 3 |
| b4_seed123 | ConvNeXt-T + BiFPN + DeformConv | 123 | 0.942 @ 0.70 | 0.83 @ 0.35 | Phase 5 |
| b4_seed7 | ConvNeXt-T + BiFPN + DeformConv | 7 | 0.938 @ 0.60 | 1.08 @ 0.35 | Phase 5 |
| b3_seed7 | ConvNeXt-V2-T + Gabor | 7 | 0.935 @ 0.60 | 1.17 @ 0.50 | Phase 5 |

⭐ **Designated warm-start checkpoint** for the [data-scaling experiment](../../docs/data_scaling_experiment.md). Has the lowest single-checkpoint MAE (0.75) — the metric the scaling experiment cares about.

## Useful combinations

The 6 checkpoints support 4 different ensemble configurations, each documented in [Phase 6](../../docs/benchmarks/phase6_inference_tricks.md) and [Phase 8](../../docs/benchmarks/phase8_cross_arch_ensemble.md).

| Combination | Members | best F1 | best MAE | Use when |
|---|---|---|---|---|
| **Cross-arch (Phase 8 production)** | all 6 | 0.972 | 0.50 | F1-optimal counting; default deployment |
| B4×3 (Phase 6 same-arch) | b4_seed{42,123,7} | 0.963 | 0.67 | If only B4 is available; same-arch ensemble |
| B3×3 (same-arch) | b3_seed{42,123,7} | not separately tested | — | Reserved for cross-arch follow-ups |
| Single best (warm-start) | b4_seed42 | 0.955 | 0.75 | Phase 13 data-scaling baseline |

## Loading

Each checkpoint contains the full Hydra config it was trained with (under the `config` key) plus a `training_info` dict (added in Phase 6) — `tools/ensemble_infer.py` reads these to instantiate the right model class per member, so cross-architecture ensembling Just Works.

```bash
# Single-model inference
python tools/infer.py \
  --config-dir configs/demo \
  --config-name fmo03_full_v2_bifpn \
  --images data_fmo03/val/Default \
  --model best_models/phase8/b4_seed42/best_model.pth \
  --evaluate

# Production cross-arch ensemble
python tools/ensemble_infer.py \
  --config-dir configs/demo \
  --config-name fmo03_full_v2_bifpn \
  --images data_fmo03/val/Default \
  --models best_models/phase8/b3_seed42/best_model.pth \
           best_models/phase8/b3_seed123/best_model.pth \
           best_models/phase8/b3_seed7/best_model.pth \
           best_models/phase8/b4_seed42/best_model.pth \
           best_models/phase8/b4_seed123/best_model.pth \
           best_models/phase8/b4_seed7/best_model.pth \
  --evaluate \
  --overrides "training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=0.25"
```

## Configs

All 6 checkpoints were trained with:

- **Dataset**: `fmo03_new_objcrop_augplus` (augplus pipeline + ObjectAwareRandomCrop on full-size source images)
- **Loss**: `herdnet_fmo03` (Focal + weighted CE) — *override* on B3/B4 from their native `density_aware*` configs (Phase 5 found this combination wins)
- **Epochs**: 30
- **Batch size**: 4
- **lr**: 8e-5

Training launch lines for reproduction are recorded in `run_phase3.sh` (seed=42) and `run_seed_replication.sh` (seeds 123 and 7).

## Phase 8 production operating points

| Goal | Ensemble | adapt_ts | F1 | precision | recall | MAE |
|---|---|---|---|---|---|---|
| Counting (lowest MAE) | cross-arch (6 models) | 0.25 | 0.967 | 0.972 | 0.961 | **0.50** |
| F1-optimal | cross-arch + TTA | 0.30 | **0.972** | 0.994 | 0.950 | 0.83 |
| Zero-FP screening | cross-arch | 0.50 | 0.960 | **1.000** | 0.923 | 1.17 |

For all metrics: full-size stitched val, 12 frames, 181 GT iguanas. See [Phase 8 doc](../../docs/benchmarks/phase8_cross_arch_ensemble.md) for the full sweep.
