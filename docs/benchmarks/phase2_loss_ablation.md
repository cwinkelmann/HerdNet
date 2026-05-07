# Phase 2 — loss-function ablation

> ⚠️ **Seed-sensitivity caveat (added retrospectively, updated 3 seeds in)**: [Phase 5](phase5_seed_replication.md) ran 2 additional seeds (123, 7). Across 3 seeds, **L2's mean tuned F1 is 0.924 vs L1's 0.930 — within noise**. The original "L1 clearly beats L2" claim was a seed-42 artifact. At seed=7, L2 actually has the highest single-seed F1 of any variant (0.948). The Phase 2b finding that DensityAware needs threshold tuning is reinforced; the operating-point ranking below should be read as "L1 and tuned-L2 are practically tied" rather than a strict win.

**Run date**: 2026-05-05 (training 07:31 → 09:13).
**Branch**: `convnext_extension` @ `48575f0`.
**Hardware**: single RTX 4080 SUPER.
**Script**: [`run_phase2.sh`](../../run_phase2.sh).

## Setup

Fixed: `HerdNetTimmConvNext` (`convnext_baseline`, 32.6 M params), `fmo03_new_objcrop_augplus` dataset (Phase 1 winner), 30 epochs, `lr=8e-5`, `batch_size=4`, seed=42.

Varied: heatmap loss only.

| Tag | Loss config | Description |
|---|---|---|
| **L1** | `herdnet_fmo03` | FocalLoss (α=2, β=4) + weighted CE — same as Phase 1 A3, reused |
| **L2** | `density_aware_fmo03` | `DensityAwareHerdNetLoss` (radius=100, min_weight=0.5, max_weight=3.0, `density_scale=inverse_sqrt`) + weighted CE |

L3 (`density_aware_fmo03_aux`) needs aux P3/P4 outputs that `HerdNetTimmConvNext` does not emit (`herdnet_timm_convnext.py:555` returns `(heatmap, cls_out)`). Only `_v2` and `_v3` backbones return `(heatmap, cls_out, aux_p3, aux_p4)`. Deferred to Phase 3 where the backbone is varied.

## Per-epoch metrics (small cropped val)

| Tag | best F1 | F2 | precision | recall | MAE |
|---|---|---|---|---|---|
| L1 | 0.9107 | 0.8937 | 0.9405 | 0.8827 | 0.21 |
| L2 | **0.9195** | 0.9040 | 0.9467 | 0.8939 | 0.22 |

Cropped-val ranking: L2 ≳ L1 by ~1 pt F1. **This signal is misleading** — see below.

## Full-size stitched validation (HerdNetStitcher, overlap 120)

| Tag | Loss | F1 | precision | recall | MAE | RMSE | AP | detections |
|---|---|---|---|---|---|---|---|---|
| **L1** | `herdnet_fmo03` | **0.9056** | **0.9106** | 0.9006 | **1.50** | **1.87** | 0.8945 | 179 |
| L2 | `density_aware_fmo03` | 0.7981 | 0.6880 | **0.9503** | 5.75 | 6.41 | **0.9353** | 250 |

GT count: 181 iguanas across 12 frames.

## Effect — calibration, not quality

The two-headline takeaway is contradictory at first glance:

- **L1 wins F1 by 10.7 pts**, MAE by 73 %, RMSE by 71 %, precision by 22 pts.
- **L2 wins AP by 4 pts** and recall by 5 pts.

AP is a ranking metric (averaged across all thresholds). F1 here is computed at the configured operating point (`adapt_ts=0.3`, `evaluator.threshold=100`). L2 ranks detections better but its peak-confidence distribution is shifted — the `DensityAwareHerdNetLoss` reweights pixels in dense regions by 0.5×–3.0× (inverse-sqrt density), pushing high-confidence outputs onto cluster pixels that aren't actually peak iguanas. Result: 40 % more detections (250 vs 179) at the same threshold, ~80 of them false positives.

This is consistent with the loss's design intent — emphasising hard-to-detect cluster individuals — but the recalibration costs us at fixed-threshold inference. A threshold sweep on L2 (or local-max-suppression tuning) would likely close most of the gap, but the comparison above is at the same operating point both arms inherited from `training_settings/evaluator.yaml`.

Per-epoch metrics on the cropped val again disagreed with full-size. L2 looked marginally better there (+0.009 F1) and the calibration issue was invisible because the val set is small, individual-only (the full-size frames have density variation that the cropped val does not).

## Recommendations

1. **Lock `herdnet_fmo03` (L1) as the Phase-3 default loss.** It is the operating-point winner without further tuning.
2. **Don't drop L2 yet.** Higher AP means the underlying detector is stronger; the issue is calibration. If we reach a point where we want to push recall further, revisit L2 with a threshold/LMDS sweep on the full-size val before the final operating point is fixed. → Done, see [Phase 2b](phase2b_l2_threshold_sweep.md): tuning `adapt_ts` 0.30 → 0.70 closes ~80 % of L2's gap (F1 0.798 → 0.889, MAE 5.75 → 1.92), but L1 still wins on F1, MAE, and recall at comparable precision.
3. **L3 (`density_aware_fmo03_aux`) remains untested.** Roll into Phase 3: when the backbone is V2 or V3, also vary the loss to confirm that the aux deep-supervision heads help on top of DensityAware.

## Artifacts

- L1 baseline = Phase 1 A3 — `output/phase1_A3_convnext_baseline/2026-05-04/22-09-16/best_model.pth`, metrics at `output/phase1_fullval_20260504_210213/A3/`.
- L2 checkpoint: `output/phase2_L2_convnext_baseline/2026-05-05/07-31-14/best_model.pth`.
- L2 full-size metrics: `output/phase2_fullval_20260505_073107/L2/metrics_results.csv`.
- Per-run logs: `/tmp/phase2_20260505_073107/{1_L2_train.log, 1_L2_fullval.log}`.
- Wrapper log: `/tmp/phase2_20260505_073107_wrapper.log`.
- WandB project: `hn_phase2`, run `phase2_L2_density_aware_fmo03`.
