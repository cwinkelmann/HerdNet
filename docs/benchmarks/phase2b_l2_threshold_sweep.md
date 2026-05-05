# Phase 2b — L2 (DensityAware) threshold sweep

Follow-up to [Phase 2](phase2_loss_ablation.md). Phase 2 found L2 (`density_aware_fmo03`) lost F1 by 10.7 pts to L1 (`herdnet_fmo03`) at the default operating point but won AP by 4 pts. The hypothesis was that L2 had the right *ranking* of candidates but wrong *calibration* of confidences — fixable by tuning the LMDS detection threshold (`adapt_ts`) without retraining. This phase tests that hypothesis.

**Run date**: 2026-05-05 (sweep started 13:35, ran concurrently with Phase 3 B5 training).
**Branch**: `convnext_extension` @ `48575f0`.
**Wrapper**: ad-hoc bash, see `/tmp/l2_sweep_20260505_133553.sh`.

## Methodology

Single fixed checkpoint (Phase 2 L2 best_model, ConvNeXt-Tiny + DensityAwareHerdNetLoss, 30 epochs). For each `adapt_ts ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70}` (11 values), ran `tools/infer.py --evaluate` against the full-size val set (12 frames, 181 GT iguanas) with HerdNetStitcher (overlap=120). All other parameters held constant.

Override applied:
```yaml
training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts: <swept>
```

Other LMDS parameters (`kernel_size=(9,9)`, `neg_ts` default, `scale_factor=1`, `up=True`) and the peak-heatmap threshold (`evaluator.threshold=100`) were left at their config defaults.

## Full sweep table

| ts | F1 | precision | recall | MAE | RMSE | AP | detections* |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.153 | 0.083 | 0.989 | 164.3 | 170.4 | 0.949 | ~2150 |
| 0.10 | 0.491 | 0.327 | 0.983 | 30.25 | 31.45 | 0.954 | ~545 |
| 0.15 | 0.641 | 0.478 | 0.972 | 15.58 | 16.76 | 0.950 | ~370 |
| 0.20 | 0.721 | 0.579 | 0.956 | 9.83 | 10.80 | 0.939 | ~300 |
| 0.25 | 0.766 | 0.638 | 0.956 | 7.50 | 8.31 | 0.940 | ~270 |
| **0.30** (default) | 0.798 | 0.688 | 0.950 | 5.75 | 6.41 | 0.935 | 250 |
| 0.35 | 0.824 | 0.731 | 0.945 | 4.58 | 5.35 | 0.931 | ~234 |
| 0.40 | 0.836 | 0.760 | 0.928 | 3.50 | 3.98 | 0.918 | ~221 |
| 0.50 | 0.860 | 0.810 | 0.917 | 2.17 | 2.65 | 0.909 | ~205 |
| 0.60 | 0.878 | 0.862 | 0.895 | 1.92 | 2.18 | 0.889 | ~188 |
| **0.70** (best F1) | **0.889** | **0.918** | 0.862 | **1.92** | 2.43 | 0.857 | ~170 |

\* Detection counts inferred from precision/recall × GT=181; exact at ts=0.30 from log.

## Operating-point picks

| Goal | Threshold | F1 | precision | recall | MAE | AP |
|---|---|---|---|---|---|---|
| Best F1 | `0.70` | 0.889 | 0.918 | 0.862 | 1.92 | 0.857 |
| Best MAE | `0.60` (or `0.70`, tied) | 0.878 | 0.862 | 0.895 | **1.92** | 0.889 |
| Best recall @ P ≥ 0.85 | `0.60` | 0.878 | 0.862 | 0.895 | 1.92 | 0.889 |

For census-style counting (MAE-driven), `adapt_ts=0.60` is the practical pick: same MAE as 0.70 but with 3.3 pts more recall.

## Comparison with L1 at its default operating point

| | L1 @ ts=0.30 (no sweep yet) | L2 @ ts=0.30 (Phase 2) | **L2 @ ts=0.70 (tuned)** |
|---|---|---|---|
| F1 | **0.9056** | 0.7981 | 0.8889 |
| precision | 0.9106 | 0.6880 | **0.9176** |
| recall | **0.9006** | 0.9503 | 0.8619 |
| MAE | **1.50** | 5.75 | 1.92 |
| AP | 0.8945 | 0.9353 | 0.8574 |
| detections | 179 | 250 | ~170 |

**Calibration hypothesis: partly correct.** Tuning ts=0.30 → 0.70 closed the F1 gap from −0.107 to −0.017 and the MAE gap from +4.25 to +0.42. ~80 % of L2's apparent loss to L1 was indeed calibration, not detector quality.

**But L1 still wins**, even at L2's best operating point:
- F1: L1 ahead by 0.017
- MAE: L1 ahead by 0.42 (1.50 vs 1.92)
- Recall at comparable precision (~0.91): L1 0.901 vs L2 0.862 — L1 sustains higher recall.

L2 hits a recall ceiling around 0.90 once precision climbs to 0.86; L1 holds 0.90 recall at precision 0.91. The DensityAware reweighting (0.5×–3.0× by inverse-sqrt density) traded peak-detector cleanliness for cluster-pixel sensitivity. Tuning the threshold can recover the cleanliness but not the lost peaks.

A direct apples-to-apples answer waits on Phase 3c, which sweeps L1 (= B2 = convnext_baseline) on the same grid. If L1's tuned best is also around 0.905–0.910, L1 is the genuine winner. If L1 climbs to 0.92+, the gap widens.

## Methodological finding — AP is *not* threshold-independent here

This sweep also exposed an interpretation pitfall in this codebase's metrics. Classical AP is computed on all detections (sorted by score) and is independent of any operating-point threshold. In `animaloc/eval/metrics.py`, the LMDS `adapt_ts` filter is applied *before* the PR curve is computed — so the AP we report is the area under the PR curve **of the surviving detections**, not over all heatmap peaks.

Evidence: L2's AP varies with threshold from 0.857 (ts=0.70) to 0.954 (ts=0.10). At very low thresholds, recall stretches close to 1.0 and the curve covers more of the PR plane → AP appears higher. At high thresholds, low-confidence true positives are filtered out before curve construction → AP appears lower.

**Practical implication**: AP in this codebase should be read as "ranking quality at this threshold," not as a global model-quality summary. Quoting AP without the threshold is meaningless. For Phase 3c we should report AP at the chosen operating point alongside the operating-point F1 / MAE / precision / recall.

## Recommendations

1. **Don't ship L2 over L1.** Even tuned, L2 doesn't beat L1's defaults on F1, MAE, or recall@high-precision.
2. **Tune `adapt_ts` per checkpoint, not per loss.** This sweep showed an 11-pt F1 swing on a single trained model just by changing the inference threshold. Every Phase-3 backbone should get the same sweep before final selection.
3. **Re-baseline L1 with a sweep too.** L1's default ts=0.3 may not be its best either. Phase 3c covers this.
4. **Update operating-point reporting**: when comparing models, report `(metric, threshold)` pairs, not bare numbers. AP especially.

## Artifacts

- L2 checkpoint: `output/phase2_L2_convnext_baseline/2026-05-05/07-31-14/best_model.pth`
- Sweep summary CSV: `output/l2_sweep_20260505_133553/L2_sweep.csv`
- Per-threshold infer outputs: `output/l2_sweep_20260505_133553/ts_<value>/{metrics_results.csv, confusion_matrix.csv, detections.csv}`
- Per-threshold infer logs: `/tmp/l2_sweep_20260505_133553/L2_sweep_<value>.log`
- Sweep wrapper script: `/tmp/l2_sweep_20260505_133553.sh`
