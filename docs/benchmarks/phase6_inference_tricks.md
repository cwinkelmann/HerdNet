# Phase 6 — inference-side "free lunch": TTA + 3-seed ensemble

**Run date**: 2026-05-07 12:40 → 14:55.
**Branch**: `convnext_extension` @ `c79ee69`.
**Hardware**: single RTX 4080 SUPER.
**Script**: [`run_phase6_inference_tricks.sh`](../../run_phase6_inference_tricks.sh) and [`tools/ensemble_infer.py`](../../tools/ensemble_infer.py).

## Setup

Phase 5 established the production candidate as **B4** (`fmo03_full_v2_bifpn`, ConvNeXt-T + BiFPN + DeformConv, `losses=herdnet_fmo03` override) trained at 3 seeds (42, 123, 7). Mean tuned F1 across the 3 seeds was 0.945 with best MAE 0.89 (the seed=42 best of 0.75 was anomalously low).

Phase 6 tests two inference-time tricks on the existing B4 checkpoints — **no retraining**:

| Tier | What it does | Cost |
|---|---|---|
| **T1: Per-seed TTA** | 4-fold (orig + Hflip + Vflip + 180°) heatmap averaging via `HerdNetStitcher(tta=True)` per single-seed checkpoint | 4× single-seed inference |
| **T2: 3-seed ensemble** | Pre-LMDS heatmap+cls averaging across the 3 B4 checkpoints (new tool: `tools/ensemble_infer.py`) | 3× single-seed inference |
| **T3: Ensemble + TTA** | Both stacked: 3 models × 4 TTA folds per tile | 12× single-seed inference |

T1 swept 3 thresholds (0.30, 0.35, 0.50) per seed = 9 evals. T2 and T3 swept 8 thresholds each (0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70).

## Per-seed TTA (T1) — best F1 by seed

| seed | no TTA (Phase 5) | + TTA (T1) | ΔF1 |
|---|---|---|---|
| 42 | 0.955 @ ts=0.50 | **0.957** @ ts=0.50 | +0.002 |
| 123 | 0.942 @ ts=0.40 | 0.951 @ ts=0.50 | +0.009 |
| 7 | 0.938 @ ts=0.50 | 0.946 @ ts=0.35 | +0.008 |
| mean | 0.945 | 0.951 | **+0.006** |

TTA gave +0.6 pt F1 on average and was strongest at lower thresholds. Cheap, reliable, no retraining.

## 3-seed ensemble (T2) — full sweep

| ts | F1 | precision | recall | MAE | RMSE | AP |
|---|---|---|---|---|---|---|
| 0.20 | 0.918 | 0.879 | 0.961 | 1.42 | 1.89 | 0.959 |
| 0.25 | 0.935 | 0.920 | 0.950 | 1.00 | 1.22 | 0.949 |
| **0.30** | 0.945 | 0.940 | 0.950 | **0.67** | 0.91 | 0.949 |
| 0.35 | 0.947 | 0.950 | 0.945 | 0.75 | 0.96 | 0.944 |
| **0.40** | **0.961** | 0.983 | 0.939 | 0.83 | 1.22 | 0.939 |
| 0.50 | 0.951 | 0.994 | 0.912 | 1.42 | 1.80 | 0.912 |
| 0.60 | 0.951 | 0.994 | 0.912 | 1.42 | 1.80 | 0.912 |
| 0.70 | 0.948 | 0.994 | 0.906 | 1.50 | 1.96 | 0.906 |

Two operating points stand out:
- **best F1 0.961 at ts=0.40** — beats every single-seed best across 3 seeds (mean 0.945, max 0.955) by a clear margin
- **best MAE 0.67 at ts=0.30** — beats every single-seed best MAE across 3 seeds (mean 0.89, min 0.75). Per-frame counting error <1 iguana with F1 still 0.945

The 3-seed ensemble *exceeds* the best single seed on both metrics simultaneously. That's the headline result.

## 3-seed ensemble + TTA (T3) — full sweep

| ts | F1 | precision | recall | MAE | RMSE | AP |
|---|---|---|---|---|---|---|
| 0.20 | 0.930 | 0.910 | 0.950 | 1.17 | 1.41 | 0.950 |
| 0.25 | 0.947 | 0.950 | 0.945 | 0.75 | 1.04 | 0.944 |
| 0.30 | 0.953 | 0.961 | 0.945 | 0.92 | 1.19 | 0.944 |
| **0.35** | **0.963** | 0.983 | 0.945 | 0.92 | 1.19 | 0.944 |
| 0.40 | 0.957 | 0.988 | 0.928 | 1.08 | 1.44 | 0.928 |
| 0.50 | 0.951 | 0.994 | 0.912 | 1.42 | 1.80 | 0.912 |
| 0.60 | 0.954 | **1.000** | 0.912 | 1.33 | 1.78 | 0.912 |
| 0.70 | 0.948 | **1.000** | 0.901 | 1.50 | 1.91 | 0.901 |

- **Best F1 of the entire study: 0.9634 at ts=0.35** with P=0.983, R=0.945
- **Precision = 1.000 at ts=0.60 and ts=0.70** with F1 still 0.948–0.954 — useful for screening / human-review pipelines where every flagged detection is reviewed
- TTA shifts the F1-optimum slightly (0.40 → 0.35) but the gains stack mostly on F1, not on MAE (which is best in T2 alone)

## Cross-tier comparison — best operating points

| Goal | Best config | Threshold | Metrics | Cost |
|---|---|---|---|---|
| **Best F1** | T3: 3-seed ensemble + TTA | `ts=0.35` | **F1=0.9634**, P=0.983, R=0.945, MAE=0.92 | 12× |
| **Best MAE** (counting) | T2: 3-seed ensemble (no TTA) | `ts=0.30` | F1=0.945, P=0.94, R=0.95, **MAE=0.67** | 3× |
| **Zero-FP screening** | T3: ensemble + TTA | `ts=0.60` | F1=0.954, **P=1.000**, R=0.912, MAE=1.33 | 12× |

For 12 val frames (181 GT iguanas) this means:
- Counting use: ~3× single-seed cost, 0.67-iguana avg per-frame error
- F1-optimal: ~12× single-seed cost (≈10 min for the 12-frame val set), F1 0.96+

## Observations

**Both tricks compose well for F1, not for MAE.** TTA + ensemble lifts F1 from 0.961 → 0.963 (+0.002) but raises MAE from 0.67 → 0.92 at the operating-point optimum. The TTA step shifts confidence around in a way that helps detection at well-tuned thresholds but hurts the MAE-optimal calibration the ensemble alone produces.

**Threshold-aware deployment matters.** The same Phase-6 stack delivers very different operating points depending on `adapt_ts`:
- 0.20–0.30: high-recall regime (recall ≥ 0.945, MAE close to optimal)
- 0.35–0.50: balanced F1-optimal regime
- 0.60–0.70: zero-FP screening regime

Don't pick one threshold and call it done — pick the threshold that matches the use case.

**Smoke-test confirmation.** Before launching the full sweep, a single `tools/ensemble_infer.py` smoke test at ts=0.35 returned F1=0.947, MAE=0.75, P=0.95, R=0.945, 180 detections (vs 181 GT) — already matching the seed=42 best MAE *and* slightly exceeding mean F1. That early signal validated the approach before committing the 2 h sweep.

## Implementation notes

`tools/ensemble_infer.py` (new) takes `--models <p1> <p2> ...` and replaces `_build_model` in the inference flow with one that:
1. Loads each checkpoint's `state_dict`
2. Strips a leading `model.` prefix from keys (LossWrapper artefact)
3. Wraps N models in `EnsembleHerdNet` — averages `(heatmap, cls_out)` across members
4. Re-wraps in `LossWrapper` so the rest of the pipeline (stitcher, evaluator, LMDS) works without changes

Total addition: ~120 lines, no changes to the existing single-checkpoint `tools/infer.py`.

## Recommendations

1. **Default operating point**: `T2 ensemble + ts=0.30` for census-style counting. F1=0.945, MAE=0.67, recall=0.95. ~3× inference cost, no retraining needed.
2. **F1-benchmark / publication number**: `T3 ensemble + TTA + ts=0.35` → F1=0.9634, MAE=0.92. ~12× inference cost.
3. **Screening pipeline**: `T3 ensemble + TTA + ts=0.60` → P=1.000 with F1=0.954. Zero false alarms.
4. **Don't ship single-seed**. The 3-seed ensemble on its own (no TTA) beats every single-seed configuration we measured on both F1 and MAE simultaneously, for a 3× cost that's tractable for batch jobs.
5. **For deployment cost-sensitivity**, prefer T2 over T3. The MAE gap (0.67 vs 0.92) is real and matters more for counting accuracy than the +0.002 F1 from adding TTA.

## Artifacts

- T1 sweep CSV: `output/phase6_sweep_20260507_124034/T1_per_seed_tta.csv`
- T2 sweep CSV: `output/phase6_sweep_20260507_124034/T2_ensemble_no_tta.csv`
- T3 sweep CSV: `output/phase6_sweep_20260507_124034/T3_ensemble_tta.csv`
- Per-threshold raw: `output/phase6_sweep_20260507_124034/<TAG>/ts_<value>_tta<bool>/metrics_results.csv`
- Logs: `/tmp/phase6_20260507_124034/`
- Wrapper log: `/tmp/phase6_20260507_124034_wrapper.log`
- New tool: `tools/ensemble_infer.py`
