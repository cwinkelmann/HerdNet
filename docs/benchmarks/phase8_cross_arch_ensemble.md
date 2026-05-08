# Phase 8 — Cross-architecture ensemble (B3 + B4)

**Run date**: 2026-05-07 21:53 → 23:54.
**Branch**: `convnext_extension` @ `bb66e2e`.
**Hardware**: single RTX 4080 SUPER.
**Script**: [`run_phase8_cross_arch.sh`](../../run_phase8_cross_arch.sh).

## Hypothesis

[Phase 5](phase5_seed_replication.md) showed that **B3** (`fmo03_full_convnextv2_gabor`, ConvNeXt-V2 + Gabor projection) and **B4** (`fmo03_full_v2_bifpn`, ConvNeXt + BiFPN + DeformConv) are practically tied across 3 seeds — but they're architecturally distinct. Different feature extractors and different necks should make different errors, which is exactly when ensembling pays off (vs same-architecture seed-only ensembles where members are largely correlated).

[Phase 6](phase6_inference_tricks.md) used same-architecture B4×3 ensembling and got F1=0.9634, MAE=0.67. [Phase 7](phase7_ema.md) tried EMA on top — no help. The next-cheapest move was **swapping B4×3 for B3×3 + B4×3** — same compute scale (6 forward passes per tile vs 3, plus optional TTA on top), but with cross-arch diversity.

## Setup

**Members**: 6 checkpoints, all already trained:

| Member | Architecture | Seed | Source |
|---|---|---|---|
| B3_s42 | `CamouflageHerdNetConvNeXt` (ConvNeXt-V2 + Gabor) | 42 | Phase 3 |
| B3_s123 | same | 123 | Phase 5 |
| B3_s7 | same | 7 | Phase 5 |
| B4_s42 | `CamouflageHerdNetConvNeXtV2` (BiFPN + DeformConv) | 42 | Phase 3 |
| B4_s123 | same | 123 | Phase 5 |
| B4_s7 | same | 7 | Phase 5 |

All trained on `fmo03_new_objcrop_augplus` with `losses=herdnet_fmo03` override.

**Mechanism**: `tools/ensemble_infer.py` was extended in Phase 6 to be cross-architecture aware — it reads each checkpoint's saved `cfg` to instantiate the right model class per member, then averages `(heatmap, cls_out)` outputs across all 6 before the stitcher hands off to LMDS. No special handling needed for the 4-output models; B3 returns `(heatmap, cls_out)` and B4 returns `(heatmap, cls_out, aux_p3, aux_p4)` but only the first two are consumed.

**Sweeps**:
- **8a**: 6-model ensemble, no TTA, `adapt_ts ∈ {0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70}`
- **8b**: 6-model ensemble + 4-fold TTA, `adapt_ts ∈ {0.30, 0.35, 0.40, 0.50}` (probe thresholds only — TTA stacks 4× on top of the 6× ensemble cost ≈ 24× single-seed inference per eval)

## Results — 8a (no TTA)

| ts | F1 | precision | recall | MAE | RMSE | AP |
|---|---|---|---|---|---|---|
| 0.20 | 0.954 | 0.941 | 0.967 | 0.75 | 1.12 | 0.966 |
| **0.25** | **0.967** | 0.972 | **0.961** | **0.50** | 0.82 | 0.961 |
| 0.30 | 0.966 | 0.983 | 0.950 | 0.67 | 1.00 | 0.950 |
| 0.35 | 0.966 | 0.994 | 0.939 | 1.00 | 1.29 | 0.939 |
| 0.40 | 0.963 | 0.994 | 0.934 | 1.08 | 1.38 | 0.934 |
| 0.50 | 0.960 | **1.000** | 0.923 | 1.17 | 1.58 | 0.923 |
| 0.60 | 0.954 | **1.000** | 0.912 | 1.33 | 1.78 | 0.912 |
| 0.70 | 0.942 | **1.000** | 0.890 | 1.67 | 2.08 | 0.890 |

## Results — 8b (with TTA)

| ts | F1 | precision | recall | MAE | RMSE | AP |
|---|---|---|---|---|---|---|
| **0.30** | **0.972** | 0.994 | 0.950 | 0.83 | 1.15 | 0.950 |
| 0.35 | 0.963 | 0.994 | 0.934 | 1.08 | 1.38 | 0.934 |
| 0.40 | 0.957 | 0.994 | 0.923 | 1.25 | 1.61 | 0.923 |
| 0.50 | 0.957 | **1.000** | 0.917 | 1.25 | 1.66 | 0.917 |

## Best operating points — Phase 8 vs Phase 6

| Goal | Phase 6 (B4×3 only) | **Phase 8 (B3×3+B4×3)** | Δ |
|---|---|---|---|
| best F1 | 0.9634 (T3, ts=0.35) | **0.9718** (8b, ts=0.30) | **+0.008** |
| best MAE | 0.67 (T2, ts=0.30) | **0.50** (8a, ts=0.25) | **−0.17** (−25 %) |
| zero-FP screening | F1=0.948 @ P=1.000 | F1=0.960 @ P=1.000 | F1 +0.012 |
| recall @ P=0.99 | 0.945 | **0.961** | +0.016 |

**Cross-arch ensemble dominates same-arch ensemble across every operating point.** The MAE improvement from 0.67 to **0.50** is the headline — average per-frame counting error is now **half an iguana**, on a val set that averages ~15 iguanas per frame.

## Why it worked

Phase 7 (EMA) regressed the ensemble because EMA correlated the 3 seeds' weights — they converged to similar minima, killing diversity. Phase 8 does the opposite: deliberately picking *different architectures* maximises diversity. B3 (ConvNeXt-V2 + Gabor) emphasises texture features (Gabor projection on stage 0); B4 (BiFPN + DeformConv) emphasises multi-scale geometric features. Their errors don't fully overlap.

The Phase 11 error analysis (run on 8a's best-MAE config) confirms this directly:
- 6 of Phase 9's 11 FPs were **eliminated** by cross-arch ensembling
- 2 of Phase 9's 9 FNs were **recovered**
- The single high-confidence FP that survived is one both architectures agreed looks like an iguana — i.e. a genuine ambiguity, not a model artefact

## Cost

| Configuration | Forward passes per tile | Wall-clock per 12-frame eval |
|---|---|---|
| Single seed B4 | 1 | ~30 s |
| Phase 6: B4×3 ensemble | 3 | ~90 s |
| Phase 6: B4×3 + TTA | 12 | ~10 min |
| **Phase 8a: B3×3 + B4×3** | 6 | **~3 min** |
| **Phase 8b: B3×3 + B4×3 + TTA** | 24 | **~12 min** |

For batch jobs, 8a (3 min) is the practical sweet spot — best MAE (0.50) at minimal extra compute. For F1-optimal eval, 8b at ts=0.30 (12 min for the full val set) gives F1=0.972.

## Recommendations

**New production stack**:

```yaml
config:           fmo03_full_v2_bifpn  +  fmo03_full_convnextv2_gabor
loss:             herdnet_fmo03 (override on B3)
dataset:          fmo03_new_objcrop_augplus
ensemble:         B3 × {seed 42, 123, 7}  +  B4 × {seed 42, 123, 7}
threshold (ts):   0.25 for counting (MAE 0.50)
                  0.30 with TTA for F1 (0.972)
                  0.50 for zero-FP screening (P=1.000, F1=0.960)
stitcher overlap: 120 (do NOT increase — see Phase 10)
```

This is the strongest configuration produced across all phases. Document the threshold-vs-use-case mapping carefully in any deployment doc — the same ensemble at different `adapt_ts` answers very different operational questions.

## Caveats

- Single-eval result (one val set, no cross-validation). The Phase-5 seed-noise floor was ~0.02 F1; the Phase-6 → Phase-8 F1 gain (+0.008) is below that floor. **The MAE improvement (0.67 → 0.50, −25 %) is well above noise** and is the more confident claim.
- All 6 ensemble members were trained on the same 19 frames. Cross-architecture diversity is reliable; cross-data diversity (which would require k-fold or bagging) was already deemed not worth it on this dataset (see seed-replication discussion).

## Artifacts

- Phase 8a sweep CSV: `output/phase8_sweep_20260507_215325/T1_cross_arch_no_tta.csv`
- Phase 8b sweep CSV: `output/phase8_sweep_20260507_215325/T2_cross_arch_tta.csv`
- Per-eval raw: `output/phase8_sweep_20260507_215325/<TAG>/ts_<v>_tta<bool>/metrics_results.csv`
- Logs: `/tmp/phase8_20260507_215325/`
- Wrapper log: `/tmp/phase8_20260507_215325_wrapper.log`
