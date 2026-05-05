# Phase 4 — B4 with native `density_aware_fmo03_aux` loss

**Run date**: 2026-05-05 (training 16:29 → 18:23, sweep 18:23 → 18:42).
**Branch**: `convnext_extension` @ `69589bb`.
**Hardware**: single RTX 4080 SUPER.
**Script**: [`run_phase4.sh`](../../run_phase4.sh).

## Hypothesis

[Phase 3](phase3_backbone_threshold.md) ran B4 (`fmo03_full_v2_bifpn`) with `losses=herdnet_fmo03` overridden — same loss as every other backbone for clean ablation. That handicap removed the gradient signal from B4's `aux_p3` and `aux_p4` heads (which `herdnet_fmo03` doesn't reference). Even handicapped, B4 produced the best results of the phase: F1 0.955 @ ts=0.50, MAE 0.75 @ ts=0.35.

The natural follow-up: B4's *native* loss is `density_aware_fmo03_aux` (DensityAware + weighted CE + aux FocalLoss on P3 λ=0.3 and P4 λ=0.1). With aux gradients flowing, the deep-supervision intuition says intermediate FPN levels should learn richer representations and the final detector should improve. **Phase 4 tests this.**

## Setup

Identical to Phase 3 B4 except for the loss:

| Variable | Phase 3 B4 | Phase 4 B4 |
|---|---|---|
| Backbone / config | `fmo03_full_v2_bifpn` (`CamouflageHerdNetConvNeXtV2`, BiFPN + DeformConv) | same |
| Dataset | `fmo03_new_objcrop_augplus` | same |
| Epochs / bs / seed | 30 / 4 / 42 | same |
| **Loss** | `herdnet_fmo03` (FocalLoss + weighted CE) — **override** | `density_aware_fmo03_aux` — **native** |
| Aux heads | trained but ignored by loss (no gradient) | actively supervised (λ=0.3 on P3, λ=0.1 on P4) |
| GPU peak memory | ~3 GiB | 9.3 GiB (aux gradients) |

## Per-epoch metrics (cropped val — convergence tracking)

| | per-epoch best F1 | precision | recall | MAE |
|---|---|---|---|---|
| Phase 3 B4 (override) | **0.9459** | 0.9651 | 0.9274 | **0.11** |
| Phase 4 B4 (native) | 0.9318 | 0.9480 | 0.9162 | 0.16 |

Phase 4 already loses on per-epoch by 1.4 pts F1 and +0.05 MAE. (This time per-epoch was a faithful predictor — see "What this tells us" below.)

## Phase 4b — full-size eval at default `adapt_ts=0.30`

```
tag         F1      P       R       MAE     RMSE    AP
Phase 3 B4  0.934   0.929   0.939   0.83    1.22    0.933
Phase 4 B4  0.925   0.933   0.917   1.25    1.66    0.916
Δ           -0.009  +0.004  -0.022  +0.42   +0.44   -0.017
```

Native loses every operating-point metric except a 0.4 pt precision uptick. MAE is +0.42 worse — already a meaningful regression.

## Phase 4c — threshold sweep

| ts | F1 | precision | recall | MAE | RMSE | AP |
|---|---|---|---|---|---|---|
| 0.05 | 0.547 | 0.384 | 0.956 | 22.50 | 23.59 | 0.951 |
| 0.10 | 0.821 | 0.723 | 0.950 | 4.75 | 5.59 | 0.946 |
| 0.15 | 0.883 | 0.833 | 0.939 | 2.08 | 2.57 | 0.937 |
| 0.20 | 0.909 | 0.885 | 0.934 | 1.50 | 1.91 | 0.932 |
| **0.25** | 0.923 | 0.918 | 0.928 | **1.17** | 1.41 | 0.926 |
| 0.30 | 0.925 | 0.933 | 0.917 | 1.25 | 1.66 | 0.916 |
| 0.35 | 0.927 | 0.938 | 0.917 | 1.33 | 1.68 | 0.916 |
| 0.40 | 0.924 | 0.948 | 0.901 | 1.42 | 1.71 | 0.900 |
| **0.50** | **0.940** | 0.982 | 0.901 | 1.58 | 1.89 | 0.900 |
| 0.60 | 0.939 | 0.988 | 0.895 | 1.75 | 2.06 | 0.895 |
| 0.70 | 0.939 | 0.988 | 0.895 | 1.75 | 2.06 | 0.895 |

## Best operating points — Phase 4 vs Phase 3 B4

| Goal | Phase 3 B4 (override) | Phase 4 B4 (native) | Δ |
|---|---|---|---|
| Best F1 | **0.955** @ ts=0.50 (P=0.977, R=0.934, MAE=0.83) | 0.940 @ ts=0.50 (P=0.982, R=0.901, MAE=1.58) | F1 −0.015, MAE +0.75 |
| Best MAE | **0.75** @ ts=0.35 (F1=0.942) | 1.17 @ ts=0.25 (F1=0.923) | MAE +0.42, F1 −0.019 |
| Recall @ P ≥ 0.85 | **0.956** @ ts=0.20 (P=0.883) | 0.934 @ ts=0.20 (P=0.885) | R −0.022 |

**Phase 3 B4 (handicapped) wins every operating point.** Native is worse on F1 by 0.015, on MAE by 0.42 (more than half a iguana per frame), on recall at high precision by 0.022.

## What this tells us

**The DensityAware reweighting hurts calibration on this architecture too**, not just on convnext_baseline. Phase 2 showed it on a plain ConvNeXt; Phase 4 reproduces the same effect on BiFPN+DeformConv. The pattern: density-weighted pixels (cluster regions) get pushed to high confidence, generating high-confidence false positives at the operating point. This time the additional aux deep-supervision didn't compensate.

**Deep-supervision aux heads alone aren't a free lunch.** With the same architecture and dataset, adding gradient signal to `aux_p3` and `aux_p4` produced *worse* end metrics (per-epoch F1 0.932 vs 0.946 with aux heads frozen). Possible reasons:

1. The aux losses are weighted relative to the main loss (`λ_p3=0.3, λ_p4=0.1`) and trained simultaneously. The optimisation problem is harder (4 loss terms vs 2), and on a 30-epoch budget the joint optimum may not match the main-only optimum the override variant found.
2. The DensityAware main loss already concentrates training signal on cluster pixels via the 0.5×–3.0× weight. Compounding that with aux losses on intermediate FPN levels may have over-emphasised cluster regions further.
3. The aux heads' outputs are interpolated to match heatmap size and supervised against the *same* target. If the aux supervision encourages multiple FPN levels to fire on the same locations, the final heatmap may have more spurious high-confidence peaks (matching what we see — F1 only 0.940 even at ts=0.50 because precision rises to 0.982 but recall caps at 0.901).

**Per-epoch was a faithful signal here.** That's worth noting against the pattern of Phases 1–3, where per-epoch lied. Phase 4's per-epoch F1 was 0.014 below Phase 3 B4's; full-size eval confirmed the gap (0.015 F1 at best). Probably because both runs share the same architecture and dataset; the only difference is the loss, and its effect propagates consistently from cropped val to full-size val.

**AP doesn't hide the regression** here, unlike Phase 2 L2. Phase 4 AP at default = 0.916, vs Phase 3 B4's 0.933. The native loss didn't even produce a stronger ranker — it produced a worse one *and* worse calibration.

## Recommendations

1. **Drop the Phase 3 follow-up suggestion to use native loss for B4.** Phase 4 falsifies it. The previous recommendation in [phase3 doc](phase3_backbone_threshold.md) under "Suggested Phase 4" was wrong; the native config underperforms.
2. **`herdnet_fmo03` (Focal + weighted CE) is the production loss** across all backbones tested so far. Phase 2 (convnext_baseline), Phase 3 (5 backbones with override), and now Phase 4 (B4 native is worse) all support this.
3. **The Phase 3 production recommendations stand**:
   - Default operating point: B4 + `adapt_ts=0.35` → F1=0.942, MAE=0.75
   - Detection benchmark: B4 + `adapt_ts=0.50` → F1=0.955
   - Zero-FP screening: B3 + `adapt_ts=0.70` → P=1.000, F1=0.942
4. **Don't bother running Phase 4-style native-loss experiments on B3 or B5.** B3's native is `density_aware_fmo03` (no aux); we already established density-aware loses to Focal+CE in Phase 2. B5's native is the same. The pattern is consistent: density-aware loss hurts at the operating point, no aux heads change that.

## Caveats

- This is a single seed (42) per configuration. Phase 4 lost by 0.015 F1 — within the range that 3-seed variance might cover. If seed variance matters, re-run both Phase 3 B4 and Phase 4 B4 with seeds {42, 123, 7}. But the +0.42 MAE gap is much harder to explain by seed noise.
- The Phase 4 trainer's GPU peak memory was 9.3 GiB vs ~3 GiB for the override variant — the aux heads more than tripled memory. If memory ever becomes a constraint, this is another point against the native config.

## Artifacts

- B4 native checkpoint: `output/phase4_B4_native_v2_bifpn/2026-05-05/16-29-05/best_model.pth`
- Phase 4b full-size metrics: `output/phase4_fullval_20260505_162856/B4_native/metrics_results.csv`
- Phase 4c sweep:
  - Summary: `output/phase4_sweep_20260505_162856/sweep.csv`
  - Per-threshold: `output/phase4_sweep_20260505_162856/ts_<value>/metrics_results.csv`
- Logs: `/tmp/phase4_20260505_162856/`
- Wrapper log: `/tmp/phase4_20260505_162856_wrapper.log`
- WandB project: `hn_phase4`, run `phase4_B4_native`
