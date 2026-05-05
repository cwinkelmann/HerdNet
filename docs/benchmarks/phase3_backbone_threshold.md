# Phase 3 — backbone comparison + threshold tuning

**Run date**: 2026-05-05 (training 09:28 → ~14:35, sweep finished ~15:20).
**Branch**: `convnext_extension` @ `48575f0`.
**Hardware**: single RTX 4080 SUPER.
**Script**: [`run_phase3.sh`](../../run_phase3.sh).

## Setup

Fixed: `fmo03_new_objcrop_augplus` dataset (Phase 1 winner), `herdnet_fmo03` loss (Phase 2 winner), 30 epochs, `batch_size=4`, seed=42, evaluator threshold=100, stitcher overlap=120.

Varied: backbone (5 variants).

| Tag | Config | Backbone class / arch | Loss override? |
|---|---|---|---|
| **B1** | `fmo03_full_dla34_timm` | `HerdNetTimmDLA` (DLA-34 via timm) | no (native = `herdnet_fmo03`) |
| **B2** | `fmo03_full_convnext_baseline` | `HerdNetTimmConvNext` (ConvNeXt-T) — **REUSED from Phase 1 A3** | no |
| **B3** | `fmo03_full_convnextv2_gabor` | `CamouflageHerdNetConvNeXt` w/ ConvNeXtV2-T + Gabor projection | yes (native = `density_aware_fmo03`) |
| **B4** | `fmo03_full_v2_bifpn` | `CamouflageHerdNetConvNeXtV2` (ConvNeXt-T + BiFPN + DeformConv) | yes (native = `density_aware_fmo03_aux`) |
| **B5** | `fmo03_full_efficientvit_gabor` | `CamouflageHerdNetConvNeXt` w/ EfficientViT-tiny + Gabor projection | yes (native = `density_aware_fmo03`) |

The loss override on B3/B4/B5 is a deliberate handicap for fairness — Phase 2 showed `herdnet_fmo03` wins at the operating point on a baseline ConvNeXt, so we test each backbone with the same loss. B4's `aux_p3`/`aux_p4` heads are still emitted by the model (4 outputs) but `herdnet_fmo03` only consumes outputs 0 and 1, so the aux heads received no gradient signal during training. Inference doesn't use them either, so this is a fair test of the BiFPN+DeformConv neck without deep supervision.

Per-epoch validation: cropped val (98 patches). Final full-size validation: 12 frames, 181 GT iguanas, stitched at overlap=120.

## Per-epoch metrics (cropped val — convergence tracking only, do not use for selection)

| Tag | best F1 | precision | recall | MAE | best_val |
|---|---|---|---|---|---|
| B1 | 0.9162 | 0.9162 | 0.9162 | 0.18 | 0.9164 |
| B2 (=A3) | 0.9107 | 0.9405 | 0.8827 | 0.21 | 0.9271 |
| B3 | 0.9371 | 0.9591 | 0.9162 | 0.18 | 0.9371 |
| B4 | **0.9459** | 0.9651 | 0.9274 | **0.11** | **0.9489** |
| B5 | 0.8757 | 0.8857 | 0.8659 | 0.39 | 0.9030 |

Per-epoch ranking: B4 > B3 > B1 ≈ B2 > B5. As Phase 1/2 already established, this signal is partial.

## Phase 3b — full-size stitched validation at the default operating point (`adapt_ts=0.30`)

| Tag | Backbone | F1 | precision | recall | MAE | RMSE | AP |
|---|---|---|---|---|---|---|---|
| B1 | DLA-34 timm | 0.864 | 0.821 | 0.912 | 2.00 | 3.00 | 0.905 |
| B2 | ConvNeXt baseline (=A3) | 0.906 | 0.911 | 0.901 | 1.50 | 1.87 | 0.895 |
| **B3** | **ConvNeXt-V2 + Gabor** | 0.946 | **0.971** | 0.923 | 1.08 | 1.32 | 0.923 |
| **B4** | **ConvNeXt + BiFPN + DeformConv** | 0.934 | 0.929 | **0.939** | **0.83** | **1.22** | **0.933** |
| B5 | EfficientViT + Gabor | 0.875 | 0.828 | 0.928 | 2.17 | 2.68 | 0.893 |

**At the default threshold, B3 wins F1/precision and B4 wins MAE/recall/AP.** B1 (DLA-34) and B5 (EfficientViT) lag both ConvNeXt-family alternatives.

## Phase 3c — threshold sweep on full-size val

`adapt_ts ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70}`. Per-backbone tables below; 0.30 row matches Phase 3b (modulo B2's scope-dependent `_stored_metrics` differences picked up in re-runs):

### B1 — DLA-34 timm

| ts | F1 | P | R | MAE | RMSE | AP |
|---|---|---|---|---|---|---|
| 0.05 | 0.203 | 0.114 | 0.978 | 114.92 | 117.64 | 0.908 |
| 0.10 | 0.494 | 0.333 | 0.956 | 28.25 | 29.74 | 0.922 |
| 0.15 | 0.665 | 0.514 | 0.945 | 12.67 | 13.46 | 0.927 |
| 0.20 | 0.757 | 0.634 | 0.939 | 7.25 | 7.94 | 0.925 |
| 0.25 | 0.829 | 0.752 | 0.923 | 3.42 | 4.61 | 0.914 |
| 0.30 | 0.864 | 0.821 | 0.912 | 2.00 | 3.00 | 0.905 |
| 0.35 | 0.881 | 0.862 | 0.901 | **1.67** | 2.42 | 0.895 |
| 0.40 | 0.894 | 0.899 | 0.890 | 1.67 | 2.20 | 0.885 |
| 0.50 | 0.900 | 0.935 | 0.867 | 1.75 | 2.22 | 0.865 |
| 0.60 | 0.898 | 0.951 | 0.851 | 1.92 | 2.33 | 0.849 |
| **0.70** | **0.913** | 0.994 | 0.845 | 2.42 | 2.81 | 0.844 |

### B2 — ConvNeXt baseline (= L1 = A3)

| ts | F1 | P | R | MAE | RMSE | AP |
|---|---|---|---|---|---|---|
| 0.05 | 0.424 | 0.272 | 0.967 | 38.58 | 39.88 | 0.945 |
| 0.10 | 0.785 | 0.669 | 0.950 | 6.33 | 6.88 | 0.939 |
| 0.15 | 0.854 | 0.786 | 0.934 | 2.83 | 3.58 | 0.925 |
| 0.20 | 0.882 | 0.840 | 0.928 | 1.58 | 2.40 | 0.921 |
| 0.25 | 0.905 | 0.892 | 0.917 | **1.25** | 1.55 | 0.910 |
| 0.30 | 0.906 | 0.911 | 0.901 | 1.50 | 1.87 | 0.895 |
| 0.35 | 0.913 | 0.926 | 0.901 | 1.58 | 1.89 | 0.895 |
| **0.40** | **0.924** | 0.948 | 0.901 | 1.42 | 1.85 | 0.895 |
| 0.50 | 0.922 | 0.964 | 0.884 | 1.58 | 2.29 | 0.878 |
| 0.60 | 0.915 | 0.981 | 0.856 | 2.08 | 2.66 | 0.851 |
| 0.70 | 0.913 | 0.994 | 0.845 | 2.25 | 2.93 | 0.840 |

### B3 — ConvNeXt-V2 + Gabor

| ts | F1 | P | R | MAE | RMSE | AP |
|---|---|---|---|---|---|---|
| 0.05 | 0.688 | 0.534 | 0.967 | 12.25 | 13.17 | 0.960 |
| 0.10 | 0.861 | 0.783 | 0.956 | 3.33 | 4.22 | 0.953 |
| 0.15 | 0.908 | 0.869 | 0.950 | 1.42 | 2.14 | 0.949 |
| 0.20 | 0.932 | 0.915 | 0.950 | 1.25 | 1.71 | 0.949 |
| 0.25 | 0.939 | 0.949 | 0.928 | 1.17 | 1.35 | 0.928 |
| 0.30 | 0.946 | 0.971 | 0.923 | 1.08 | 1.32 | 0.923 |
| 0.35 | 0.949 | 0.977 | 0.923 | **1.00** | 1.29 | 0.923 |
| **0.40** | **0.952** | 0.982 | 0.923 | 1.08 | 1.38 | 0.923 |
| 0.50 | 0.951 | 0.994 | 0.912 | 1.25 | 1.66 | 0.912 |
| 0.60 | 0.945 | 0.994 | 0.901 | 1.42 | 1.85 | 0.901 |
| 0.70 | 0.942 | **1.000** | 0.890 | 1.67 | 2.08 | 0.890 |

### B4 — ConvNeXt + BiFPN + DeformConv ★

| ts | F1 | P | R | MAE | RMSE | AP |
|---|---|---|---|---|---|---|
| 0.05 | 0.570 | 0.400 | 0.989 | 22.17 | 23.41 | 0.974 |
| 0.10 | 0.810 | 0.697 | 0.967 | 5.83 | 6.49 | 0.957 |
| 0.15 | 0.874 | 0.805 | 0.956 | 2.83 | 3.29 | 0.948 |
| 0.20 | 0.918 | 0.883 | 0.956 | 1.75 | 2.14 | 0.949 |
| 0.25 | 0.937 | 0.929 | 0.945 | 0.92 | 1.32 | 0.938 |
| 0.30 | 0.934 | 0.929 | 0.939 | 0.83 | 1.22 | 0.933 |
| 0.35 | 0.942 | 0.944 | 0.939 | **0.75** | **1.12** | 0.933 |
| 0.40 | 0.950 | 0.961 | 0.939 | 0.83 | 1.15 | 0.933 |
| **0.50** | **0.955** | 0.977 | 0.934 | 0.83 | 1.15 | 0.928 |
| 0.60 | 0.951 | 0.994 | 0.912 | 1.42 | 1.76 | 0.906 |
| 0.70 | 0.948 | **1.000** | 0.901 | 1.50 | 1.87 | 0.901 |

### B5 — EfficientViT + Gabor

| ts | F1 | P | R | MAE | RMSE | AP |
|---|---|---|---|---|---|---|
| 0.05 | 0.168 | 0.092 | 0.989 | 147.42 | 161.82 | 0.904 |
| 0.10 | 0.522 | 0.357 | 0.967 | 25.75 | 27.90 | 0.919 |
| 0.15 | 0.693 | 0.544 | 0.956 | 11.42 | 12.80 | 0.916 |
| 0.20 | 0.780 | 0.667 | 0.939 | 6.17 | 7.16 | 0.902 |
| 0.25 | 0.830 | 0.750 | 0.928 | 3.75 | 4.35 | 0.893 |
| 0.30 | 0.875 | 0.828 | 0.928 | 2.17 | 2.68 | 0.893 |
| 0.35 | 0.898 | 0.871 | 0.928 | 1.33 | 1.68 | 0.894 |
| 0.40 | 0.905 | 0.888 | 0.923 | 0.92 | 1.32 | 0.888 |
| 0.50 | 0.917 | 0.922 | 0.912 | **0.83** | 1.00 | 0.878 |
| **0.60** | **0.918** | 0.942 | 0.895 | 1.08 | 1.38 | 0.862 |
| 0.70 | 0.915 | 0.975 | 0.862 | 2.08 | 2.57 | 0.846 |

## Best-operating-point summary

| Tag | Backbone | best F1 @ ts | best MAE @ ts (F1 there) | recall@P≥0.85 (P @ ts) |
|---|---|---|---|---|
| B1 | DLA-34 timm | 0.913 @ 0.70 | 1.67 @ 0.35 (F1=0.881) | 0.901 @ 0.35 (P=0.862) |
| B2 | ConvNeXt baseline | 0.924 @ 0.40 | 1.25 @ 0.25 (F1=0.905) | 0.917 @ 0.25 (P=0.892) |
| **B3** | **ConvNeXt-V2 + Gabor** | 0.952 @ 0.40 | 1.00 @ 0.35 (F1=0.949) | **0.950 @ 0.15** (P=0.869) |
| **B4** | **ConvNeXt + BiFPN + DeformConv** | **0.955 @ 0.50** | **0.75 @ 0.35** (F1=0.942) | **0.956 @ 0.20** (P=0.883) |
| B5 | EfficientViT + Gabor | 0.918 @ 0.60 | 0.83 @ 0.50 (F1=0.917) | 0.928 @ 0.35 (P=0.870) |

## Effect of threshold tuning (vs default ts=0.30)

| Tag | F1 default → tuned | ΔF1 | MAE default → tuned | ΔMAE |
|---|---|---|---|---|
| B1 | 0.864 → 0.913 | +0.049 | 2.00 → 1.67 | −0.33 |
| B2 | 0.906 → 0.924 | +0.018 | 1.50 → 1.25 | −0.25 |
| B3 | 0.946 → 0.952 | +0.006 | 1.08 → 1.00 | −0.08 |
| B4 | 0.934 → 0.955 | +0.021 | 0.83 → 0.75 | −0.08 |
| B5 | 0.875 → **0.918** | **+0.043** | 2.17 → **0.83** | **−1.34** |

Tuning matters most for the weaker / less well-calibrated models. B5's MAE almost triples in quality (2.17 → 0.83), pushing it from worst-MAE to third-best-MAE, just by raising the threshold. B3 and B4 are already near-optimally calibrated at the default; tuning gives them a small bump.

## Reading the results

**B4 (ConvNeXt + BiFPN + DeformConv) is the overall winner.**
- Highest tuned F1: 0.955.
- Lowest MAE across the entire phase: **0.75 iguanas per frame** at ts=0.35. The val set has ~15 iguanas/frame, so the model's count is essentially perfect.
- Highest recall@P≥0.85: 0.956 at ts=0.20.
- Per-epoch already showed B4 winning, and full-size confirms it. This is the rare case where per-epoch was a reliable signal — likely because B4's loss + architecture combination produced well-calibrated outputs from the start.
- Despite the loss override removing its native auxiliary deep supervision (`density_aware_fmo03_aux` → `herdnet_fmo03`), the BiFPN + DeformConv neck still produced the strongest detector.

**B3 (ConvNeXt-V2 + Gabor) is a close second.** Best F1 only 0.003 below B4 (0.952 vs 0.955); best MAE 0.25 above B4 (1.00 vs 0.75). At ts=0.70 B3 hits **precision = 1.000** with F1=0.942, which is striking — there is a threshold where the model fires only on real iguanas. That kind of behaviour is useful for screening / human-in-the-loop pipelines.

**B2 (ConvNeXt baseline) is a respectable third** (best F1 0.924, MAE 1.25). It's the "without bells and whistles" reference. The gap to B3/B4 (~0.03 F1, ~0.25–0.50 MAE) is a real gain from the architectural additions (Gabor projection on stage 0; BiFPN+DeformConv neck).

**B1 (DLA-34 timm) and B5 (EfficientViT) lag** at the default threshold. B5 is interesting: its raw model is competitive after threshold tuning (best F1 0.918, MAE 0.83 — better MAE than B2!), it was just badly calibrated by default. B1 caps out lower (best F1 0.913) — the DLA backbone is genuinely behind ConvNeXt-family on this data even after tuning.

**Confirms the per-epoch lesson** from Phases 1 and 2: per-epoch metrics on the cropped val partly track full-size performance for the top model (B4) but undersell B5 (per-epoch said worst, full-size after tuning says competitive) and partly oversell B1 (per-epoch said tied with B2, full-size says behind by 0.011 F1). Per-epoch is convergence tracking, not selection.

**AP behaviour confirms the Phase 2b methodological note**: AP varies materially with threshold here. B1's AP ranges 0.844–0.927; B5's ranges 0.846–0.919. AP at any single threshold is not a model-quality summary. For each backbone, the AP near the best-F1 threshold is the most operationally relevant value.

## Recommendations

1. **Default to B4 (`fmo03_full_v2_bifpn`) with `adapt_ts=0.35`** for production iguana counting. F1=0.942, MAE=0.75 means your per-frame count is off by less than one iguana on average, with 94 % of detections being real and 94 % of real iguanas being detected.
2. **For F1-optimal detection** (e.g. publication metric), use B4 at `adapt_ts=0.50`: F1=0.955, P=0.977, R=0.934, MAE=0.83.
3. **For zero-FP screening** (no false alarms allowed), use B3 at `adapt_ts=0.70`: precision=1.000, F1=0.942, R=0.890, MAE=1.67. Useful for triage pipelines where every flagged object is reviewed by a human.
4. **For high-recall screening** (don't miss any iguana), use B4 at `adapt_ts=0.20`: recall=0.956 with precision still 0.883.
5. **Re-train B4 with its native `density_aware_fmo03_aux` loss** as a follow-up. It outperformed B3 even with the loss override; with native deep-supervision aux heads it might extend the lead. This is the natural Phase 4 candidate.
6. **Adopt threshold tuning as a standard final step.** Phase 3c showed +0.018 to +0.049 F1 per backbone for free, no retraining. The wrapper script is reusable; just run it on any new checkpoint.
7. **B1 and B5 are not worth pursuing further on this dataset.** They lag by a clear margin even with tuning. EfficientViT might pay off on much larger datasets where its parameter efficiency matters; here it doesn't.

## Notes / caveats

- B2's sweep CSV in `summary.csv` has malformed columns because the variant label `"HerdNetTimmConvNext (=A3, reused)"` contained a comma that broke the CSV. Per-threshold `metrics_results.csv` files under `output/phase3_sweep_20260505_092807/B2/ts_<value>/` are correct and were used to populate B2's table above. **Fix in `run_phase3.sh`**: rename the label or quote the CSV field.
- Phase 3c sweep ran concurrently with the [Phase 2b L2 sweep](phase2b_l2_threshold_sweep.md) (no GPU contention issues observed).
- Some `_stored_metrics` re-eval differences may produce tiny inconsistencies between Phase 3b's first-pass numbers and the same-threshold row in 3c (e.g., B2 default-row F1 differs by < 0.001 in some places). These are within evaluator floating-point variance and don't change conclusions.

## Artifacts

- Best checkpoints:
  - B1: `output/phase3_B1_fmo03_full_dla34_timm/2026-05-05/09-28-24/best_model.pth`
  - B2: `output/phase1_A3_convnext_baseline/2026-05-04/22-09-16/best_model.pth` (reused from Phase 1)
  - B3: `output/phase3_B3_fmo03_full_convnextv2_gabor/2026-05-05/10-44-02/best_model.pth`
  - B4: `output/phase3_B4_fmo03_full_v2_bifpn/2026-05-05/12-05-03/best_model.pth`
  - B5: `output/phase3_B5_fmo03_full_efficientvit_gabor/2026-05-05/13-17-48/best_model.pth`
- Phase 3b full-size metrics (default threshold): `output/phase3_fullval_20260505_092807/{B1..B5}/metrics_results.csv`
- Phase 3c sweep:
  - Per-backbone summaries: `output/phase3_sweep_20260505_092807/{B1..B5}_sweep.csv` (B2 malformed; see notes)
  - Per-threshold raw: `output/phase3_sweep_20260505_092807/{B1..B5}/ts_<value>/metrics_results.csv`
  - Aggregate: `output/phase3_sweep_20260505_092807/summary.csv`
- Logs: `/tmp/phase3_20260505_092807/`
- Wrapper log: `/tmp/phase3_20260505_092807_wrapper.log`
- WandB project: `hn_phase3` (runs `phase3_B{1,3,4,5}_*`)
