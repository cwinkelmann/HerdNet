# Phase 10 — stitcher overlap sweep (negative result + a stitcher bug)

**Run date**: 2026-05-08 00:01 → 00:33.
**Branch**: `convnext_extension` @ `bb66e2e`.
**Hardware**: single RTX 4080 SUPER.
**Script**: [`run_phase10_overlap_sweep.sh`](../../run_phase10_overlap_sweep.sh).

## Hypothesis

[Phase 9](phase9_error_analysis.md) found that ~30 % of remaining errors on the Phase-6 production stack clustered at image borders — 3 edge-truncation FNs and 3 corner/edge high-confidence FPs out of 20 total errors. The reasoning at the end of that doc:

> Roughly 6 of 20 errors (30 %) are clustered at image borders. This is a single-fix opportunity: bump the stitcher overlap from 120 to ≥256, or pad the input.

Phase 10 tests that hypothesis: keep everything else fixed, sweep `training_settings.stitcher.kwargs.overlap` over {120, 192, 256, 320, 384}, see whether the edge-error cluster shrinks.

## Setup

- Model: 3-seed B4 ensemble (Phase-6 best-MAE config)
- Threshold: `adapt_ts=0.30` (Phase-6 best-MAE)
- All else: identical to Phase 6 T2

## Results

| overlap | F1 | precision | recall | MAE | RMSE | AP | detections |
|---|---|---|---|---|---|---|---|
| **120** (default) | **0.9451** | 0.9399 | 0.9503 | **0.67** | 0.91 | 0.9491 | 183 |
| 192 | 0.9396 | 0.9344 | 0.9448 | 0.67 | 1.00 | 0.9443 | 183 |
| 256 | 0.9348 | 0.9198 | 0.9503 | 0.83 | 1.15 | 0.9492 | 187 |
| 320 | **broken** | 0.000 | 0.094 | 57712.92 | 57712.92 | 0.0000 | **692,736** |
| 384 | **broken** | 0.000 | 0.094 | 43376.92 | 43376.92 | 0.0001 | **520,704** |

## Hypothesis falsified

Increasing overlap **does not help recall** and slightly hurts F1:
- 120 → 192: F1 −0.005, MAE flat
- 120 → 256: F1 −0.010, MAE +0.16

The edge-cluster errors I attributed to "low overlap near image borders" in Phase 9 are not actually caused by tile coverage. Most of those errors are iguanas literally cut off by the *image frame* (y < 10 or y > 3640) — adding more tile context inside the image doesn't help when the animal is partially outside it.

The proper fix for those errors would be to **reflection-pad the input image** before stitching, so partial iguanas at the image border get a synthetic mirrored context. The stitcher's overlap parameter is the wrong knob.

## Bonus finding — stitcher bug at high overlap

`overlap ∈ {320, 384}` produced spectacular nonsense: **520k–693k detections** on a 12-frame val set with 181 GT iguanas, MAE in the tens of thousands. F1 ≈ 0.

Patch size is 512, so:
- overlap=320 → stride=192 → ~29 × 19 = 551 tiles per 5472×3648 frame
- overlap=384 → stride=128 → ~43 × 29 = 1247 tiles per frame

With the stitcher's `mean` reduction, every output pixel should be averaged across all overlapping tiles. The fact that detection counts explode by ~3000× suggests the averaging is no longer normalising by the per-pixel contributor count — heatmaps stack instead of average, peaks appear everywhere, LMDS picks them all up.

This is a silent stitcher failure mode: instead of erroring out, it produces structurally invalid outputs that the rest of the pipeline accepts. Worth a sanity-check assertion in `HerdNetStitcher` (e.g., warn when `overlap >= patch_size * 0.5` or assert that the per-pixel count map is bounded). Filed as a TODO in the code-review pile.

## Verdict

- **Production overlap stays at 120.** The default is already optimal.
- The edge-error cluster from Phase 9 needs a different fix — reflection padding, not overlap. **Listed as a future experiment but not pursued in this round** since the [Phase 8 cross-arch ensemble](phase8_cross_arch_ensemble.md) recovered 2 of the 3 edge-truncation FNs anyway, and the remaining edge errors are now a smaller share of an already-tighter error budget.
- **Stitcher silent-fail bug**: noted, low priority (no production impact since we don't run with overlap that high), but should eventually have a guard.

## Artifacts

- Sweep CSV: `output/phase10_sweep_20260507_235411/overlap_sweep.csv`
- Per-overlap raw: `output/phase10_sweep_20260507_235411/overlap_<v>/{metrics_results.csv, detections.csv, plots/}`
- Logs: `/tmp/phase10_20260507_235411/`
- Wrapper log: `/tmp/phase10_20260507_233811_pending_wrapper.log`
