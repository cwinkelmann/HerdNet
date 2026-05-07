# Phase 5 — seed replication (seeds 42 / 123 / 7)

**Run dates**: seed=42 across 2026-05-04 / 05; seed=123 from 2026-05-05 19:04 to 2026-05-06; seed=7 from 2026-05-06 08:22 to 2026-05-07.
**Branch**: `convnext_extension` @ `a0cb969`.
**Hardware**: single RTX 4080 SUPER.
**Script**: [`run_seed_replication.sh`](../../run_seed_replication.sh).

## Why this phase

Phases 1–4 established a model selection and recommended `B4 + herdnet_fmo03 + adapt_ts=0.35` for production — all single-seed (42). Phase 5 re-runs every Phase-1-through-4 training with two additional seeds (123, 7) to test whether the rankings held under seed variance.

Output paths tagged `seed{42,123,7}_*` so all three runs' artifacts coexist for direct comparison.

## Setup

9 trainings + 7 threshold sweeps (11 thresholds each) + 9 default-threshold full-size evals — same matrix per seed.

| Tag | Config | Loss override | Dataset override |
|---|---|---|---|
| A1 | `fmo03_full_convnext_baseline` | – | `fmo03_new_clean` |
| A2 | `fmo03_full_convnext_baseline` | – | `fmo03_new_clean_augplus` |
| A3 | `fmo03_full_convnext_baseline` | – | – |
| L2 | `fmo03_full_convnext_baseline` | `density_aware_fmo03` | – |
| B1 | `fmo03_full_dla34_timm` | – | – |
| B3 | `fmo03_full_convnextv2_gabor` | `herdnet_fmo03` | – |
| B4 | `fmo03_full_v2_bifpn` | `herdnet_fmo03` | – |
| B5 | `fmo03_full_efficientvit_gabor` | `herdnet_fmo03` | – |
| B4_native | `fmo03_full_v2_bifpn` | – | – |

Phase 2 L1 = A3 (reused per seed). Phase 3 B2 = A3 (reused per seed).

## Phase 1 — augmentation × crop strategy (default ts=0.30)

| Tag | seed=42 | seed=123 | seed=7 | mean | spread |
|---|---|---|---|---|---|
| A1 — minimal aug, pre-cropped | 0.7648 | 0.7200 | 0.6489 | 0.711 | 0.116 |
| A2 — augplus, pre-cropped | 0.8000 | 0.8535 | 0.8667 | 0.840 | 0.067 |
| **A3 — augplus + ObjectAwareRandomCrop** | **0.9056** | **0.9314** | **0.9151** | **0.917** | 0.026 |

**A3 wins all 3 seeds by ≥0.05 F1 over A2 and ≥0.14 F1 over A1.** The crop-strategy + augplus combination is the only Phase-1 conclusion that should ever have been called robust, and it is.

## Best-tuned full-size F1 across 3 seeds

| Tag | seed=42 | seed=123 | seed=7 | mean | std |
|---|---|---|---|---|---|
| L2 | 0.889 | 0.935 | **0.948** | 0.924 | 0.025 |
| B1 | 0.913 | 0.913 | 0.903 | 0.910 | 0.005 |
| B2 (=A3=L1) | 0.924 | 0.936 | 0.931 | 0.930 | 0.005 |
| **B3** | 0.952 | **0.958** | 0.935 | **0.948** | 0.010 |
| **B4** | **0.955** | 0.942 | 0.938 | 0.945 | 0.007 |
| B5 | 0.918 | 0.894 | 0.913 | 0.908 | 0.010 |
| **B4_native** | 0.940 | 0.930 | **0.947** | 0.939 | 0.007 |

## Best-tuned full-size MAE across 3 seeds

| Tag | seed=42 | seed=123 | seed=7 | mean |
|---|---|---|---|---|
| L2 | 1.92 | 1.33 | **0.92** | 1.39 |
| B1 | 1.67 | 1.08 | 1.25 | 1.33 |
| B2 | 1.25 | 1.17 | 1.50 | 1.31 |
| B3 | 1.00 | 1.17 | 1.17 | 1.11 |
| **B4** | **0.75** | **0.83** | 1.08 | **0.89** |
| B5 | 0.83 | 1.58 | 1.17 | 1.19 |
| B4_native | 1.17 | 1.00 | **0.75** | 0.97 |

## Per-seed best-F1 ranking

```
seed=42:  B4 > B3 > B4_native > B2 > B5 > B1 > L2
seed=123: B3 > B4 > B2 > L2 > B4_native > B1 > B5
seed=7:   L2 ≈ B4_native > B4 > B3 > B2 > B5 ≈ B1
```

The "winner" depends on which seed you ask. Across 3 seeds **B3 has the highest mean (0.948), narrowly above B4 (0.945) and B4_native (0.939) — but every gap is below the cross-seed standard deviation of every model except B1/B2.** In other words: at this resolution we can't reliably distinguish B3, B4, B4_native, and L2.

## What changed when seed=7 came in

The seed=7 run shifted the picture in three concrete ways:

1. **L2 (DensityAware) keeps climbing**: 0.889 → 0.935 → **0.948**. It now has the highest single-seed F1 at seed=7 (tied with B4_native). Phase 2's "L1 clearly wins" was a seed-42 artifact: across 3 seeds, **L2's mean F1 (0.924) is below L1/B2's (0.930) by 0.006 — within noise**, not a clear win for L1.

2. **B4_native is competitive**: at seed=7 it beats B4-override on F1 by +0.009 and on MAE by 0.33. Phase 4's "native is worse" was based on 2/3 seeds. **Mean F1 over 3 seeds: 0.939 native vs 0.945 override — within noise.** Mean MAE: 0.97 native vs 0.89 override — override still leads on average. The conclusion should now be *"native is approximately equal to override on F1, slightly worse on average MAE"*, not *"native loses"*.

3. **B4 dropped on MAE at seed=7** (0.75 → 0.83 → 1.08). It still has the best mean MAE (0.89) by a meaningful margin over B4_native (0.97), so the **counting recommendation survives**. But the per-seed swing of 0.33 means single-seed MAE numbers in this codebase are not reliable to two decimals.

## What's robust across 3 seeds

✅ **A3 wins Phase 1.** Crop strategy + augplus over pre-cropped + minimal aug is the right default (ΔF1 ≥0.05 to ≥0.14 at every seed).
✅ **B1 (DLA-34) and B5 (EfficientViT) lag.** Mean F1 0.910 / 0.908 vs ConvNeXt family ≥0.930 — outside noise.
✅ **B5 is high-variance.** ΔF1 = 0.024 across seeds, ΔMAE = 0.75. Don't trust EfficientViT single-seed numbers.
✅ **B4 has the best mean MAE** (0.89). Counting champion across seeds. MAE-optimal threshold stable at `ts=0.35` at every seed.
✅ **The MAE-optimal threshold (~0.35) is more stable than the F1-optimal threshold.** F1-optimal `adapt_ts` drifted 0.40 → 0.40 → 0.50 for B2; 0.40 → 0.50 → 0.60 for B3; 0.50 → 0.70 → 0.60 for B4. Counting is the more stable use case.

## What was a seed-42 artifact

⚠️ **B3 vs B4 best-F1 ordering.** Three seeds, three different rankings. They are practically tied.
⚠️ **L1 strictly beats L2 (Phase 2).** Cross-seed mean is L1 0.930 vs L2 0.924 — within noise. The strong claim should be retracted.
⚠️ **B4_native is worse than B4-override (Phase 4).** Cross-seed F1 mean is 0.939 vs 0.945 — within noise. Cross-seed MAE mean still favours override (0.89 vs 0.97), so the production recommendation for counting holds, but the F1 conclusion was overstated.
⚠️ **B5 MAE 0.83 was anomalous.** seed=123/7 say 1.58 / 1.17 — the 0.83 was a lucky seed.

## Production recommendation (revised, 3 seeds in)

For **counting** (MAE-optimal): **`fmo03_full_v2_bifpn` (B4) + `losses=herdnet_fmo03` + `adapt_ts=0.35`.** Mean MAE 0.89 across seeds; threshold is stable. **This is the production default.**

For **F1**: **B3 / B4 / B4_native / L2 are practically tied** (mean F1 within 0.02). Pick on secondary criteria:
- **B4 (override)** — best mean MAE; best F1-tied; standard architecture. Default choice.
- **B3** — strongest at high precision (hits P=1.000 at high `adapt_ts` in seed=42); useful for human-review pipelines.
- **B4_native** — equivalent F1 and MAE to override on average; uses 3× more GPU memory during training. Only worth picking if the deep-supervision aux heads are needed for downstream multi-task purposes.
- **L2** — surprisingly competitive; would be the right choice if the dataset were known to have heavy clustering. With only 3 seeds we can't say it's worse.

For **screening (recall ≥ 0.85)**: any of {B3, B4, B4_native, L2} is fine; tune `adapt_ts ∈ [0.15, 0.25]`.

## A practical rule of thumb

**Don't act on single-seed F1 differences below ~0.02.** Across the 3 seeds we have, every cross-seed swing on the top-4 models was in that band. Published or shipped rankings should either (a) average ≥3 seeds, or (b) acknowledge the noise floor.

## Artifacts

Per-seed (replace `<S>` with 42, 123, or 7):

- Per-variant checkpoints:
  - seed=42: `output/phase{1,2,3,4}_*/...` (original layout)
  - seed=123 / seed=7: `output/seed<S>_<TAG>_<config>/<date>/<HH-MM-SS>/best_model.pth`
- Default-threshold full-size evals: `output/seed<S>_fullval_<ts>/<TAG>/metrics_results.csv`
- Sweep CSVs: `output/seed<S>_sweep_<ts>/{<TAG>_sweep.csv, summary.csv}`
- Per-threshold raw: `output/seed<S>_sweep_<ts>/<TAG>/ts_<value>/metrics_results.csv`
- Per-run training logs: `/tmp/seed<S>_<ts>/`
- Wrapper logs:
  - seed=123: `/tmp/seed123_replication_20260505_190449_wrapper.log`
  - seed=7: `/tmp/seed7_replication_20260506_082154_wrapper.log`
- WandB projects: `hn_replication_seed123`, `hn_replication_seed7` (seed=42 was split across `hn_phase{1,2,3,4}`)
