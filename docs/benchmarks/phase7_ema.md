# Phase 7 — weight EMA on B4 (negative result)

**Run date**: 2026-05-07 15:05 → 21:40.
**Branch**: `convnext_extension` @ `c79ee69`.
**Hardware**: single RTX 4080 SUPER.
**Script**: [`run_phase7_ema.sh`](../../run_phase7_ema.sh).

## Hypothesis

The lit-review at the end of Phase 5 ranked **Run #1** as a "free lunch" stack of three inference- and training-side tricks: TTA (✓ tested in Phase 6), 3-seed ensemble (✓ tested in Phase 6), and **EMA on model weights**. Phase 6 already confirmed TTA and ensemble work — both individually and stacked. Phase 7 closes the third leg: does maintaining an exponential moving average of B4's weights during training (and evaluating / saving the EMA copy) push the production stack further past Phase 6's `F1=0.9634 / MAE=0.67`?

Standard EMA reasoning: late-epoch updates are noisy; smoothing them via `θ_ema ← d·θ_ema + (1-d)·θ` finds a flatter minimum and improves generalisation. Reported gains in literature: +0.5–1.5 % for detection on long training runs, and improved ensembling because EMA reduces per-model variance.

## Setup

Identical to Phase 5 B4 except for EMA — same `fmo03_full_v2_bifpn` config, `losses=herdnet_fmo03` override, `fmo03_new_objcrop_augplus` dataset, 30 epochs, batch_size=4, lr=8e-5, 3 seeds {42, 123, 7}.

EMA implementation in `animaloc/models/utils.py` and `animaloc/train/trainers.py`:
- `ModelEMA` class with timm-style decay warmup: `d_t = min(0.9999, (1+t)/(10+t))` per step
- Update after every `optimizer.step()`
- `_prepare_evaluator()` points `evaluator.model` at the EMA copy
- `best_model.pth` saves EMA's `state_dict()` so downstream tools (`tools/infer.py`, `tools/ensemble_infer.py`) load it transparently

GPU memory cost: +1.0 GiB during training (shadow copy of weights). 30-epoch wall-clock unchanged at ~1.5 h/seed.

## Per-seed results (best-tuned full-size F1)

| seed | Phase-5 best F1 (no EMA) | Phase-7 best F1 (EMA) | Δ | Phase-5 best MAE | Phase-7 best MAE | Δ |
|---|---|---|---|---|---|---|
| 42 | 0.955 | 0.9521 | −0.003 | 0.75 | 1.08 | +0.33 |
| 123 | 0.942 | 0.9402 | −0.002 | 0.83 | 0.83 | 0.00 |
| 7 | 0.938 | 0.9429 | +0.005 | 1.08 | 0.92 | −0.16 |
| **mean** | **0.945** | **0.945** | **0.000** | **0.89** | **0.94** | **+0.05** |

Within seed noise. EMA helped seed=7 by ~0.5 pt F1; hurt seed=42 by 0.3 pt F1 and 0.33 MAE. No systematic gain.

## Ensemble results (3-seed EMA ensemble)

The bigger surprise: the 3-seed ensemble of EMA-trained checkpoints under-performs the 3-seed ensemble of plain-trained checkpoints.

| | Phase-6 ensemble (no EMA) | Phase-7 ensemble (EMA) | Δ |
|---|---|---|---|
| best F1, no TTA | 0.9605 @ ts=0.40 | 0.9540 @ ts=0.50 | **−0.0065** |
| best F1, with TTA | **0.9634** @ ts=0.35 | 0.9492 @ ts=0.35 | **−0.0142** |
| best MAE, no TTA | **0.67** @ ts=0.30 | 0.75 @ ts=0.25 | +0.08 |
| best MAE, with TTA | 0.75 @ ts=0.25 | 0.75 @ ts=0.30 | 0.00 |

Phase 6's F1=0.9634 stands. EMA + ensemble + TTA peaked at 0.9492 — a clean regression of 0.014 F1.

## Why EMA didn't help (and made the ensemble slightly worse)

**Per-seed neutrality**: EMA's smoothing benefit is largest when training noise is large relative to the signal — long runs, large datasets, or high learning rates. Our regime is the opposite: 30 epochs (short), 1425 effective examples per epoch (small), already heavily augmented (augplus + ObjectAwareRandomCrop). Late-epoch noise is small to begin with, so smoothing it provides little additional generalisation.

**Ensemble regression**: the −0.014 F1 in the ensemble is the more interesting finding. The most plausible explanation is **reduced ensemble diversity**. Ensemble lift comes from members that make uncorrelated errors; averaging away their independent noise. EMA averages each seed's late-epoch trajectory toward a similarly-flat minimum, which makes the 3 seeds' final weights more correlated than they would be without smoothing. The gain in per-member calibration is eaten by the loss in diversity, and net F1 drops slightly.

This is consistent with reports in the literature that EMA + ensembling can interact poorly when both are tuned for the same source of variance — a known caveat noted, e.g., in `Switch EMA: A Free Lunch for Better Flatness and Sharpness` (Pan et al., arXiv 2402.09240, 2024) discussion of when SEMA helps versus plain EMA.

## What about the per-seed best F1 anomaly at ts=0.35?

EMA seed=42 best F1 = 0.9521 @ ts=0.35 — that's actually higher than non-EMA seed=42 at the same threshold (0.942 @ ts=0.35 from Phase 5). EMA made the *low-threshold* operating points sharper for that seed. But the *best-overall* F1 for non-EMA seed=42 was 0.955 @ ts=0.50 (sharper still), so EMA shifted the optimum downward in threshold without raising the global maximum. Thread-level effect, no tier-level effect.

## Verdict

**Production stack remains Phase 6.** EMA-trained checkpoints (`output/phase7_B4_ema_s{42,123,7}/...`) are not better than the non-EMA Phase-3 / Phase-5 B4 checkpoints already in use. They can be archived but not deployed.

The lit-review's "Run #1" stack is now resolved:
- ✅ TTA — confirmed (+0.01–0.02 F1)
- ✅ 3-seed ensemble — confirmed (+0.02 F1, −0.16 MAE vs single seed)
- ❌ EMA — no improvement on this dataset

## What's worth keeping

The EMA implementation itself (`ModelEMA` in `animaloc/models/utils.py`, plumbing in `Trainer.__init__` / `_train` / `_prepare_evaluator` / `_save_checkpoint` / `animaloc/utils/train.py`) is small, correct, and fully optional (defaults to `None`). It adds <50 lines of code and no overhead when unused. **Keep it merged** — future longer training runs or larger datasets may benefit, and the code is non-invasive.

## Artifacts

- EMA-trained checkpoints: `output/phase7_B4_ema_s{42,123,7}/2026-05-07/<HH-MM-SS>/best_model.pth`
- Sweep CSVs:
  - T1 per-seed EMA: `output/phase7_sweep_20260507_150537/T1_ema_per_seed.csv`
  - T2 EMA ensemble (no TTA): `output/phase7_sweep_20260507_150537/T2_ema_ensemble_no_tta.csv`
  - T3 EMA ensemble + TTA: `output/phase7_sweep_20260507_150537/T3_ema_ensemble_tta.csv`
- Per-threshold raw: `output/phase7_sweep_20260507_150537/<TAG>/ts_<value>_tta<bool>/metrics_results.csv`
- Logs: `/tmp/phase7_20260507_150537/`
- Wrapper log: `/tmp/phase7_20260507_150537_wrapper.log`
- WandB project: `hn_phase7_ema`, runs `phase7_B4_ema_s{42,123,7}`
