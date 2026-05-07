#!/bin/bash
# Seed replication — re-runs every training from Phases 1-4 with a
# different seed (default 123, original was 42) to confirm rankings
# weren't coincidental. Mirrors the original experiment design exactly:
#
#   Phase 1: A1, A2, A3  — augmentation × crop strategy on convnext_baseline
#   Phase 2: L2          — loss ablation (L1 = A3 reused)
#   Phase 3: B1, B3, B4, B5 — backbone comparison (B2 = A3 reused)
#   Phase 4: B4_native   — B4 with native density_aware_fmo03_aux loss
#
# After all training: full-size eval at default ts=0.30 for Phase 1/3/4,
# threshold sweep for Phase 2 L2 + every Phase 3 backbone + Phase 4.
#
# Usage:
#   SEED=N bash run_seed_replication.sh [START_FROM]
#   SEED: seed override for all trainings (default 123)
#   START_FROM: skip variants before this number (1=A1 ... 9=B4_native)
#
# All artifacts are tagged with seed${SEED} prefix so seed=42 results
# stay intact for direct comparison.

set -e
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="$SCRIPT_DIR/configs/demo"
FULL_VAL_IMAGES="$SCRIPT_DIR/data_fmo03/val/Default"

SEED="${SEED:-123}"
START_FROM=${1:-1}
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/tmp/seed${SEED}_${TS}"
RESULTS_DIR="$SCRIPT_DIR/output/seed${SEED}_fullval_${TS}"
SWEEP_DIR="$SCRIPT_DIR/output/seed${SEED}_sweep_${TS}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$SWEEP_DIR"

WANDB_FLAG="${WANDB_FLAG:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-hn_replication_seed${SEED}}"

SWEEP_TS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70)

echo "============================================================"
echo "  Seed replication — re-run Phases 1-4 with SEED=$SEED"
echo "  Original seed:  42"
echo "  This run:       $SEED"
echo "  Wandb project:  $WANDB_PROJECT"
echo "  Logs:           $LOG_DIR"
echo "  Final val:      $RESULTS_DIR"
echo "  Sweep:          $SWEEP_DIR"
echo "  START_FROM:     $START_FROM"
echo "============================================================"

# ------------------------------------------------------------------
# Variants table.
# Format: NUM:TAG:CONFIG:DATASET_OVERRIDE:LOSS_OVERRIDE:LABEL
#   DATASET_OVERRIDE / LOSS_OVERRIDE = "-" means use the config default.
# ------------------------------------------------------------------
VARIANTS=(
  # Phase 1 — augmentation × crop on convnext_baseline + herdnet_fmo03
  "1:A1:fmo03_full_convnext_baseline:fmo03_new_clean:-:Phase1 A1 — minimal aug, pre-cropped"
  "2:A2:fmo03_full_convnext_baseline:fmo03_new_clean_augplus:-:Phase1 A2 — augplus, pre-cropped"
  "3:A3:fmo03_full_convnext_baseline:-:-:Phase1 A3 — augplus + ObjectAwareRandomCrop"
  # Phase 2 — loss ablation (L1 = A3, only L2 retrained)
  "4:L2:fmo03_full_convnext_baseline:-:density_aware_fmo03:Phase2 L2 — DensityAware loss"
  # Phase 3 — backbone comparison (B2 = A3, retrain B1/B3/B4/B5 with herdnet_fmo03)
  "5:B1:fmo03_full_dla34_timm:-:-:Phase3 B1 — DLA-34 timm"
  "6:B3:fmo03_full_convnextv2_gabor:-:herdnet_fmo03:Phase3 B3 — ConvNeXt-V2 + Gabor (loss override)"
  "7:B4:fmo03_full_v2_bifpn:-:herdnet_fmo03:Phase3 B4 — ConvNeXt + BiFPN + DeformConv (loss override)"
  "8:B5:fmo03_full_efficientvit_gabor:-:herdnet_fmo03:Phase3 B5 — EfficientViT + Gabor (loss override)"
  # Phase 4 — B4 with native loss (no override)
  "9:B4_native:fmo03_full_v2_bifpn:-:-:Phase4 — B4 with native density_aware_fmo03_aux"
)

# Some variants share a checkpoint via reuse:
# Phase 2 L1 = A3, Phase 3 B2 = A3.  Eval-only entries below.

# ------------------------------------------------------------------
# Phase A — train all 9 variants sequentially.
# ------------------------------------------------------------------
TRAIN_RESULTS=()
TRAIN_PASSED=0
TRAIN_FAILED=0

for entry in "${VARIANTS[@]}"; do
  IFS=':' read -r NUM TAG CONFIG DSO LO LABEL <<< "$entry"

  if [ "$NUM" -lt "$START_FROM" ]; then
    echo ""
    echo "  SKIP [$NUM/9] $TAG — $LABEL"
    TRAIN_RESULTS+=("SKIP  [$NUM/9] $TAG $LABEL")
    continue
  fi

  RUN_NAME="seed${SEED}_${TAG}_${CONFIG}"
  OUT_DIR="seed${SEED}_${TAG}_${CONFIG}"
  LOG_FILE="$LOG_DIR/${NUM}_${TAG}_train.log"

  EXTRA_OVERRIDES=("seed=$SEED")
  [ "$DSO" != "-" ] && EXTRA_OVERRIDES+=("datasets=$DSO")
  [ "$LO"  != "-" ] && EXTRA_OVERRIDES+=("losses=$LO")

  echo ""
  echo "------------------------------------------------------------"
  echo "  TRAIN [$NUM/9] $TAG — $LABEL"
  echo "  Config:    $CONFIG"
  echo "  Overrides: ${EXTRA_OVERRIDES[*]}"
  echo "  Wandb run: $RUN_NAME"
  echo "  Out:       output/$OUT_DIR/<date>/<time>/best_model.pth"
  echo "  Log:       $LOG_FILE"
  echo "  Started:   $(date '+%Y-%m-%d %H:%M:%S')"
  echo "------------------------------------------------------------"

  if python tools/train.py \
      --config-path "$CONFIG_DIR" \
      --config-name "$CONFIG" \
      "${EXTRA_OVERRIDES[@]}" \
      "wandb_flag=$WANDB_FLAG" \
      "wandb_project=$WANDB_PROJECT" \
      "wandb_run=$RUN_NAME" \
      "hydra.run.dir=./output/${OUT_DIR}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}" \
      2>&1 | tee "$LOG_FILE"; then
    echo "  => PASSED [$NUM/9] $TAG $(date '+%H:%M:%S')"
    TRAIN_RESULTS+=("PASS  [$NUM/9] $TAG $LABEL")
    TRAIN_PASSED=$((TRAIN_PASSED + 1))
  else
    echo "  => FAILED [$NUM/9] $TAG $(date '+%H:%M:%S')"
    TRAIN_RESULTS+=("FAIL  [$NUM/9] $TAG $LABEL")
    TRAIN_FAILED=$((TRAIN_FAILED + 1))
  fi
done

echo ""
echo "============================================================"
echo "  Phase A — Training Summary (seed=$SEED)"
echo "============================================================"
for r in "${TRAIN_RESULTS[@]}"; do echo "  $r"; done
echo "  Passed: $TRAIN_PASSED  Failed: $TRAIN_FAILED"

if [ "$TRAIN_FAILED" -gt 0 ]; then
  echo "  Some training runs failed — skipping eval phase."
  exit 1
fi

# ------------------------------------------------------------------
# Phase B — full-size eval at default ts=0.30 for everything.
# Reuse mapping: Phase 2 L1 = A3, Phase 3 B2 = A3 (same checkpoint).
# ------------------------------------------------------------------
find_best_model() {
  local base="$1"
  [ -d "$base" ] || return 1
  find "$base" -name "best_model.pth" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}

EVAL_VARIANTS=(
  # Phase 1 (A1, A2, A3 — convnext_baseline)
  "A1:fmo03_full_convnext_baseline:fmo03_new_clean:-:seed${SEED}_A1_fmo03_full_convnext_baseline"
  "A2:fmo03_full_convnext_baseline:fmo03_new_clean_augplus:-:seed${SEED}_A2_fmo03_full_convnext_baseline"
  "A3:fmo03_full_convnext_baseline:-:-:seed${SEED}_A3_fmo03_full_convnext_baseline"
  # Phase 2 (L2 only; L1 = A3 reused, no separate eval)
  "L2:fmo03_full_convnext_baseline:-:density_aware_fmo03:seed${SEED}_L2_fmo03_full_convnext_baseline"
  # Phase 3 (B1, B3, B4, B5 — B2 = A3 reused)
  "B1:fmo03_full_dla34_timm:-:-:seed${SEED}_B1_fmo03_full_dla34_timm"
  "B3:fmo03_full_convnextv2_gabor:-:herdnet_fmo03:seed${SEED}_B3_fmo03_full_convnextv2_gabor"
  "B4:fmo03_full_v2_bifpn:-:herdnet_fmo03:seed${SEED}_B4_fmo03_full_v2_bifpn"
  "B5:fmo03_full_efficientvit_gabor:-:herdnet_fmo03:seed${SEED}_B5_fmo03_full_efficientvit_gabor"
  # Phase 4
  "B4_native:fmo03_full_v2_bifpn:-:-:seed${SEED}_B4_native_fmo03_full_v2_bifpn"
)

echo ""
echo "============================================================"
echo "  Phase B — Full-Size Eval (default ts=0.30)"
echo "============================================================"
for entry in "${EVAL_VARIANTS[@]}"; do
  IFS=':' read -r TAG CONFIG DSO LO OUT_DIR <<< "$entry"

  MODEL_PATH=$(find_best_model "$SCRIPT_DIR/output/$OUT_DIR" || true)
  RUN_OUT="$RESULTS_DIR/${TAG}"
  LOG_FILE="$LOG_DIR/${TAG}_fullval.log"

  EVAL_OVERRIDES=("wandb_flag=False")
  [ "$DSO" != "-" ] && EVAL_OVERRIDES+=("datasets=$DSO")
  [ "$LO"  != "-" ] && EVAL_OVERRIDES+=("losses=$LO")

  if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "  $TAG: missing best_model.pth under output/$OUT_DIR — skipping"
    continue
  fi

  mkdir -p "$RUN_OUT"
  echo "  $TAG @ ts=0.30 ..."
  (cd "$RUN_OUT" && python "$SCRIPT_DIR/tools/infer.py" \
      --config-dir "$CONFIG_DIR" \
      --config-name "$CONFIG" \
      --images "$FULL_VAL_IMAGES" \
      --model "$MODEL_PATH" \
      --evaluate \
      --overrides "${EVAL_OVERRIDES[@]}") \
      > "$LOG_FILE" 2>&1 || echo "    eval failed for $TAG"
done

# ------------------------------------------------------------------
# Phase C — threshold sweep for L2 + Phase-3 backbones + Phase-4.
# (No sweep for A1/A2 — Phase 1 didn't sweep them and the conclusion
#  doesn't depend on tuning since they lose by large margins.)
# ------------------------------------------------------------------
SWEEP_VARIANTS=(
  "L2:fmo03_full_convnext_baseline:-:density_aware_fmo03:seed${SEED}_L2_fmo03_full_convnext_baseline"
  "B1:fmo03_full_dla34_timm:-:-:seed${SEED}_B1_fmo03_full_dla34_timm"
  "B2:fmo03_full_convnext_baseline:-:-:seed${SEED}_A3_fmo03_full_convnext_baseline"
  "B3:fmo03_full_convnextv2_gabor:-:herdnet_fmo03:seed${SEED}_B3_fmo03_full_convnextv2_gabor"
  "B4:fmo03_full_v2_bifpn:-:herdnet_fmo03:seed${SEED}_B4_fmo03_full_v2_bifpn"
  "B5:fmo03_full_efficientvit_gabor:-:herdnet_fmo03:seed${SEED}_B5_fmo03_full_efficientvit_gabor"
  "B4_native:fmo03_full_v2_bifpn:-:-:seed${SEED}_B4_native_fmo03_full_v2_bifpn"
)

SWEEP_SUMMARY="$SWEEP_DIR/summary.csv"
echo "tag,config,adapt_ts,f1,precision,recall,mae,rmse,ap" > "$SWEEP_SUMMARY"

echo ""
echo "============================================================"
echo "  Phase C — Threshold sweep"
echo "  adapt_ts ∈ { ${SWEEP_TS[*]} }"
echo "============================================================"

for entry in "${SWEEP_VARIANTS[@]}"; do
  IFS=':' read -r TAG CONFIG DSO LO OUT_DIR <<< "$entry"

  MODEL_PATH=$(find_best_model "$SCRIPT_DIR/output/$OUT_DIR" || true)
  if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "  $TAG: missing best_model.pth — skipping sweep"
    continue
  fi

  PER_TAG_CSV="$SWEEP_DIR/${TAG}_sweep.csv"
  echo "tag,config,adapt_ts,f1,precision,recall,mae,rmse,ap" > "$PER_TAG_CSV"

  EXTRA=("wandb_flag=False")
  [ "$DSO" != "-" ] && EXTRA+=("datasets=$DSO")
  [ "$LO"  != "-" ] && EXTRA+=("losses=$LO")

  for ATS in "${SWEEP_TS[@]}"; do
    SUB="$SWEEP_DIR/${TAG}/ts_${ATS}"
    LOG_FILE="$LOG_DIR/${TAG}_sweep_${ATS}.log"
    mkdir -p "$SUB"

    echo "  $TAG @ adapt_ts=$ATS ..."
    (cd "$SUB" && python "$SCRIPT_DIR/tools/infer.py" \
        --config-dir "$CONFIG_DIR" \
        --config-name "$CONFIG" \
        --images "$FULL_VAL_IMAGES" \
        --model "$MODEL_PATH" \
        --evaluate \
        --overrides "${EXTRA[@]}" \
          "training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=$ATS") \
        > "$LOG_FILE" 2>&1 || echo "    sweep failed at $ATS (continuing)"

    CSV="$SUB/metrics_results.csv"
    if [ -f "$CSV" ]; then
      awk -F, -v tag="$TAG" -v cfg="$CONFIG" -v ats="$ATS" '$1=="binary" {
        printf "%s,%s,%s,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f\n",
               tag, cfg, ats, $6, $5, $4, $8, $11, $12 }' "$CSV" \
        | tee -a "$PER_TAG_CSV" "$SWEEP_SUMMARY" >/dev/null
    fi
  done
done

# ------------------------------------------------------------------
# Final report — best-threshold per variant + side-by-side vs seed=42.
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Replication summary (seed=$SEED) — best operating points"
echo "============================================================"
printf "  %-10s  %-38s  %-15s  %-15s  %-18s\n" \
       "TAG" "CONFIG" "BEST_F1@ts" "BEST_MAE@ts" "BEST_REC_at_P>=.85"
for entry in "${SWEEP_VARIANTS[@]}"; do
  IFS=':' read -r TAG CONFIG DSO LO OUT_DIR <<< "$entry"
  PER_TAG_CSV="$SWEEP_DIR/${TAG}_sweep.csv"
  [ -f "$PER_TAG_CSV" ] || continue
  python3 - "$TAG" "$CONFIG" "$PER_TAG_CSV" <<'PY'
import sys, csv
tag, cfg, csv_path = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(csv_path)))
if not rows:
    print(f"  {tag:<10}  {cfg:<38}  no sweep rows")
    sys.exit(0)
def f(r,k): return float(r[k])
bf1 = max(rows, key=lambda r: f(r,'f1'))
bma = min(rows, key=lambda r: f(r,'mae'))
hp = [r for r in rows if f(r,'precision') >= 0.85]
brp = max(hp, key=lambda r: f(r,'recall')) if hp else None
def fmt(r): return f"F1={f(r,'f1'):.3f}@{r['adapt_ts']}"
def fmt_mae(r): return f"MAE={f(r,'mae'):.2f}@{r['adapt_ts']}"
def fmt_rec(r): return f"R={f(r,'recall'):.3f}@{r['adapt_ts']}" if r else "(none)"
print(f"  {tag:<10}  {cfg:<38}  {fmt(bf1):<15}  {fmt_mae(bma):<15}  {fmt_rec(brp):<18}")
PY
done

echo ""
echo "  Logs:    $LOG_DIR"
echo "  Results: $RESULTS_DIR"
echo "  Sweep:   $SWEEP_DIR"
echo "  Summary CSV: $SWEEP_SUMMARY"
echo "============================================================"
echo "  Compare against original (seed=42) results in:"
echo "    docs/benchmarks/phase{1,2,2b,3,4}*.md"
echo "============================================================"
