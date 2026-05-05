#!/bin/bash
# Phase 2 — loss-function ablation.
#
# Holds backbone (HerdNetTimmConvNext / convnext_baseline) and dataset
# (fmo03_new_objcrop_augplus, the A3 winner of Phase 1) fixed; varies
# the heatmap loss:
#
#   L1: herdnet_fmo03           — Focal + weighted CE   (= Phase-1 A3, REUSED)
#   L2: density_aware_fmo03     — DensityAware + weighted CE
#
# L3 (density_aware_fmo03_aux) requires a model that emits aux P3/P4
# outputs. HerdNetTimmConvNext only returns (heatmap, cls_out), so L3
# is deferred to Phase 3 where backbone is varied (V2 / V3 emit the
# aux heads). See docs/benchmarks/phase1_augmentation_crop.md.
#
# Per-epoch metrics on the cropped val set are tracked for convergence
# only — do not use them for selection. Final stitched full-size eval
# is the metric that matters; it runs after training on each new
# best_model.pth and a 3-way (A3, L1=A3-rerun-or-reuse, L2) comparison
# table is printed at the end.
#
# Usage: bash run_phase2.sh [START_FROM]
#   START_FROM: optional, skip variants before this number (1=L2 since L1=A3)

set -e
export LC_NUMERIC=C   # so the printf %f at the end uses '.', not ','

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="$SCRIPT_DIR/configs/demo"
FULL_VAL_IMAGES="$SCRIPT_DIR/data_fmo03/val/Default"
START_FROM=${1:-1}
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/tmp/phase2_${TS}"
RESULTS_DIR="$SCRIPT_DIR/output/phase2_fullval_${TS}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

WANDB_FLAG="${WANDB_FLAG:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-hn_phase2}"

# Phase-1 A3 best_model — used as L1 baseline so we don't retrain.
A3_OUTPUT_DIR="$SCRIPT_DIR/output/phase1_A3_convnext_baseline"
A3_FULLVAL_DIR="$SCRIPT_DIR/output/phase1_fullval_20260504_210213/A3"

echo "============================================================"
echo "  Phase 2 — loss-function ablation"
echo "  Backbone:    HerdNetTimmConvNext (convnext_baseline)"
echo "  Dataset:     fmo03_new_objcrop_augplus (A3 winner)"
echo "  Epochs:      30"
echo "  Wandb:       flag=$WANDB_FLAG project=$WANDB_PROJECT"
echo "  Logs:        $LOG_DIR"
echo "  Final val:   $RESULTS_DIR"
echo "  L1 baseline: REUSE Phase-1 A3 (best_model under $A3_OUTPUT_DIR)"
echo "  START_FROM:  $START_FROM"
echo "============================================================"

# Only L2 needs a fresh training; L1 is the existing A3 run.
VARIANTS=(
  "1:L2:density_aware_fmo03:DensityAwareHerdNetLoss + weighted CE"
)

# ------------------------------------------------------------------
# Phase 2a — train new variants sequentially.
# ------------------------------------------------------------------
TRAIN_RESULTS=()
TRAIN_PASSED=0
TRAIN_FAILED=0

for entry in "${VARIANTS[@]}"; do
  IFS=':' read -r NUM TAG LOSS_CFG LABEL <<< "$entry"

  if [ "$NUM" -lt "$START_FROM" ]; then
    echo ""
    echo "  SKIP [$NUM] $TAG — $LABEL"
    TRAIN_RESULTS+=("SKIP  [$NUM] $TAG $LABEL")
    continue
  fi

  RUN_NAME="phase2_${TAG}_${LOSS_CFG}"
  OUT_DIR="phase2_${TAG}_convnext_baseline"
  LOG_FILE="$LOG_DIR/${NUM}_${TAG}_train.log"

  echo ""
  echo "------------------------------------------------------------"
  echo "  TRAIN [$NUM] $TAG — $LABEL"
  echo "  Loss:       $LOSS_CFG"
  echo "  Wandb run:  $RUN_NAME"
  echo "  Out:        output/$OUT_DIR/<date>/<time>/best_model.pth"
  echo "  Log:        $LOG_FILE"
  echo "  Started:    $(date '+%Y-%m-%d %H:%M:%S')"
  echo "------------------------------------------------------------"

  if python tools/train.py \
      --config-path "$CONFIG_DIR" \
      --config-name fmo03_full_convnext_baseline \
      "losses=$LOSS_CFG" \
      "wandb_flag=$WANDB_FLAG" \
      "wandb_project=$WANDB_PROJECT" \
      "wandb_run=$RUN_NAME" \
      "hydra.run.dir=./output/${OUT_DIR}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}" \
      2>&1 | tee "$LOG_FILE"; then
    echo "  => PASSED [$NUM] $TAG $(date '+%H:%M:%S')"
    TRAIN_RESULTS+=("PASS  [$NUM] $TAG $LABEL")
    TRAIN_PASSED=$((TRAIN_PASSED + 1))
  else
    echo "  => FAILED [$NUM] $TAG $(date '+%H:%M:%S')"
    TRAIN_RESULTS+=("FAIL  [$NUM] $TAG $LABEL")
    TRAIN_FAILED=$((TRAIN_FAILED + 1))
  fi
done

echo ""
echo "============================================================"
echo "  Phase 2a — Training Summary"
echo "============================================================"
for r in "${TRAIN_RESULTS[@]}"; do echo "  $r"; done
echo "  Passed: $TRAIN_PASSED  Failed: $TRAIN_FAILED"

if [ "$TRAIN_FAILED" -gt 0 ]; then
  echo "  Some training runs failed — skipping full-size validation."
  exit 1
fi

# ------------------------------------------------------------------
# Phase 2b — full-size stitched validation on each new best_model.pth.
# ------------------------------------------------------------------
find_best_model() {
  local base="$1"
  [ -d "$base" ] || return 1
  find "$base" -name "best_model.pth" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}

VAL_RESULTS=()
VAL_PASSED=0
VAL_FAILED=0

for entry in "${VARIANTS[@]}"; do
  IFS=':' read -r NUM TAG LOSS_CFG LABEL <<< "$entry"

  if [ "$NUM" -lt "$START_FROM" ]; then continue; fi

  OUT_DIR="phase2_${TAG}_convnext_baseline"
  MODEL_PATH=$(find_best_model "$SCRIPT_DIR/output/$OUT_DIR" || true)
  RUN_OUT="$RESULTS_DIR/${TAG}"
  LOG_FILE="$LOG_DIR/${NUM}_${TAG}_fullval.log"

  echo ""
  echo "------------------------------------------------------------"
  echo "  FULLVAL [$NUM] $TAG — $LABEL"
  echo "  Model:   ${MODEL_PATH:-<none>}"
  echo "  Out:     $RUN_OUT"
  echo "  Log:     $LOG_FILE"
  echo "------------------------------------------------------------"

  if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "  => MISSING [$NUM] $TAG"
    VAL_RESULTS+=("MISSING [$NUM] $TAG $LABEL")
    continue
  fi

  mkdir -p "$RUN_OUT"
  if (cd "$RUN_OUT" && python "$SCRIPT_DIR/tools/infer.py" \
      --config-dir "$CONFIG_DIR" \
      --config-name fmo03_full_convnext_baseline \
      --images "$FULL_VAL_IMAGES" \
      --model "$MODEL_PATH" \
      --evaluate \
      --overrides \
        "losses=$LOSS_CFG" \
        "wandb_flag=False") \
      2>&1 | tee "$LOG_FILE"; then
    echo "  => PASSED [$NUM] $TAG full-size eval"
    VAL_RESULTS+=("PASS  [$NUM] $TAG $LABEL")
    VAL_PASSED=$((VAL_PASSED + 1))
  else
    echo "  => FAILED [$NUM] $TAG full-size eval"
    VAL_RESULTS+=("FAIL  [$NUM] $TAG $LABEL")
    VAL_FAILED=$((VAL_FAILED + 1))
  fi
done

echo ""
echo "============================================================"
echo "  Phase 2b — Full-Size Validation Summary"
echo "============================================================"
for r in "${VAL_RESULTS[@]}"; do echo "  $r"; done
echo "  Passed: $VAL_PASSED  Failed: $VAL_FAILED"

# ------------------------------------------------------------------
# Final 2-way comparison: L1 (= A3 from Phase 1) vs L2 (this phase).
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Phase 2 — Full-Size Comparison (binary aggregate)"
echo "============================================================"
printf "  %-4s  %-22s  %-8s  %-9s  %-7s  %-6s  %-6s  %-6s\n" \
       "TAG" "LOSS" "F1" "PRECISION" "RECALL" "MAE" "RMSE" "AP"

# L1 baseline = A3 metrics CSV.
L1_CSV="$A3_FULLVAL_DIR/metrics_results.csv"
if [ -f "$L1_CSV" ]; then
  awk -F, '$1=="binary" {
    printf "  %-4s  %-22s  %-8.4f  %-9.4f  %-7.4f  %-6.2f  %-6.2f  %-6.4f\n",
           "L1", "herdnet_fmo03 (=A3)", $6, $5, $4, $8, $11, $12 }' "$L1_CSV"
else
  printf "  %-4s  %-22s  (no Phase-1 A3 metrics_results.csv at %s)\n" \
         "L1" "herdnet_fmo03 (=A3)" "$L1_CSV"
fi

for entry in "${VARIANTS[@]}"; do
  IFS=':' read -r NUM TAG LOSS_CFG LABEL <<< "$entry"
  CSV="$RESULTS_DIR/${TAG}/metrics_results.csv"
  if [ -f "$CSV" ]; then
    awk -F, -v tag="$TAG" -v lc="$LOSS_CFG" '$1=="binary" {
      printf "  %-4s  %-22s  %-8.4f  %-9.4f  %-7.4f  %-6.2f  %-6.2f  %-6.4f\n",
             tag, lc, $6, $5, $4, $8, $11, $12 }' "$CSV"
  else
    printf "  %-4s  %-22s  (no metrics_results.csv)\n" "$TAG" "$LOSS_CFG"
  fi
done

echo ""
echo "  Logs:    $LOG_DIR"
echo "  Results: $RESULTS_DIR"
echo "============================================================"

if [ "$VAL_FAILED" -gt 0 ]; then exit 1; fi
