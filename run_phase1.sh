#!/bin/bash
# Phase 1 — augmentation × crop-strategy ablation.
#
# Holds backbone (HerdNetTimmConvNext / convnext_baseline) and loss
# (herdnet_fmo03) fixed; varies only the dataset config:
#
#   A1: fmo03_new_clean             — minimal aug, pre-cropped
#   A2: fmo03_new_clean_augplus     — rich aug (augplus), pre-cropped
#   A3: fmo03_new_objcrop_augplus   — rich aug + dynamic ObjectAwareRandomCrop
#
# A1 vs A2 isolates the augmentation effect (same crop source, +rich aug).
# A2 vs A3 isolates the crop-strategy effect (same augs, dynamic vs pre-baked).
#
# Per the latest dla34 result, per-epoch metrics on the cropped val set don't
# reflect stitched-inference behaviour. So this script runs all 3 trainings
# first (per-epoch metrics on the cheap cropped val set, for convergence
# tracking only), then runs full-size stitched evaluation on each best_model.
#
# Usage: bash run_phase1.sh [START_FROM]
#   START_FROM: optional, skip variants before this number (1=A1, 2=A2, 3=A3)
#
# Outputs:
#   output/phase1_<TAG>_convnext_baseline/<date>/<time>/best_model.pth
#   output/phase1_fullval_<TS>/<TAG>/{metrics_results.csv,confusion_matrix.csv,...}

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="$SCRIPT_DIR/configs/demo"
FULL_VAL_IMAGES="$SCRIPT_DIR/data_fmo03/val/Default"
START_FROM=${1:-1}
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/tmp/phase1_${TS}"
RESULTS_DIR="$SCRIPT_DIR/output/phase1_fullval_${TS}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR"

WANDB_FLAG="${WANDB_FLAG:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-hn_phase1}"

echo "============================================================"
echo "  Phase 1 — augmentation × crop-strategy ablation"
echo "  Backbone:    HerdNetTimmConvNext (convnext_baseline)"
echo "  Loss:        herdnet_fmo03"
echo "  Epochs:      30"
echo "  Wandb:       flag=$WANDB_FLAG project=$WANDB_PROJECT"
echo "  Logs:        $LOG_DIR"
echo "  Final val:   $RESULTS_DIR"
echo "  START_FROM:  $START_FROM"
echo "============================================================"

VARIANTS=(
  "1:A1:fmo03_new_clean:Minimal aug, pre-cropped"
  "2:A2:fmo03_new_clean_augplus:Rich aug (augplus), pre-cropped"
  "3:A3:fmo03_new_objcrop_augplus:Rich aug (augplus) + ObjectAwareRandomCrop"
)

# ------------------------------------------------------------------
# Phase 1a — train all 3 variants sequentially (per-epoch val on the
# cheap cropped val set; full-size eval is deferred to Phase 1b).
# ------------------------------------------------------------------
TRAIN_RESULTS=()
TRAIN_PASSED=0
TRAIN_FAILED=0

for entry in "${VARIANTS[@]}"; do
  IFS=':' read -r NUM TAG DATASET LABEL <<< "$entry"

  if [ "$NUM" -lt "$START_FROM" ]; then
    echo ""
    echo "  SKIP [$NUM/3] $TAG — $LABEL"
    TRAIN_RESULTS+=("SKIP  [$NUM/3] $TAG $LABEL")
    continue
  fi

  RUN_NAME="phase1_${TAG}_${DATASET}"
  OUT_DIR="phase1_${TAG}_convnext_baseline"
  LOG_FILE="$LOG_DIR/${NUM}_${TAG}_train.log"

  echo ""
  echo "------------------------------------------------------------"
  echo "  TRAIN [$NUM/3] $TAG — $LABEL"
  echo "  Dataset:    $DATASET"
  echo "  Wandb run:  $RUN_NAME"
  echo "  Out:        output/$OUT_DIR/<date>/<time>/best_model.pth"
  echo "  Log:        $LOG_FILE"
  echo "  Started:    $(date '+%Y-%m-%d %H:%M:%S')"
  echo "------------------------------------------------------------"

  if python tools/train.py \
      --config-path "$CONFIG_DIR" \
      --config-name fmo03_full_convnext_baseline \
      "datasets=$DATASET" \
      "wandb_flag=$WANDB_FLAG" \
      "wandb_project=$WANDB_PROJECT" \
      "wandb_run=$RUN_NAME" \
      "hydra.run.dir=./output/${OUT_DIR}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}" \
      2>&1 | tee "$LOG_FILE"; then
    echo "  => PASSED [$NUM/3] $TAG $(date '+%H:%M:%S')"
    TRAIN_RESULTS+=("PASS  [$NUM/3] $TAG $LABEL")
    TRAIN_PASSED=$((TRAIN_PASSED + 1))
  else
    echo "  => FAILED [$NUM/3] $TAG $(date '+%H:%M:%S')"
    TRAIN_RESULTS+=("FAIL  [$NUM/3] $TAG $LABEL")
    TRAIN_FAILED=$((TRAIN_FAILED + 1))
  fi
done

echo ""
echo "============================================================"
echo "  Phase 1a — Training Summary"
echo "============================================================"
for r in "${TRAIN_RESULTS[@]}"; do echo "  $r"; done
echo "  Passed: $TRAIN_PASSED  Failed: $TRAIN_FAILED"

if [ "$TRAIN_FAILED" -gt 0 ]; then
  echo "  Some training runs failed — skipping full-size validation."
  exit 1
fi

# ------------------------------------------------------------------
# Phase 1b — full-size stitched validation on each best_model.pth.
# Runs sequentially after all training is done. Uses the same
# `test:` section across all three configs (full-size val/Default).
# ------------------------------------------------------------------
find_best_model() {
  local out_name="$1"
  local base="$SCRIPT_DIR/output/$out_name"
  [ -d "$base" ] || return 1
  find "$base" -name "best_model.pth" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}

VAL_RESULTS=()
VAL_PASSED=0
VAL_FAILED=0

for entry in "${VARIANTS[@]}"; do
  IFS=':' read -r NUM TAG DATASET LABEL <<< "$entry"

  if [ "$NUM" -lt "$START_FROM" ]; then continue; fi

  OUT_DIR="phase1_${TAG}_convnext_baseline"
  MODEL_PATH=$(find_best_model "$OUT_DIR" || true)
  RUN_OUT="$RESULTS_DIR/${TAG}"
  LOG_FILE="$LOG_DIR/${NUM}_${TAG}_fullval.log"

  echo ""
  echo "------------------------------------------------------------"
  echo "  FULLVAL [$NUM/3] $TAG — $LABEL"
  echo "  Model:   ${MODEL_PATH:-<none>}"
  echo "  Out:     $RUN_OUT"
  echo "  Log:     $LOG_FILE"
  echo "------------------------------------------------------------"

  if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "  => MISSING [$NUM/3] $TAG: no best_model.pth under output/$OUT_DIR"
    VAL_RESULTS+=("MISSING [$NUM/3] $TAG $LABEL")
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
        "datasets=$DATASET" \
        "wandb_flag=False") \
      2>&1 | tee "$LOG_FILE"; then
    echo "  => PASSED [$NUM/3] $TAG full-size eval"
    VAL_RESULTS+=("PASS  [$NUM/3] $TAG $LABEL")
    VAL_PASSED=$((VAL_PASSED + 1))
  else
    echo "  => FAILED [$NUM/3] $TAG full-size eval"
    VAL_RESULTS+=("FAIL  [$NUM/3] $TAG $LABEL")
    VAL_FAILED=$((VAL_FAILED + 1))
  fi
done

echo ""
echo "============================================================"
echo "  Phase 1b — Full-Size Validation Summary"
echo "============================================================"
for r in "${VAL_RESULTS[@]}"; do echo "  $r"; done
echo "  Passed: $VAL_PASSED  Failed: $VAL_FAILED"

# ------------------------------------------------------------------
# Final comparison table — pulls metrics from each variant's
# metrics_results.csv (binary row = aggregate across classes).
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Phase 1 — Full-Size Comparison (binary aggregate)"
echo "============================================================"
printf "  %-4s  %-8s  %-9s  %-7s  %-6s  %-6s  %-6s\n" "TAG" "F1" "PRECISION" "RECALL" "MAE" "RMSE" "AP"
for entry in "${VARIANTS[@]}"; do
  IFS=':' read -r NUM TAG DATASET LABEL <<< "$entry"
  CSV="$RESULTS_DIR/${TAG}/metrics_results.csv"
  if [ -f "$CSV" ]; then
    # Pull the 'binary' row (aggregate). Columns:
    # class,species,n,recall,precision,f1_score,confusion,mae,me,mse,rmse,ap
    awk -F, -v tag="$TAG" '
      $1=="binary" {
        printf "  %-4s  %-8.4f  %-9.4f  %-7.4f  %-6.2f  %-6.2f  %-6.4f\n",
               tag, $6, $5, $4, $8, $11, $12
      }' "$CSV"
  else
    printf "  %-4s  (no metrics_results.csv)\n" "$TAG"
  fi
done

echo ""
echo "  Logs:    $LOG_DIR"
echo "  Results: $RESULTS_DIR"
echo "============================================================"

if [ "$VAL_FAILED" -gt 0 ]; then exit 1; fi
