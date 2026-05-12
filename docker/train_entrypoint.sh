#!/bin/bash
# Runs one Phase-13 training inside the container and (by default) uploads
# the resulting best_model.pth as a wandb artifact at the end.
#
# All paths inside the container are fixed by the build:
#   /app/data/train/   — training subset (one N, baked in by build.sh)
#   /app/data/val/     — fixed validation set
#   /app/data/test/    — fixed test set (not used by training; for final infer)
#   /app/best_models/phase8/b4_seed42/best_model.pth   — warm-start
#
# Environment variables (all have sensible defaults from the Dockerfile):
#   TRAIN_N         : the training-size tag, used only for log/artifact
#                     naming (the actual training data is in /app/data/train)
#   SEED            : random seed (default 42)
#   WANDB_PROJECT   : wandb project (default hn_phase13_data_scaling)
#   WANDB_API_KEY   : *required* for wandb integration to work
#   WANDB_FLAG      : True/False (default True)
#   AUG_MULT        : optional override for datasets.train.augmentation_multiplier
#                     (leave empty to use config default of 75)
#   UPLOAD_MODEL    : 1/0 (default 1) — upload best_model.pth to wandb
#   ARTIFACT_NAME   : wandb artifact name (default phase13_best_model)

set -e
export LC_NUMERIC=C

cd /app

echo "============================================================"
echo "  HerdNet Phase-13 training container"
echo "  Started:        $(date '+%Y-%m-%d %H:%M:%S')"
echo "  TRAIN_N:        ${TRAIN_N}"
echo "  SEED:           ${SEED}"
echo "  WANDB_PROJECT:  ${WANDB_PROJECT}"
echo "  WANDB_FLAG:     ${WANDB_FLAG}"
echo "  AUG_MULT:       ${AUG_MULT:-(config default)}"
echo "  UPLOAD_MODEL:   ${UPLOAD_MODEL}"
echo "============================================================"

# wandb sanity: training pushes per-epoch metrics live (same observability
# you'd get locally). The model is the only thing we hold for the end —
# uploaded once via upload_model.py to keep wandb storage minimal.
if [ "${WANDB_FLAG}" = "True" ] && [ -z "${WANDB_API_KEY}" ]; then
  echo ""
  echo "WARNING: WANDB_FLAG=True but WANDB_API_KEY is unset."
  echo "         wandb will run in offline mode; the artifact upload is skipped."
  export WANDB_MODE=offline
fi

# Dataset paths inside the image.
TRAIN_CSV=/app/data/train/herdnet_format.csv
TRAIN_ROOT=/app/data/train/Default
VAL_CROP_CSV=/app/data/val/herdnet_format_512_0_crops.csv
VAL_CROP_ROOT=/app/data/val/crops_512_numNone_overlap0
VAL_FULL_CSV=/app/data/val/herdnet_format.csv
VAL_FULL_ROOT=/app/data/val/Default
WARM_START=/app/best_models/phase8/b4_seed42/best_model.pth

# Sanity checks — fail loudly before consuming GPU time.
for p in "$TRAIN_CSV" "$TRAIN_ROOT" "$VAL_CROP_CSV" "$VAL_CROP_ROOT" \
         "$VAL_FULL_CSV" "$VAL_FULL_ROOT" "$WARM_START"; do
  if [ ! -e "$p" ]; then
    echo "ERROR: missing $p — was the image built correctly?"
    exit 1
  fi
done
echo "  all paths present."
echo ""

# Output dir for this run (also where best_model.pth lands).
RUN_NAME="phase13_N${TRAIN_N}_s${SEED}_docker"
OUT_DIR="/app/output/${RUN_NAME}/$(date '+%Y-%m-%d')/$(date '+%H-%M-%S')"
mkdir -p "$OUT_DIR"
TRAIN_LOG="${OUT_DIR}/training.log"

# Optional AUG_MULT override.
TRAIN_EXTRA=()
if [ -n "${AUG_MULT}" ]; then
  TRAIN_EXTRA+=("datasets.train.augmentation_multiplier=${AUG_MULT}")
fi

# ---- Train ----
# Training pushes per-epoch metrics to wandb live (same observability you
# get locally — losses, validation F1 / MAE, learning rate, etc.). The
# only thing we hold for the end is the model checkpoint, uploaded as
# an Artifact via upload_model.py so wandb storage stays small.
echo "============================================================"
echo "  Training (output → ${OUT_DIR})"
echo "  wandb_flag during training: ${WANDB_FLAG} (per-epoch curves)"
echo "============================================================"
python tools/train.py \
  --config-path /app/configs/demo \
  --config-name data_scaling_b4 \
  "seed=${SEED}" \
  "datasets.train.csv_file=${TRAIN_CSV}" \
  "datasets.train.root_dir=${TRAIN_ROOT}" \
  "datasets.validate.csv_file=${VAL_CROP_CSV}" \
  "datasets.validate.root_dir=${VAL_CROP_ROOT}" \
  "datasets.test.csv_file=${VAL_FULL_CSV}" \
  "datasets.test.root_dir=${VAL_FULL_ROOT}" \
  "model.load_from=${WARM_START}" \
  "wandb_flag=${WANDB_FLAG}" \
  "wandb_project=${WANDB_PROJECT}" \
  "wandb_run=${RUN_NAME}" \
  "+wandb_tags=[phase13,data_scaling,docker,N${TRAIN_N}]" \
  "${TRAIN_EXTRA[@]}" \
  "hydra.run.dir=${OUT_DIR}" \
  2>&1 | tee "$TRAIN_LOG"
TRAIN_EXIT=${PIPESTATUS[0]}

echo ""
if [ "$TRAIN_EXIT" -ne 0 ]; then
  echo "Training failed with exit $TRAIN_EXIT — skipping artifact upload."
  exit $TRAIN_EXIT
fi
echo "Training finished at $(date '+%Y-%m-%d %H:%M:%S')."

# Find the actual best_model.pth that was just written.
BEST_MODEL=$(find "$OUT_DIR" -name 'best_model.pth' | head -1)
if [ ! -f "$BEST_MODEL" ]; then
  echo "WARNING: best_model.pth not found under $OUT_DIR"
  exit 0
fi
echo "best_model.pth: $BEST_MODEL ($(du -h "$BEST_MODEL" | cut -f1))"

# ---- Upload as wandb artifact ----
if [ "${UPLOAD_MODEL}" = "1" ] && [ "${WANDB_FLAG}" = "True" ]; then
  echo ""
  echo "============================================================"
  echo "  Uploading model to wandb"
  echo "============================================================"
  WANDB_PROJECT="${WANDB_PROJECT}" \
  WANDB_API_KEY="${WANDB_API_KEY:-}" \
  ARTIFACT_NAME="${ARTIFACT_NAME}" \
  RUN_NAME="${RUN_NAME}" \
  MODEL_PATH="$BEST_MODEL" \
  TRAIN_LOG="$TRAIN_LOG" \
  TRAIN_N="${TRAIN_N}" \
  SEED="${SEED}" \
  OUT_DIR="$OUT_DIR" \
  python /app/docker/upload_model.py
fi

echo ""
echo "Done at $(date '+%Y-%m-%d %H:%M:%S')."
