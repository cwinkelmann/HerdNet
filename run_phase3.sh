#!/bin/bash
# Phase 3 — backbone comparison + threshold tuning.
#
# Holds dataset (fmo03_new_objcrop_augplus, A3 winner of Phase 1) and loss
# (herdnet_fmo03, L1 winner of Phase 2) fixed; varies the backbone.
#
# Phase 3a: train 4 new backbones (B2/convnext_baseline reused from Phase 1 A3).
# Phase 3b: full-size stitched eval each, at the default LMDS threshold (0.3).
# Phase 3c: sweep adapt_ts to find the threshold that minimises the
#           full-size counting error (MAE) and the threshold that maximises
#           full-size F1, per backbone.
#
# Backbones tested:
#   B1 dla34_timm           — DLA-34 via timm                  (NEW)
#   B2 convnext_baseline    — HerdNetTimmConvNext              (REUSED, =A3)
#   B3 convnextv2_gabor     — ConvNeXtV2 + Gabor projection    (NEW)
#   B4 v2_bifpn             — ConvNeXt + BiFPN + DeformConv    (NEW; aux heads
#                              run but get no gradient since loss is herdnet_fmo03)
#   B5 efficientvit_gabor   — EfficientViT + Gabor             (NEW)
#
# Note: B3/B4/B5 native loss is density_aware* but we override to herdnet_fmo03
# for a clean backbone ablation — Phase 2 showed herdnet_fmo03 wins at the
# operating point. v2_bifpn's aux heads are present in the model but ignored
# by the loss; inference doesn't use them either, so this is fair.
#
# Usage: bash run_phase3.sh [START_FROM]
#   START_FROM: 1=B1 (dla34_timm), 2=skip-B1, 3=B3 (cnext-v2 gabor), 4=B4 (v2_bifpn), 5=B5

set -e
export LC_NUMERIC=C   # printf %f uses '.', not ','

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="$SCRIPT_DIR/configs/demo"
FULL_VAL_IMAGES="$SCRIPT_DIR/data_fmo03/val/Default"
START_FROM=${1:-1}
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/tmp/phase3_${TS}"
RESULTS_DIR="$SCRIPT_DIR/output/phase3_fullval_${TS}"
SWEEP_DIR="$SCRIPT_DIR/output/phase3_sweep_${TS}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$SWEEP_DIR"

WANDB_FLAG="${WANDB_FLAG:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-hn_phase3}"

# Phase-1 A3 = convnext_baseline + herdnet_fmo03 — already trained.
B2_OUTPUT_DIR="$SCRIPT_DIR/output/phase1_A3_convnext_baseline"

echo "============================================================"
echo "  Phase 3 — backbone comparison + threshold tuning"
echo "  Dataset:     fmo03_new_objcrop_augplus (A3)"
echo "  Loss:        herdnet_fmo03 (L1, Phase 2 winner)"
echo "  Epochs:      30"
echo "  Wandb:       flag=$WANDB_FLAG project=$WANDB_PROJECT"
echo "  Logs:        $LOG_DIR"
echo "  Final val:   $RESULTS_DIR"
echo "  Sweep:       $SWEEP_DIR"
echo "  B2 reused:   $B2_OUTPUT_DIR"
echo "  START_FROM:  $START_FROM"
echo "============================================================"

# Variants: NUM:TAG:CONFIG:LOSS_OVERRIDE_NEEDED
# B2 is excluded from the train loop (reused from A3) but included in eval.
TRAIN_VARIANTS=(
  "1:B1:fmo03_full_dla34_timm:0:DLA-34 timm"
  "3:B3:fmo03_full_convnextv2_gabor:1:ConvNeXt-V2 + Gabor"
  "4:B4:fmo03_full_v2_bifpn:1:ConvNeXt + BiFPN + DeformConv"
  "5:B5:fmo03_full_efficientvit_gabor:1:EfficientViT + Gabor"
)

# All backbones for evaluation (B2 = reused from Phase 1 A3).
EVAL_VARIANTS=(
  "1:B1:fmo03_full_dla34_timm:DLA-34 timm"
  "2:B2:fmo03_full_convnext_baseline:HerdNetTimmConvNext (=A3 reused)"
  "3:B3:fmo03_full_convnextv2_gabor:ConvNeXt-V2 + Gabor"
  "4:B4:fmo03_full_v2_bifpn:ConvNeXt + BiFPN + DeformConv"
  "5:B5:fmo03_full_efficientvit_gabor:EfficientViT + Gabor"
)

# Threshold values to sweep (LMDS adapt_ts). 11 points covering the
# practical range; 0.3 is the Phase-1/2 default.
SWEEP_TS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70)

# ------------------------------------------------------------------
# Phase 3a — train 4 new backbones (B2 is reused from Phase 1).
# ------------------------------------------------------------------
TRAIN_RESULTS=()
TRAIN_PASSED=0
TRAIN_FAILED=0

for entry in "${TRAIN_VARIANTS[@]}"; do
  IFS=':' read -r NUM TAG CONFIG LOSS_OVERRIDE LABEL <<< "$entry"

  if [ "$NUM" -lt "$START_FROM" ]; then
    echo ""
    echo "  SKIP [$NUM] $TAG — $LABEL"
    TRAIN_RESULTS+=("SKIP  [$NUM] $TAG $LABEL")
    continue
  fi

  RUN_NAME="phase3_${TAG}_${CONFIG}"
  OUT_DIR="phase3_${TAG}_${CONFIG}"
  LOG_FILE="$LOG_DIR/${NUM}_${TAG}_train.log"

  EXTRA_OVERRIDES=()
  if [ "$LOSS_OVERRIDE" = "1" ]; then
    EXTRA_OVERRIDES+=("losses=herdnet_fmo03")
  fi

  echo ""
  echo "------------------------------------------------------------"
  echo "  TRAIN [$NUM] $TAG — $LABEL"
  echo "  Config:     $CONFIG"
  echo "  Loss:       herdnet_fmo03 ${EXTRA_OVERRIDES[*]:+(override)}"
  echo "  Wandb run:  $RUN_NAME"
  echo "  Out:        output/$OUT_DIR/<date>/<time>/best_model.pth"
  echo "  Log:        $LOG_FILE"
  echo "  Started:    $(date '+%Y-%m-%d %H:%M:%S')"
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
echo "  Phase 3a — Training Summary"
echo "============================================================"
for r in "${TRAIN_RESULTS[@]}"; do echo "  $r"; done
echo "  Passed: $TRAIN_PASSED  Failed: $TRAIN_FAILED"

if [ "$TRAIN_FAILED" -gt 0 ]; then
  echo "  Some training runs failed — skipping Phase 3b/3c."
  exit 1
fi

# ------------------------------------------------------------------
# Phase 3b — full-size stitched eval at default LMDS threshold (0.3).
# ------------------------------------------------------------------
find_best_model() {
  local base="$1"
  [ -d "$base" ] || return 1
  find "$base" -name "best_model.pth" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}

VAL_RESULTS=()

for entry in "${EVAL_VARIANTS[@]}"; do
  IFS=':' read -r NUM TAG CONFIG LABEL <<< "$entry"

  # Locate best_model: B2 is reused from Phase 1 A3.
  if [ "$TAG" = "B2" ]; then
    OUT_DIR_PATH="$B2_OUTPUT_DIR"
  else
    OUT_DIR_PATH="$SCRIPT_DIR/output/phase3_${TAG}_${CONFIG}"
  fi

  MODEL_PATH=$(find_best_model "$OUT_DIR_PATH" || true)
  RUN_OUT="$RESULTS_DIR/${TAG}"
  LOG_FILE="$LOG_DIR/${NUM}_${TAG}_fullval.log"

  echo ""
  echo "------------------------------------------------------------"
  echo "  FULLVAL [$NUM] $TAG — $LABEL  (default adapt_ts=0.3)"
  echo "  Model:   ${MODEL_PATH:-<none>}"
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
      --config-name "$CONFIG" \
      --images "$FULL_VAL_IMAGES" \
      --model "$MODEL_PATH" \
      --evaluate \
      --overrides \
        "losses=herdnet_fmo03" \
        "wandb_flag=False") \
      2>&1 | tee "$LOG_FILE"; then
    VAL_RESULTS+=("PASS  [$NUM] $TAG $LABEL")
  else
    VAL_RESULTS+=("FAIL  [$NUM] $TAG $LABEL")
  fi
done

echo ""
echo "============================================================"
echo "  Phase 3b — Full-Size Comparison (default threshold)"
echo "============================================================"
printf "  %-4s  %-38s  %-8s  %-9s  %-7s  %-6s  %-6s  %-6s\n" \
       "TAG" "BACKBONE" "F1" "PRECISION" "RECALL" "MAE" "RMSE" "AP"
for entry in "${EVAL_VARIANTS[@]}"; do
  IFS=':' read -r NUM TAG CONFIG LABEL <<< "$entry"
  CSV="$RESULTS_DIR/${TAG}/metrics_results.csv"
  if [ -f "$CSV" ]; then
    awk -F, -v tag="$TAG" -v lbl="$LABEL" '$1=="binary" {
      printf "  %-4s  %-38s  %-8.4f  %-9.4f  %-7.4f  %-6.2f  %-6.2f  %-6.4f\n",
             tag, lbl, $6, $5, $4, $8, $11, $12 }' "$CSV"
  else
    printf "  %-4s  %-38s  (no metrics_results.csv)\n" "$TAG" "$LABEL"
  fi
done

# ------------------------------------------------------------------
# Phase 3c — threshold sweep per backbone on full-size val.
# For each backbone, run tools/infer.py at each adapt_ts and collect
# the binary-row metrics. Output one CSV per backbone plus a summary.
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Phase 3c — Threshold sweep on full-size val"
echo "  adapt_ts ∈ { ${SWEEP_TS[*]} }"
echo "============================================================"

SWEEP_SUMMARY="$SWEEP_DIR/summary.csv"
echo "tag,backbone,adapt_ts,f1,precision,recall,mae,rmse,ap" > "$SWEEP_SUMMARY"

for entry in "${EVAL_VARIANTS[@]}"; do
  IFS=':' read -r NUM TAG CONFIG LABEL <<< "$entry"

  if [ "$TAG" = "B2" ]; then
    OUT_DIR_PATH="$B2_OUTPUT_DIR"
  else
    OUT_DIR_PATH="$SCRIPT_DIR/output/phase3_${TAG}_${CONFIG}"
  fi
  MODEL_PATH=$(find_best_model "$OUT_DIR_PATH" || true)

  if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
    echo "  $TAG: missing best_model — skipping sweep"
    continue
  fi

  PER_TAG_CSV="$SWEEP_DIR/${TAG}_sweep.csv"
  echo "tag,backbone,adapt_ts,f1,precision,recall,mae,rmse,ap" > "$PER_TAG_CSV"

  for ATS in "${SWEEP_TS[@]}"; do
    SUB="$SWEEP_DIR/${TAG}/ts_${ATS}"
    LOG_FILE="$LOG_DIR/${NUM}_${TAG}_sweep_${ATS}.log"
    mkdir -p "$SUB"

    echo ""
    echo "  $TAG @ adapt_ts=$ATS ..."
    (cd "$SUB" && python "$SCRIPT_DIR/tools/infer.py" \
        --config-dir "$CONFIG_DIR" \
        --config-name "$CONFIG" \
        --images "$FULL_VAL_IMAGES" \
        --model "$MODEL_PATH" \
        --evaluate \
        --overrides \
          "losses=herdnet_fmo03" \
          "wandb_flag=False" \
          "training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=$ATS") \
        > "$LOG_FILE" 2>&1 || echo "    sweep failed at $ATS (continuing)"

    CSV="$SUB/metrics_results.csv"
    if [ -f "$CSV" ]; then
      awk -F, -v tag="$TAG" -v lbl="$LABEL" -v ats="$ATS" '$1=="binary" {
        printf "%s,%s,%s,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f\n",
               tag, lbl, ats, $6, $5, $4, $8, $11, $12 }' "$CSV" \
        | tee -a "$PER_TAG_CSV" "$SWEEP_SUMMARY" >/dev/null
    fi
  done

  echo ""
  echo "  --- $TAG sweep summary ---"
  printf "  %-8s  %-8s  %-9s  %-7s  %-6s  %-6s  %-6s\n" \
         "ats" "F1" "PRECISION" "RECALL" "MAE" "RMSE" "AP"
  awk -F, 'NR>1 { printf "  %-8s  %-8.4f  %-9.4f  %-7.4f  %-6.2f  %-6.2f  %-6.4f\n",
                         $3, $4, $5, $6, $7, $8, $9 }' "$PER_TAG_CSV"
done

# ------------------------------------------------------------------
# Best-threshold report — best F1, best MAE, recall@P>=0.85 for each backbone.
# ------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Phase 3 — Best-Threshold Summary"
echo "============================================================"
printf "  %-4s  %-38s  %-15s  %-15s  %-15s\n" \
       "TAG" "BACKBONE" "BEST_F1@ts" "BEST_MAE@ts" "BEST_REC_at_P>=.85"
for entry in "${EVAL_VARIANTS[@]}"; do
  IFS=':' read -r NUM TAG CONFIG LABEL <<< "$entry"
  PER_TAG_CSV="$SWEEP_DIR/${TAG}_sweep.csv"
  [ -f "$PER_TAG_CSV" ] || continue

  python3 - <<PY
import csv, sys
rows = list(csv.DictReader(open("$PER_TAG_CSV")))
if not rows:
    print(f"  $TAG  ($LABEL)  no sweep rows")
    sys.exit(0)
def f(r,k): return float(r[k])
best_f1  = max(rows, key=lambda r: f(r,'f1'))
best_mae = min(rows, key=lambda r: f(r,'mae'))
hi_p = [r for r in rows if f(r,'precision') >= 0.85]
best_rec = max(hi_p, key=lambda r: f(r,'recall')) if hi_p else None
def fmt(r): return f"F1={f(r,'f1'):.3f}@{r['adapt_ts']}" if r else "-"
def fmt_mae(r): return f"MAE={f(r,'mae'):.2f}@{r['adapt_ts']}" if r else "-"
def fmt_rec(r): return f"R={f(r,'recall'):.3f}@{r['adapt_ts']}" if r else "(none P>=.85)"
print(f"  {'$TAG':<4}  {'$LABEL':<38}  {fmt(best_f1):<15}  {fmt_mae(best_mae):<15}  {fmt_rec(best_rec):<15}")
PY
done

echo ""
echo "  Logs:    $LOG_DIR"
echo "  Results: $RESULTS_DIR"
echo "  Sweep:   $SWEEP_DIR"
echo "  Summary CSV: $SWEEP_SUMMARY"
echo "============================================================"
