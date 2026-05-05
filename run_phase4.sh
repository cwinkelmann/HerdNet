#!/bin/bash
# Phase 4 — re-train the Phase-3 winner (B4) with its NATIVE loss.
#
# Phase 3 ran B4 (fmo03_full_v2_bifpn) with herdnet_fmo03 (Focal+CE) for
# clean cross-backbone comparison. That handicapped B4: its V2 backbone
# emits aux_p3 / aux_p4 heads that received no gradient signal because
# herdnet_fmo03 doesn't reference them. Phase 4 removes that handicap and
# trains the SAME backbone with its native density_aware_fmo03_aux loss
# (DensityAwareHerdNetLoss + weighted CE + auxiliary FocalLoss on P3 (λ=0.3)
# and P4 (λ=0.1)).
#
# Comparing Phase 4 to Phase 3 B4 isolates the effect of the deep-supervision
# aux heads, on a fixed architecture, dataset, and 30-epoch budget.
#
# Reference (Phase 3 B4 with herdnet_fmo03 override):
#   default ts=0.30: F1=0.934, P=0.929, R=0.939, MAE=0.83
#   tuned   ts=0.50: F1=0.955, P=0.977, R=0.934, MAE=0.83
#                   (best MAE 0.75 at ts=0.35 with F1=0.942)
#
# After training, Phase 4 runs the same threshold sweep so we can compare
# best-F1 / best-MAE / recall@P>=.85 directly to the Phase 3 B4 row.
#
# Usage: bash run_phase4.sh

set -e
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="$SCRIPT_DIR/configs/demo"
FULL_VAL_IMAGES="$SCRIPT_DIR/data_fmo03/val/Default"
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/tmp/phase4_${TS}"
RESULTS_DIR="$SCRIPT_DIR/output/phase4_fullval_${TS}"
SWEEP_DIR="$SCRIPT_DIR/output/phase4_sweep_${TS}"
mkdir -p "$LOG_DIR" "$RESULTS_DIR" "$SWEEP_DIR"

WANDB_FLAG="${WANDB_FLAG:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-hn_phase4}"

CONFIG="fmo03_full_v2_bifpn"
TAG="B4_native"
RUN_NAME="phase4_${TAG}"
OUT_DIR="phase4_${TAG}_v2_bifpn"

echo "============================================================"
echo "  Phase 4 — re-train B4 with native density_aware_fmo03_aux loss"
echo "  Backbone:    HerdNetTimmConvNext_Camouflaged_V2 (BiFPN+DeformConv)"
echo "  Dataset:     fmo03_new_objcrop_augplus (A3 winner)"
echo "  Loss:        density_aware_fmo03_aux (NATIVE — aux_p3/p4 active)"
echo "  Epochs:      30"
echo "  Wandb:       flag=$WANDB_FLAG project=$WANDB_PROJECT run=$RUN_NAME"
echo "  Logs:        $LOG_DIR"
echo "  Final val:   $RESULTS_DIR"
echo "  Sweep:       $SWEEP_DIR"
echo "============================================================"

SWEEP_TS=(0.05 0.10 0.15 0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70)

# ------------------------------------------------------------------
# Phase 4a — train B4 with its native loss (no override).
# ------------------------------------------------------------------
TRAIN_LOG="$LOG_DIR/${TAG}_train.log"
echo ""
echo "------------------------------------------------------------"
echo "  TRAIN  $TAG"
echo "  Out:    output/$OUT_DIR/<date>/<time>/best_model.pth"
echo "  Log:    $TRAIN_LOG"
echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "------------------------------------------------------------"

if ! python tools/train.py \
    --config-path "$CONFIG_DIR" \
    --config-name "$CONFIG" \
    "wandb_flag=$WANDB_FLAG" \
    "wandb_project=$WANDB_PROJECT" \
    "wandb_run=$RUN_NAME" \
    "hydra.run.dir=./output/${OUT_DIR}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}" \
    2>&1 | tee "$TRAIN_LOG"; then
  echo "  TRAIN FAILED at $(date '+%H:%M:%S')"
  exit 1
fi

echo "  TRAIN done at $(date '+%H:%M:%S')"

# ------------------------------------------------------------------
# Phase 4b — full-size eval at default ts=0.30 (apples-to-apples with Phase 3b).
# ------------------------------------------------------------------
find_best_model() {
  local base="$1"
  [ -d "$base" ] || return 1
  find "$base" -name "best_model.pth" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}

MODEL_PATH=$(find_best_model "$SCRIPT_DIR/output/$OUT_DIR" || true)
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH" ]; then
  echo "  best_model.pth missing under output/$OUT_DIR — aborting"
  exit 1
fi

DEFAULT_OUT="$RESULTS_DIR/${TAG}"
DEFAULT_LOG="$LOG_DIR/${TAG}_fullval.log"
mkdir -p "$DEFAULT_OUT"
echo ""
echo "------------------------------------------------------------"
echo "  FULLVAL  $TAG (default ts=0.30)"
echo "  Model:   $MODEL_PATH"
echo "  Log:     $DEFAULT_LOG"
echo "------------------------------------------------------------"
(cd "$DEFAULT_OUT" && python "$SCRIPT_DIR/tools/infer.py" \
    --config-dir "$CONFIG_DIR" \
    --config-name "$CONFIG" \
    --images "$FULL_VAL_IMAGES" \
    --model "$MODEL_PATH" \
    --evaluate \
    --overrides "wandb_flag=False") \
    2>&1 | tee "$DEFAULT_LOG" || echo "default-threshold eval failed (continuing to sweep)"

# ------------------------------------------------------------------
# Phase 4c — threshold sweep, same grid as Phase 3c.
# ------------------------------------------------------------------
SWEEP_SUMMARY="$SWEEP_DIR/sweep.csv"
echo "tag,backbone,adapt_ts,f1,precision,recall,mae,rmse,ap" > "$SWEEP_SUMMARY"

for ATS in "${SWEEP_TS[@]}"; do
  SUB="$SWEEP_DIR/ts_${ATS}"
  LOG="$LOG_DIR/${TAG}_sweep_${ATS}.log"
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
        "wandb_flag=False" \
        "training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=$ATS") \
      > "$LOG" 2>&1 || echo "    sweep failed at $ATS (continuing)"
  CSV="$SUB/metrics_results.csv"
  if [ -f "$CSV" ]; then
    awk -F, -v ats="$ATS" '$1=="binary" {
      printf "B4_native,v2_bifpn_native_aux,%s,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f\n",
             ats, $6, $5, $4, $8, $11, $12 }' "$CSV" >> "$SWEEP_SUMMARY"
  fi
done

echo ""
echo "============================================================"
echo "  Phase 4 — Sweep Summary (B4 with native aux loss)"
echo "============================================================"
printf "  %-8s  %-8s  %-9s  %-7s  %-6s  %-6s  %-6s\n" \
       "ats" "F1" "PRECISION" "RECALL" "MAE" "RMSE" "AP"
awk -F, 'NR>1 { printf "  %-8s  %-8.4f  %-9.4f  %-7.4f  %-6.2f  %-6.2f  %-6.4f\n",
                       $3, $4, $5, $6, $7, $8, $9 }' "$SWEEP_SUMMARY"

echo ""
echo "  Best operating points:"
python3 - <<PY
import csv
rows = list(csv.DictReader(open("$SWEEP_SUMMARY")))
def f(r,k): return float(r[k])
bf1 = max(rows, key=lambda r: f(r,'f1'))
bma = min(rows, key=lambda r: f(r,'mae'))
hp = [r for r in rows if f(r,'precision') >= 0.85]
brp = max(hp, key=lambda r: f(r,'recall')) if hp else None
print(f"    best F1   : F1={f(bf1,'f1'):.4f}  P={f(bf1,'precision'):.4f}  R={f(bf1,'recall'):.4f}  MAE={f(bf1,'mae'):.2f}  @ ts={bf1['adapt_ts']}")
print(f"    best MAE  : F1={f(bma,'f1'):.4f}  P={f(bma,'precision'):.4f}  R={f(bma,'recall'):.4f}  MAE={f(bma,'mae'):.2f}  @ ts={bma['adapt_ts']}")
if brp:
    print(f"    best R@P>=.85 : F1={f(brp,'f1'):.4f}  P={f(brp,'precision'):.4f}  R={f(brp,'recall'):.4f}  MAE={f(brp,'mae'):.2f}  @ ts={brp['adapt_ts']}")
PY

echo ""
echo "  Logs:    $LOG_DIR"
echo "  Results: $RESULTS_DIR"
echo "  Sweep:   $SWEEP_DIR"
echo "  Summary CSV: $SWEEP_SUMMARY"
echo "============================================================"
