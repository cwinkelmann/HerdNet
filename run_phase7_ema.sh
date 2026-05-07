#!/bin/bash
# Phase 7 — train B4 with weight EMA at 3 seeds, then re-form the
# ensemble and re-run the Phase-6-style sweeps to see if EMA-trained
# checkpoints push past the Phase-6 production picks:
#
#   Phase-6 best F1   : 0.9634 (3-seed ensemble + TTA, ts=0.35)
#   Phase-6 best MAE  : 0.67   (3-seed ensemble no-TTA,  ts=0.30)
#
# Architecture and dataset are fixed: fmo03_full_v2_bifpn (B4, BiFPN +
# DeformConv) on fmo03_new_objcrop_augplus, herdnet_fmo03 loss override
# (Phase 5 showed this loss combo wins the operating point across seeds).
# Only difference vs the Phase-3/5 B4 trainings: +training_settings.ema_decay=0.9999.
#
# After all 3 trainings finish: full-size eval at ts=0.30 and the
# Phase-6 sweep grid for (a) best single seed (s42), (b) 3-seed ensemble
# no-TTA, (c) 3-seed ensemble + TTA. Compare to the matching Phase-6
# numbers to attribute any lift cleanly to EMA.
#
# Usage: bash run_phase7_ema.sh [START_FROM]
#   START_FROM: 1=s42, 2=s123, 3=s7 (default 1)

set -e
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="$SCRIPT_DIR/configs/demo"
CONFIG="fmo03_full_v2_bifpn"
FULL_VAL_IMAGES="$SCRIPT_DIR/data_fmo03/val/Default"
EMA_DECAY="${EMA_DECAY:-0.9999}"

START_FROM=${1:-1}
TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/tmp/phase7_${TS}"
SWEEP_DIR="$SCRIPT_DIR/output/phase7_sweep_${TS}"
mkdir -p "$LOG_DIR" "$SWEEP_DIR"

WANDB_FLAG="${WANDB_FLAG:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-hn_phase7_ema}"

echo "============================================================"
echo "  Phase 7 — B4 with EMA (decay_max=$EMA_DECAY) at 3 seeds"
echo "  Config:        $CONFIG"
echo "  Loss:          herdnet_fmo03 (override, per Phase 5 winner)"
echo "  Dataset:       fmo03_new_objcrop_augplus"
echo "  Wandb project: $WANDB_PROJECT"
echo "  Logs:          $LOG_DIR"
echo "  Sweep:         $SWEEP_DIR"
echo "  START_FROM:    $START_FROM"
echo "============================================================"

# --------------------------------------------------------------------
# Phase 7a — train B4_ema at 3 seeds.
# --------------------------------------------------------------------
SEEDS=(42 123 7)
TRAIN_RESULTS=()
TRAIN_FAILED=0

for i in 0 1 2; do
  num=$((i + 1))
  seed=${SEEDS[$i]}

  if [ "$num" -lt "$START_FROM" ]; then
    echo "  SKIP [$num/3] seed=$seed"
    TRAIN_RESULTS+=("SKIP  seed=$seed")
    continue
  fi

  RUN_NAME="phase7_B4_ema_s${seed}"
  OUT_DIR="phase7_B4_ema_s${seed}"
  LOG_FILE="$LOG_DIR/${num}_seed${seed}_train.log"

  echo ""
  echo "------------------------------------------------------------"
  echo "  TRAIN [$num/3] B4_ema seed=$seed"
  echo "  Out:    output/$OUT_DIR/<date>/<time>/best_model.pth"
  echo "  Log:    $LOG_FILE"
  echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "------------------------------------------------------------"

  if python tools/train.py \
      --config-path "$CONFIG_DIR" \
      --config-name "$CONFIG" \
      "losses=herdnet_fmo03" \
      "seed=$seed" \
      "+training_settings.ema_decay=$EMA_DECAY" \
      "wandb_flag=$WANDB_FLAG" \
      "wandb_project=$WANDB_PROJECT" \
      "wandb_run=$RUN_NAME" \
      "hydra.run.dir=./output/${OUT_DIR}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}" \
      2>&1 | tee "$LOG_FILE"; then
    TRAIN_RESULTS+=("PASS  seed=$seed")
  else
    TRAIN_RESULTS+=("FAIL  seed=$seed")
    TRAIN_FAILED=$((TRAIN_FAILED + 1))
  fi
done

echo ""
echo "============================================================"
echo "  Phase 7a — Training Summary"
echo "============================================================"
for r in "${TRAIN_RESULTS[@]}"; do echo "  $r"; done
if [ "$TRAIN_FAILED" -gt 0 ]; then
  echo "  Some training runs failed — skipping eval."
  exit 1
fi

# --------------------------------------------------------------------
# Resolve the 3 EMA checkpoints.
# --------------------------------------------------------------------
find_best_model() {
  local base="$1"
  [ -d "$base" ] || return 1
  find "$base" -name "best_model.pth" -printf '%T@ %p\n' 2>/dev/null \
    | sort -rn | head -1 | cut -d' ' -f2-
}

M_S42=$(find_best_model "$SCRIPT_DIR/output/phase7_B4_ema_s42")
M_S123=$(find_best_model "$SCRIPT_DIR/output/phase7_B4_ema_s123")
M_S7=$(find_best_model "$SCRIPT_DIR/output/phase7_B4_ema_s7")

for p in "$M_S42" "$M_S123" "$M_S7"; do
  if [ ! -f "$p" ]; then
    echo "  ERROR: missing checkpoint — got '$p'"; exit 1
  fi
done

echo ""
echo "  EMA checkpoints:"
echo "    s42:  $M_S42"
echo "    s123: $M_S123"
echo "    s7:   $M_S7"

# --------------------------------------------------------------------
# Phase 7b — sweep grid identical to Phase 6, three regimes.
#   T1: per-seed EMA (sweep all thresholds for each seed)
#   T2: 3-seed EMA ensemble (no TTA) — full sweep
#   T3: 3-seed EMA ensemble + TTA   — full sweep
# --------------------------------------------------------------------
SWEEP_TS=(0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70)

run_single() {
  local TAG=$1; local MODEL=$2; local ATS=$3; local TTA=$4; local CSV_OUT=$5
  local SUB="$SWEEP_DIR/${TAG}/ts_${ATS}_tta${TTA}"
  local LOG="$LOG_DIR/${TAG}_ts${ATS}_tta${TTA}.log"
  mkdir -p "$SUB"
  echo "  [$(date '+%H:%M:%S')] $TAG @ ts=$ATS tta=$TTA"
  (cd "$SUB" && python "$SCRIPT_DIR/tools/infer.py" \
      --config-dir "$CONFIG_DIR" \
      --config-name "$CONFIG" \
      --images "$FULL_VAL_IMAGES" \
      --model "$MODEL" \
      --evaluate \
      --overrides \
        "losses=herdnet_fmo03" \
        "wandb_flag=False" \
        "+training_settings.stitcher.kwargs.tta=$TTA" \
        "training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=$ATS") \
      > "$LOG" 2>&1 || echo "    failed at $ATS"
  local CSV="$SUB/metrics_results.csv"
  if [ -f "$CSV" ]; then
    awk -F, -v tag="$TAG" -v ats="$ATS" -v tta="$TTA" '$1=="binary" {
      printf "%s,%s,%s,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f\n",
             tag, ats, tta, $6, $5, $4, $8, $11, $12 }' "$CSV" >> "$CSV_OUT"
  fi
}

run_ensemble() {
  local TAG=$1; local ATS=$2; local TTA=$3; local CSV_OUT=$4
  local SUB="$SWEEP_DIR/${TAG}/ts_${ATS}_tta${TTA}"
  local LOG="$LOG_DIR/${TAG}_ts${ATS}_tta${TTA}.log"
  mkdir -p "$SUB"
  echo "  [$(date '+%H:%M:%S')] $TAG @ ts=$ATS tta=$TTA"
  (cd "$SUB" && python "$SCRIPT_DIR/tools/ensemble_infer.py" \
      --config-dir "$CONFIG_DIR" \
      --config-name "$CONFIG" \
      --images "$FULL_VAL_IMAGES" \
      --models "$M_S42" "$M_S123" "$M_S7" \
      --evaluate \
      --overrides \
        "losses=herdnet_fmo03" \
        "wandb_flag=False" \
        "+training_settings.stitcher.kwargs.tta=$TTA" \
        "training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=$ATS") \
      > "$LOG" 2>&1 || echo "    failed at $ATS"
  local CSV="$SUB/metrics_results.csv"
  if [ -f "$CSV" ]; then
    awk -F, -v tag="$TAG" -v ats="$ATS" -v tta="$TTA" '$1=="binary" {
      printf "%s,%s,%s,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f\n",
             tag, ats, tta, $6, $5, $4, $8, $11, $12 }' "$CSV" >> "$CSV_OUT"
  fi
}

# T1 — per-seed EMA, full sweep.
T1_CSV="$SWEEP_DIR/T1_ema_per_seed.csv"
echo "tag,adapt_ts,tta,f1,precision,recall,mae,rmse,ap" > "$T1_CSV"
echo ""
echo "------------------------------------------------------------"
echo "  T1 — per-seed EMA sweep (no TTA)"
echo "------------------------------------------------------------"
for SEED_TAG in "B4_ema_s42:$M_S42" "B4_ema_s123:$M_S123" "B4_ema_s7:$M_S7"; do
  TAG=${SEED_TAG%%:*}; MODEL=${SEED_TAG##*:}
  for ATS in "${SWEEP_TS[@]}"; do
    run_single "$TAG" "$MODEL" "$ATS" "False" "$T1_CSV"
  done
done

# T2 — 3-seed EMA ensemble (no TTA), full sweep.
T2_CSV="$SWEEP_DIR/T2_ema_ensemble_no_tta.csv"
echo "tag,adapt_ts,tta,f1,precision,recall,mae,rmse,ap" > "$T2_CSV"
echo ""
echo "------------------------------------------------------------"
echo "  T2 — 3-seed EMA ensemble (no TTA)"
echo "------------------------------------------------------------"
for ATS in "${SWEEP_TS[@]}"; do
  run_ensemble "ENS3_EMA" "$ATS" "False" "$T2_CSV"
done

# T3 — 3-seed EMA ensemble + TTA, full sweep.
T3_CSV="$SWEEP_DIR/T3_ema_ensemble_tta.csv"
echo "tag,adapt_ts,tta,f1,precision,recall,mae,rmse,ap" > "$T3_CSV"
echo ""
echo "------------------------------------------------------------"
echo "  T3 — 3-seed EMA ensemble + TTA"
echo "------------------------------------------------------------"
for ATS in "${SWEEP_TS[@]}"; do
  run_ensemble "ENS3_EMA_TTA" "$ATS" "True" "$T3_CSV"
done

# --------------------------------------------------------------------
# Final report.
# --------------------------------------------------------------------
print_table() {
  local title=$1; local csv=$2
  echo ""
  echo "============================================================"
  echo "  $title"
  echo "============================================================"
  printf "  %-14s  %-6s  %-6s  %-8s  %-9s  %-7s  %-6s  %-6s  %-6s\n" \
         "tag" "ts" "tta" "F1" "PRECISION" "RECALL" "MAE" "RMSE" "AP"
  awk -F, 'NR>1 { printf "  %-14s  %-6s  %-6s  %-8.4f  %-9.4f  %-7.4f  %-6.2f  %-6.2f  %-6.4f\n",
                         $1, $2, $3, $4, $5, $6, $7, $8, $9 }' "$csv"
}
print_table "T1 — per-seed EMA sweep" "$T1_CSV"
print_table "T2 — EMA ensemble (no TTA)" "$T2_CSV"
print_table "T3 — EMA ensemble + TTA" "$T3_CSV"

echo ""
echo "============================================================"
echo "  Best operating points across T1/T2/T3 (Phase 7)"
echo "============================================================"
python3 - "$T1_CSV" "$T2_CSV" "$T3_CSV" <<'PY'
import sys, csv, os
def load(p, label):
    if not os.path.exists(p): return []
    rows = list(csv.DictReader(open(p)))
    for r in rows: r['_label'] = label
    return rows

all_rows = []
for p, lbl in zip(sys.argv[1:], ["T1_per_seed_EMA", "T2_ENS3_EMA", "T3_ENS3_EMA_TTA"]):
    all_rows += load(p, lbl)

def f(r,k): return float(r[k])
b = max(all_rows, key=lambda r: f(r,'f1')) if all_rows else None
if b: print(f"  best F1 overall  : {b['_label']:<22} {b['tag']:<14} ts={b['adapt_ts']} tta={b['tta']} F1={f(b,'f1'):.4f} P={f(b,'precision'):.4f} R={f(b,'recall'):.4f} MAE={f(b,'mae'):.2f}")
b = min(all_rows, key=lambda r: f(r,'mae')) if all_rows else None
if b: print(f"  best MAE overall : {b['_label']:<22} {b['tag']:<14} ts={b['adapt_ts']} tta={b['tta']} F1={f(b,'f1'):.4f} P={f(b,'precision'):.4f} R={f(b,'recall'):.4f} MAE={f(b,'mae'):.2f}")

print("")
print("  Phase-6 reference (no EMA) for direct comparison:")
print("    best F1   : T3_ENS3_TTA  ts=0.35  F1=0.9634  P=0.9828  R=0.9448  MAE=0.92")
print("    best MAE  : T2_ENS3      ts=0.30  F1=0.9451  P=0.9399  R=0.9503  MAE=0.67")
PY

echo ""
echo "  Logs:    $LOG_DIR"
echo "  Sweep:   $SWEEP_DIR"
echo "============================================================"
