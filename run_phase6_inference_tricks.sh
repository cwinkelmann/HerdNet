#!/bin/bash
# Phase 6 — inference-side "free lunch" tricks on existing B4 checkpoints.
#
# We have B4 (fmo03_full_v2_bifpn + herdnet_fmo03 override) trained at 3 seeds
# (42, 123, 7). Phase 6 tests whether two cheap inference-time tricks can lift
# us past the single-seed plateau (mean F1 ≈ 0.945, mean best MAE ≈ 0.89):
#
#   T1: Test-time augmentation per seed   — set training_settings.stitcher.kwargs.tta=True
#   T2: 3-seed heatmap ensemble (no TTA)  — average raw heatmap+cls outputs across seeds
#   T3: Ensemble + TTA combined           — full grid of inference robustness
#
# All three operate on already-trained checkpoints — no training time spent.
#
# (EMA, the third leg of the lit-review's "Run #1", requires a training-time
# change and is deferred to Phase 7.)
#
# Smoke test (2026-05-07 12:32) showed the 3-seed ensemble at ts=0.35 already
# hits F1=0.947 / MAE=0.75 / P=0.95 / R=0.945 / 180 detections — matching the
# best per-seed MAE and exceeding per-seed mean F1 (0.945). The full sweep
# below maps the threshold curve and tests whether TTA stacks on top.

set -e
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="$SCRIPT_DIR/configs/demo"
CONFIG="fmo03_full_v2_bifpn"
FULL_VAL_IMAGES="$SCRIPT_DIR/data_fmo03/val/Default"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/tmp/phase6_${TS}"
SWEEP_DIR="$SCRIPT_DIR/output/phase6_sweep_${TS}"
mkdir -p "$LOG_DIR" "$SWEEP_DIR"

# B4 checkpoints across 3 seeds.
M42="$SCRIPT_DIR/output/phase3_B4_fmo03_full_v2_bifpn/2026-05-05/12-05-03/best_model.pth"
M123="$SCRIPT_DIR/output/seed123_B4_fmo03_full_v2_bifpn/2026-05-06/01-35-04/best_model.pth"
M7="$SCRIPT_DIR/output/seed7_B4_fmo03_full_v2_bifpn/2026-05-06/12-35-31/best_model.pth"

for p in "$M42" "$M123" "$M7"; do
  if [ ! -f "$p" ]; then
    echo "  ERROR: missing checkpoint: $p"
    exit 1
  fi
done

# Threshold grid for full sweeps (T2, T3).
SWEEP_TS=(0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70)
# Single threshold for per-seed TTA (T1) — keeps the loop short, our
# main TTA conclusion comes from T3 anyway.
TTA_PROBE_TS=(0.30 0.35 0.50)

echo "============================================================"
echo "  Phase 6 — inference-side free lunch on B4 (3 seeds × tricks)"
echo "  Config:        $CONFIG (architecture only — no loss override needed for inference)"
echo "  Sweep grid:    ${SWEEP_TS[*]}"
echo "  TTA probe ts:  ${TTA_PROBE_TS[*]}"
echo "  Logs:          $LOG_DIR"
echo "  Sweep:         $SWEEP_DIR"
echo "============================================================"

# Helper: run single-checkpoint inference at a given ts, append metrics to a CSV.
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

# Helper: run ensemble inference at a given ts.
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
      --models "$M42" "$M123" "$M7" \
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

# ----------------------------------------------------------------------
# T1 — per-seed TTA at probe thresholds (3 seeds × 3 ts = 9 evals).
# ----------------------------------------------------------------------
T1_CSV="$SWEEP_DIR/T1_per_seed_tta.csv"
echo "tag,adapt_ts,tta,f1,precision,recall,mae,rmse,ap" > "$T1_CSV"

echo ""
echo "------------------------------------------------------------"
echo "  T1 — per-seed TTA"
echo "------------------------------------------------------------"
for SEED_TAG in "B4_s42:$M42" "B4_s123:$M123" "B4_s7:$M7"; do
  TAG=${SEED_TAG%%:*}; MODEL=${SEED_TAG##*:}
  for ATS in "${TTA_PROBE_TS[@]}"; do
    run_single "${TAG}_TTA" "$MODEL" "$ATS" "True" "$T1_CSV"
  done
done

# ----------------------------------------------------------------------
# T2 — 3-seed heatmap ensemble (no TTA), full sweep.
# ----------------------------------------------------------------------
T2_CSV="$SWEEP_DIR/T2_ensemble_no_tta.csv"
echo "tag,adapt_ts,tta,f1,precision,recall,mae,rmse,ap" > "$T2_CSV"

echo ""
echo "------------------------------------------------------------"
echo "  T2 — 3-seed heatmap ensemble (no TTA)"
echo "------------------------------------------------------------"
for ATS in "${SWEEP_TS[@]}"; do
  run_ensemble "ENS3" "$ATS" "False" "$T2_CSV"
done

# ----------------------------------------------------------------------
# T3 — 3-seed ensemble + TTA, full sweep (most expensive — ~6 min/threshold).
# ----------------------------------------------------------------------
T3_CSV="$SWEEP_DIR/T3_ensemble_tta.csv"
echo "tag,adapt_ts,tta,f1,precision,recall,mae,rmse,ap" > "$T3_CSV"

echo ""
echo "------------------------------------------------------------"
echo "  T3 — 3-seed ensemble + TTA"
echo "------------------------------------------------------------"
for ATS in "${SWEEP_TS[@]}"; do
  run_ensemble "ENS3_TTA" "$ATS" "True" "$T3_CSV"
done

# ----------------------------------------------------------------------
# Final summary tables.
# ----------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Phase 6 — T1 per-seed TTA results"
echo "============================================================"
printf "  %-12s  %-8s  %-8s  %-9s  %-7s  %-6s  %-6s  %-6s\n" \
       "TAG" "ts" "F1" "PRECISION" "RECALL" "MAE" "RMSE" "AP"
awk -F, 'NR>1 { printf "  %-12s  %-8s  %-8.4f  %-9.4f  %-7.4f  %-6.2f  %-6.2f  %-6.4f\n",
                       $1, $2, $4, $5, $6, $7, $8, $9 }' "$T1_CSV"

echo ""
echo "============================================================"
echo "  Phase 6 — T2 ensemble (no TTA) full sweep"
echo "============================================================"
printf "  %-8s  %-8s  %-9s  %-7s  %-6s  %-6s  %-6s\n" \
       "ts" "F1" "PRECISION" "RECALL" "MAE" "RMSE" "AP"
awk -F, 'NR>1 { printf "  %-8s  %-8.4f  %-9.4f  %-7.4f  %-6.2f  %-6.2f  %-6.4f\n",
                       $2, $4, $5, $6, $7, $8, $9 }' "$T2_CSV"

echo ""
echo "============================================================"
echo "  Phase 6 — T3 ensemble + TTA full sweep"
echo "============================================================"
printf "  %-8s  %-8s  %-9s  %-7s  %-6s  %-6s  %-6s\n" \
       "ts" "F1" "PRECISION" "RECALL" "MAE" "RMSE" "AP"
awk -F, 'NR>1 { printf "  %-8s  %-8.4f  %-9.4f  %-7.4f  %-6.2f  %-6.2f  %-6.4f\n",
                       $2, $4, $5, $6, $7, $8, $9 }' "$T3_CSV"

echo ""
echo "============================================================"
echo "  Best operating points across T1/T2/T3"
echo "============================================================"
python3 - "$T1_CSV" "$T2_CSV" "$T3_CSV" <<'PY'
import sys, csv, os
def load(p, label):
    rows = list(csv.DictReader(open(p)))
    if not rows: return []
    for r in rows: r['_label'] = label
    return rows

all_rows = []
for p, lbl in zip(sys.argv[1:], ["T1_per_seed_TTA", "T2_ENS3", "T3_ENS3_TTA"]):
    if os.path.exists(p): all_rows += load(p, lbl)

def f(r,k): return float(r[k])
print(f"  best F1 overall    : ", end="")
b = max(all_rows, key=lambda r: f(r,'f1')) if all_rows else None
if b: print(f"{b['_label']:<22} {b['tag']:<14} ts={b['adapt_ts']} tta={b['tta']:<5} F1={f(b,'f1'):.4f} P={f(b,'precision'):.4f} R={f(b,'recall'):.4f} MAE={f(b,'mae'):.2f}")

print(f"  best MAE overall   : ", end="")
b = min(all_rows, key=lambda r: f(r,'mae')) if all_rows else None
if b: print(f"{b['_label']:<22} {b['tag']:<14} ts={b['adapt_ts']} tta={b['tta']:<5} F1={f(b,'f1'):.4f} P={f(b,'precision'):.4f} R={f(b,'recall'):.4f} MAE={f(b,'mae'):.2f}")

# Best per-tier
for tier in ["T1_per_seed_TTA", "T2_ENS3", "T3_ENS3_TTA"]:
    rows = [r for r in all_rows if r['_label']==tier]
    if not rows: continue
    bf1 = max(rows, key=lambda r: f(r,'f1'))
    bma = min(rows, key=lambda r: f(r,'mae'))
    print(f"  best F1 in {tier:<18}: F1={f(bf1,'f1'):.4f}  MAE={f(bf1,'mae'):.2f}  ts={bf1['adapt_ts']}")
    print(f"  best MAE in {tier:<17}: F1={f(bma,'f1'):.4f}  MAE={f(bma,'mae'):.2f}  ts={bma['adapt_ts']}")
PY

echo ""
echo "  Logs:    $LOG_DIR"
echo "  Sweep:   $SWEEP_DIR"
echo "============================================================"
