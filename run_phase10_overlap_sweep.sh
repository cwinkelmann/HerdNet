#!/bin/bash
# Phase 10 — sweep stitcher overlap on the production stack.
#
# Phase 9's error analysis showed ~30 % of remaining errors cluster at
# image borders (3 edge FN + 3 corner/edge hi-conf FP, out of 20 total).
# The cheapest fix is to widen the stitcher overlap from 120 to give
# tiles more shared context near image edges.
#
# Tests the Phase-6 best-MAE config (3-seed B4 ensemble, no TTA,
# adapt_ts=0.30) at overlap ∈ {120, 192, 256, 320, 384}. Same
# checkpoints — no retraining. ~30 min total inference.
#
# References to compare against:
#   Phase-6 ENS3 @ overlap=120, ts=0.30 : F1=0.9451  P=0.9399  R=0.9503  MAE=0.67
#   Phase-9 error analysis (same config): 9 FN, 11 FP, 6 hi-conf FP, 3 edge FN
#
# Goal: see if larger overlap drops the edge-cluster errors without
# hurting the rest. If it does, we re-run the F1-best (T3 ensemble+TTA)
# with the winning overlap and update production.

set -e
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="$SCRIPT_DIR/configs/demo"
CONFIG="fmo03_full_v2_bifpn"
FULL_VAL_IMAGES="$SCRIPT_DIR/data_fmo03/val/Default"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/tmp/phase10_${TS}"
SWEEP_DIR="$SCRIPT_DIR/output/phase10_sweep_${TS}"
mkdir -p "$LOG_DIR" "$SWEEP_DIR"

# B4 checkpoints across 3 seeds (Phase-6 ensemble members).
M42="$SCRIPT_DIR/output/phase3_B4_fmo03_full_v2_bifpn/2026-05-05/12-05-03/best_model.pth"
M123="$SCRIPT_DIR/output/seed123_B4_fmo03_full_v2_bifpn/2026-05-06/01-35-04/best_model.pth"
M7="$SCRIPT_DIR/output/seed7_B4_fmo03_full_v2_bifpn/2026-05-06/12-35-31/best_model.pth"

for p in "$M42" "$M123" "$M7"; do
  [ -f "$p" ] || { echo "  ERROR: missing $p"; exit 1; }
done

# Overlap values. Default is 120; max is image_size - tile_size in any
# dimension, but we cap at 384 (75 %) to stay sensible.
OVERLAPS=(120 192 256 320 384)
ATS=0.30  # Phase-6 best-MAE threshold

echo "============================================================"
echo "  Phase 10 — stitcher overlap sweep on B4×3 ensemble"
echo "  Members: B4 seed={42,123,7}"
echo "  Threshold (adapt_ts): $ATS  (Phase-6 best-MAE)"
echo "  Overlap grid: ${OVERLAPS[*]}"
echo "  Logs:    $LOG_DIR"
echo "  Sweep:   $SWEEP_DIR"
echo "============================================================"

CSV="$SWEEP_DIR/overlap_sweep.csv"
echo "overlap,adapt_ts,f1,precision,recall,mae,rmse,ap,detections" > "$CSV"

for OV in "${OVERLAPS[@]}"; do
  SUB="$SWEEP_DIR/overlap_${OV}"
  LOG="$LOG_DIR/overlap_${OV}.log"
  mkdir -p "$SUB"
  echo ""
  echo "  [$(date '+%H:%M:%S')] overlap=$OV"
  (cd "$SUB" && python "$SCRIPT_DIR/tools/ensemble_infer.py" \
      --config-dir "$CONFIG_DIR" \
      --config-name "$CONFIG" \
      --images "$FULL_VAL_IMAGES" \
      --models "$M42" "$M123" "$M7" \
      --evaluate \
      --overrides \
        "losses=herdnet_fmo03" \
        "wandb_flag=False" \
        "training_settings.stitcher.kwargs.overlap=$OV" \
        "training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=$ATS") \
      > "$LOG" 2>&1 || echo "    failed at overlap=$OV"

  M="$SUB/metrics_results.csv"
  D="$SUB/detections.csv"
  if [ -f "$M" ]; then
    NDET=$(tail -n +2 "$D" 2>/dev/null | wc -l)
    awk -F, -v ov="$OV" -v ats="$ATS" -v ndet="$NDET" '$1=="binary" {
      printf "%s,%s,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f,%s\n",
             ov, ats, $6, $5, $4, $8, $11, $12, ndet }' "$M" >> "$CSV"
  fi
done

# ----------------------------------------------------------------------
# Final report.
# ----------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Phase 10 — Overlap Sweep Summary (B4×3 ensemble, ts=0.30)"
echo "============================================================"
printf "  %-8s  %-6s  %-8s  %-9s  %-7s  %-6s  %-6s  %-6s  %-5s\n" \
       "overlap" "ts" "F1" "PRECISION" "RECALL" "MAE" "RMSE" "AP" "Ndet"
awk -F, 'NR>1 { printf "  %-8s  %-6s  %-8.4f  %-9.4f  %-7.4f  %-6.2f  %-6.2f  %-6.4f  %-5s\n",
                       $1, $2, $3, $4, $5, $6, $7, $8, $9 }' "$CSV"
echo ""
echo "  Reference (overlap=120 from Phase 6): F1=0.9451  P=0.9399  R=0.9503  MAE=0.67"
echo ""

# Pick the winning overlap by best MAE (with tie-break on F1).
python3 - <<PY
import csv
rows = list(csv.DictReader(open("$CSV")))
if not rows:
    print("  no rows")
else:
    def k(r): return (float(r["mae"]), -float(r["f1"]))
    best = min(rows, key=k)
    print(f"  Best by MAE  : overlap={best['overlap']}  F1={float(best['f1']):.4f}  MAE={float(best['mae']):.2f}  Ndet={best['detections']}")
    bf1 = max(rows, key=lambda r: float(r["f1"]))
    print(f"  Best by F1   : overlap={bf1['overlap']}  F1={float(bf1['f1']):.4f}  MAE={float(bf1['mae']):.2f}  Ndet={bf1['detections']}")
PY

echo ""
echo "  Logs:    $LOG_DIR"
echo "  Sweep:   $SWEEP_DIR"
echo "============================================================"
