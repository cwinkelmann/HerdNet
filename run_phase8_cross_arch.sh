#!/bin/bash
# Phase 8 — cross-architecture ensemble (B3 + B4) across 3 seeds.
#
# B3 = HerdNetConvNeXt with Gabor projection (3 seeds: 42, 123, 7)
# B4 = HerdNetTimmConvNext_Camouflaged_V2 with BiFPN+DeformConv (3 seeds)
# Both are top-2 per Phase 5 cross-seed analysis. We average their
# pre-LMDS heatmaps for each tile, then apply LMDS + stitcher as usual.
#
# Smoke test (2 models, ts=0.35, no TTA): F1=0.961, P=0.983, R=0.939, MAE=1.17
# Reference points to beat (Phase 6, 3-seed B4 only):
#   best F1   : 0.9634 (B4×3 + TTA, ts=0.35)
#   best MAE  : 0.67   (B4×3, no TTA, ts=0.30)
#
# Phase 8a: 6-model ensemble (B3×3 + B4×3), no TTA, full threshold sweep
# Phase 8b: same 6-model ensemble with TTA at 4 key thresholds
#           (TTA + 6-model ensemble = 24× single-seed inference cost — capped)

set -e
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="$SCRIPT_DIR/configs/demo"
FULL_VAL_IMAGES="$SCRIPT_DIR/data_fmo03/val/Default"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/tmp/phase8_${TS}"
SWEEP_DIR="$SCRIPT_DIR/output/phase8_sweep_${TS}"
mkdir -p "$LOG_DIR" "$SWEEP_DIR"

# B3 checkpoints (HerdNetConvNeXt + Gabor; uses fmo03_full_convnextv2_gabor cfg).
B3_S42="$SCRIPT_DIR/output/phase3_B3_fmo03_full_convnextv2_gabor/2026-05-05/10-44-02/best_model.pth"
B3_S123="$SCRIPT_DIR/output/seed123_B3_fmo03_full_convnextv2_gabor/2026-05-06/00-12-01/best_model.pth"
B3_S7="$SCRIPT_DIR/output/seed7_B3_fmo03_full_convnextv2_gabor/2026-05-06/11-20-16/best_model.pth"

# B4 checkpoints (CamouflageHerdNetConvNeXtV2; uses fmo03_full_v2_bifpn cfg).
B4_S42="$SCRIPT_DIR/output/phase3_B4_fmo03_full_v2_bifpn/2026-05-05/12-05-03/best_model.pth"
B4_S123="$SCRIPT_DIR/output/seed123_B4_fmo03_full_v2_bifpn/2026-05-06/01-35-04/best_model.pth"
B4_S7="$SCRIPT_DIR/output/seed7_B4_fmo03_full_v2_bifpn/2026-05-06/12-35-31/best_model.pth"

ALL_MODELS=("$B3_S42" "$B3_S123" "$B3_S7" "$B4_S42" "$B4_S123" "$B4_S7")
for p in "${ALL_MODELS[@]}"; do
  [ -f "$p" ] || { echo "ERROR: missing checkpoint $p"; exit 1; }
done

# tools/ensemble_infer.py reads each checkpoint's own cfg to instantiate the
# correct model class — so the --config-name we pass at the CLI is only used
# as fallback for cfg.datasets / training_settings. Use B4's config because
# it's the dataset+stitcher+evaluator setup we standardized on.
SHARED_CFG="fmo03_full_v2_bifpn"

echo "============================================================"
echo "  Phase 8 — cross-arch ensemble (B3×3 + B4×3, 6 members)"
echo "  Architectures: HerdNetConvNeXt (B3) + CamouflageHerdNetConvNeXtV2 (B4)"
echo "  Seeds:         42, 123, 7"
echo "  Logs:          $LOG_DIR"
echo "  Sweep:         $SWEEP_DIR"
echo "============================================================"

SWEEP_TS=(0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70)
TTA_PROBE_TS=(0.30 0.35 0.40 0.50)

run_ensemble() {
  local TAG=$1; local ATS=$2; local TTA=$3; local CSV_OUT=$4
  shift 4
  local MODELS=("$@")
  local SUB="$SWEEP_DIR/${TAG}/ts_${ATS}_tta${TTA}"
  local LOG="$LOG_DIR/${TAG}_ts${ATS}_tta${TTA}.log"
  mkdir -p "$SUB"
  echo "  [$(date '+%H:%M:%S')] $TAG @ ts=$ATS tta=$TTA"
  (cd "$SUB" && python "$SCRIPT_DIR/tools/ensemble_infer.py" \
      --config-dir "$CONFIG_DIR" \
      --config-name "$SHARED_CFG" \
      --images "$FULL_VAL_IMAGES" \
      --models "${MODELS[@]}" \
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
# Phase 8a — 6-model cross-arch ensemble, no TTA, full sweep.
# ----------------------------------------------------------------------
T1_CSV="$SWEEP_DIR/T1_cross_arch_no_tta.csv"
echo "tag,adapt_ts,tta,f1,precision,recall,mae,rmse,ap" > "$T1_CSV"
echo ""
echo "------------------------------------------------------------"
echo "  Phase 8a — B3×3 + B4×3 ensemble (no TTA), full sweep"
echo "------------------------------------------------------------"
for ATS in "${SWEEP_TS[@]}"; do
  run_ensemble "ENS6_CROSS" "$ATS" "False" "$T1_CSV" "${ALL_MODELS[@]}"
done

# ----------------------------------------------------------------------
# Phase 8b — same 6-model ensemble with TTA at probe thresholds.
# ----------------------------------------------------------------------
T2_CSV="$SWEEP_DIR/T2_cross_arch_tta.csv"
echo "tag,adapt_ts,tta,f1,precision,recall,mae,rmse,ap" > "$T2_CSV"
echo ""
echo "------------------------------------------------------------"
echo "  Phase 8b — B3×3 + B4×3 ensemble + TTA, key thresholds"
echo "------------------------------------------------------------"
for ATS in "${TTA_PROBE_TS[@]}"; do
  run_ensemble "ENS6_CROSS_TTA" "$ATS" "True" "$T2_CSV" "${ALL_MODELS[@]}"
done

# ----------------------------------------------------------------------
# Final report.
# ----------------------------------------------------------------------
print_table() {
  local title=$1; local csv=$2
  echo ""
  echo "============================================================"
  echo "  $title"
  echo "============================================================"
  printf "  %-15s  %-6s  %-6s  %-8s  %-9s  %-7s  %-6s  %-6s  %-6s\n" \
         "tag" "ts" "tta" "F1" "PRECISION" "RECALL" "MAE" "RMSE" "AP"
  awk -F, 'NR>1 { printf "  %-15s  %-6s  %-6s  %-8.4f  %-9.4f  %-7.4f  %-6.2f  %-6.2f  %-6.4f\n",
                         $1, $2, $3, $4, $5, $6, $7, $8, $9 }' "$csv"
}
print_table "Phase 8a — cross-arch ensemble (no TTA)" "$T1_CSV"
print_table "Phase 8b — cross-arch ensemble + TTA" "$T2_CSV"

echo ""
echo "============================================================"
echo "  Best operating points across Phase 8"
echo "============================================================"
python3 - "$T1_CSV" "$T2_CSV" <<'PY'
import sys, csv, os
def load(p, label):
    if not os.path.exists(p): return []
    rows = list(csv.DictReader(open(p)))
    for r in rows: r['_label'] = label
    return rows
all_rows = []
for p, lbl in zip(sys.argv[1:], ["8a_ENS6_CROSS", "8b_ENS6_CROSS_TTA"]):
    all_rows += load(p, lbl)
def f(r,k): return float(r[k])
b = max(all_rows, key=lambda r: f(r,'f1')) if all_rows else None
if b: print(f"  best F1 overall  : {b['_label']:<22} ts={b['adapt_ts']} tta={b['tta']} F1={f(b,'f1'):.4f} P={f(b,'precision'):.4f} R={f(b,'recall'):.4f} MAE={f(b,'mae'):.2f}")
b = min(all_rows, key=lambda r: f(r,'mae')) if all_rows else None
if b: print(f"  best MAE overall : {b['_label']:<22} ts={b['adapt_ts']} tta={b['tta']} F1={f(b,'f1'):.4f} P={f(b,'precision'):.4f} R={f(b,'recall'):.4f} MAE={f(b,'mae'):.2f}")
print("")
print("  References to beat:")
print("    Phase-6 best F1   : T3_ENS3_TTA  (B4×3 + TTA)  ts=0.35  F1=0.9634  MAE=0.92")
print("    Phase-6 best MAE  : T2_ENS3      (B4×3)         ts=0.30  F1=0.9451  MAE=0.67")
print("    Phase-7 EMA       : best F1=0.9540, best MAE=0.75 (no improvement vs Phase 6)")
PY

echo ""
echo "  Logs:    $LOG_DIR"
echo "  Sweep:   $SWEEP_DIR"
echo "============================================================"
