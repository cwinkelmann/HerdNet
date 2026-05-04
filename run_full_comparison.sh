#!/bin/bash
# Full 9-model comparison: 30 epochs on fmo03_new_objcrop_augplus with WandB
# Dataset: 19 train images (aug x75), 12 val images, 512x512 crops
# WandB project: hn_comparison
#
# Usage: bash run_full_comparison.sh [START_FROM]
#   START_FROM: optional, skip runs before this number (e.g., 5 to resume from run 5)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="configs/demo"
START_FROM=${1:-1}
LOG_DIR="/tmp/comparison_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  HerdNet Full Comparison — 9 Models, 30 Epochs"
echo "  Dataset: fmo03_new_objcrop_augplus"
echo "  WandB project: hn_comparison"
echo "  Logs: $LOG_DIR"
echo "  Starting from run: $START_FROM"
echo "============================================================"

VARIANTS=(
  # --- Baselines (no camouflage features) ---
  "1:fmo03_full_dla34_classic:DLA-34 Classic (Delplanque original)"
  "2:fmo03_full_dla34_timm:DLA-34 Timm (modernized backbone)"
  "3:fmo03_full_convnext_baseline:ConvNeXt (HerdNetConvNeXt, separate arch)"
  "4:fmo03_full_convnext_plain:ConvNeXt plain (camouflage arch, features OFF)"
  # --- Camouflage features ON ---
  "5:fmo03_full_v1_gabor:ConvNeXt + Gabor+Edge+MultiRes (V1)"
  "6:fmo03_full_convnextv2_gabor:ConvNeXt-V2 + Gabor+Edge+MultiRes (FCMAE)"
  "7:fmo03_full_efficientvit_gabor:EfficientViT + Gabor+Edge+MultiRes"
  # --- Architecture improvements ---
  "8:fmo03_full_v2_bifpn:ConvNeXt + BiFPN+DeformConv (V2)"
  "9:fmo03_full_v3_p2p:ConvNeXt-V2 + BiFPN+DeformConv+P2P (V3)"
)

PASSED=0
FAILED=0
SKIPPED=0
RESULTS=()

for entry in "${VARIANTS[@]}"; do
  IFS=':' read -r NUM CONFIG LABEL <<< "$entry"

  if [ "$NUM" -lt "$START_FROM" ]; then
    echo ""
    echo "  SKIP [$NUM/9] $LABEL"
    SKIPPED=$((SKIPPED + 1))
    RESULTS+=("SKIP  [$NUM] $LABEL")
    continue
  fi

  LOG_FILE="$LOG_DIR/${NUM}_${CONFIG}.log"

  echo ""
  echo "------------------------------------------------------------"
  echo "  [$NUM/9] $LABEL"
  echo "  Config: $CONFIG"
  echo "  Log:    $LOG_FILE"
  echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "------------------------------------------------------------"

  if python tools/train.py \
      --config-path "$SCRIPT_DIR/$CONFIG_DIR" \
      --config-name "$CONFIG" \
      2>&1 | tee "$LOG_FILE"; then
    echo "  => PASSED [$NUM/9]: $LABEL ($(date '+%H:%M:%S'))"
    RESULTS+=("PASS  [$NUM] $LABEL")
    PASSED=$((PASSED + 1))
  else
    echo "  => FAILED [$NUM/9]: $LABEL ($(date '+%H:%M:%S'))"
    RESULTS+=("FAIL  [$NUM] $LABEL")
    FAILED=$((FAILED + 1))
  fi
done

echo ""
echo "============================================================"
echo "  Summary"
echo "============================================================"
for r in "${RESULTS[@]}"; do
  echo "  $r"
done
echo ""
echo "  Passed: $PASSED  Failed: $FAILED  Skipped: $SKIPPED"
echo "  Logs: $LOG_DIR"
if [ $FAILED -gt 0 ]; then
  echo "  SOME RUNS FAILED — check logs"
  exit 1
else
  echo "  All runs passed!"
fi
