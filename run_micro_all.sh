#!/bin/bash
# Smoke test: run all 4 model variants for 1 epoch on the micro dataset with WandB
# Usage: bash run_micro_all.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="configs/demo"

echo "============================================================"
echo "  HerdNet Micro Smoke Test — All Model Variants, 1 Epoch"
echo "============================================================"

VARIANTS=(
  # --- Baselines (no camouflage features) ---
  "fmo03_micro_dla34_classic:1. DLA-34 Classic (Delplanque original)"
  "fmo03_micro_dla34:2. DLA-34 Timm (modernized backbone)"
  "fmo03_micro_baseline:3. ConvNeXt (HerdNetConvNeXt, separate architecture)"
  "fmo03_micro_convnext_plain:4. ConvNeXt plain (camouflage arch, features OFF)"
  # --- Camouflage features ON ---
  "fmo03_micro_v1:5. ConvNeXt + Gabor+Edge+MultiRes (V1)"
  "fmo03_micro_convnextv2:6. ConvNeXt-V2 + Gabor+Edge+MultiRes (FCMAE)"
  "fmo03_micro_efficientvit:7. EfficientViT + Gabor+Edge+MultiRes"
  # --- Architecture improvements ---
  "fmo03_micro_v2:8. ConvNeXt + BiFPN+DeformConv (V2)"
  "fmo03_micro_v3:9. ConvNeXt-V2 + BiFPN+DeformConv+P2P (V3)"
)

PASSED=0
FAILED=0
RESULTS=()

for entry in "${VARIANTS[@]}"; do
  CONFIG="${entry%%:*}"
  LABEL="${entry##*:}"

  echo ""
  echo "------------------------------------------------------------"
  echo "  Running: $LABEL"
  echo "  Config:  $CONFIG"
  echo "------------------------------------------------------------"

  if python tools/train.py \
      --config-path "$SCRIPT_DIR/$CONFIG_DIR" \
      --config-name "$CONFIG" \
      2>&1 | tee "/tmp/micro_${CONFIG}.log"; then
    echo "  => PASSED: $LABEL"
    RESULTS+=("PASS  $LABEL")
    PASSED=$((PASSED + 1))
  else
    echo "  => FAILED: $LABEL"
    RESULTS+=("FAIL  $LABEL")
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
echo "  Passed: $PASSED / $((PASSED + FAILED))"
if [ $FAILED -gt 0 ]; then
  echo "  SOME RUNS FAILED — check /tmp/micro_*.log"
  exit 1
else
  echo "  All runs passed!"
fi
