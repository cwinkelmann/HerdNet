#!/bin/bash
# Run all 3 FMO03 new dataset training configs sequentially
set -e

echo "=== [1/3] Clean (no hard negatives) ==="
echo "Already running — skipping"

echo "=== [2/3] Full (with hard negatives) ==="
python tools/train.py --config-path ../configs/demo --config-name fmo03_new_full_convnext 2>&1 | tee 20260407_fmo03_new_full_training.log

echo "=== [3/3] ObjectAwareRandomCrop (full-size images) ==="
python tools/train.py --config-path ../configs/demo --config-name fmo03_new_objcrop_convnext 2>&1 | tee 20260407_fmo03_new_objcrop_training.log

echo "=== All done ==="
