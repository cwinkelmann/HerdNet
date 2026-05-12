#!/bin/bash
# Phase 13 — data-scaling experiment, end-to-end (train + eval per N).
#
# For each training-set size N in {19, 38, 76, 152, 304, 608, 1216, 2432, full}:
#   1. Warm-start from best_models/phase8/b4_seed42/best_model.pth
#   2. Train B4 (HerdNetTimmConvNext_Camouflaged_V2 + herdnet_fmo03 loss)
#      for 30 epochs on the N-size train split
#   3. Per-epoch validate on the cropped val patches (cheap; mainly for
#      auto-LR + early-stopping signal — see refactor.md section E for
#      caveats about per-epoch-vs-full-size disagreement)
#   4. After training: full-size stitched evaluation on the fixed val
#      set across a threshold sweep, with EVERY eval pushed to wandb
#      so you can monitor live remotely.
#
# Doing train+eval per N (rather than all training first then all eval
# afterwards) means:
#   - Earliest N has eval results visible in wandb within ~30 min of
#     starting the script.
#   - If a later training fails for any reason, the smaller-N results
#     are already on disk + wandb — nothing wasted.
#
# Usage:
#   DATA_ROOT=/path/to/2026_05_08_data_scaling \
#   bash run_phase13.sh [START_FROM]
#
# Environment variables:
#   DATA_ROOT      : path to the assembled scaling dataset (required)
#   WARM_START     : warm-start checkpoint
#                    (default: best_models/phase8/b4_seed42/best_model.pth)
#   SEED           : data-prep seed (default 42)
#   SKIP_MISSING   : 1 = silently skip Ns whose split isn't synced yet
#                    (default 0 = abort)
#   WANDB_FLAG     : True/False (default True). Eval also goes to wandb.
#   WANDB_PROJECT  : default 'hn_phase13_data_scaling'
#                    (eval logs under '<project>_Test' per
#                     animaloc/utils/inference.py:171)
#   SWEEP_TS       : space-separated thresholds for the per-N eval sweep
#                    (default '0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70')

set -e
export LC_NUMERIC=C

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

CONFIG_DIR="$SCRIPT_DIR/configs/demo"
CONFIG="data_scaling_b4"

DATA_ROOT="${DATA_ROOT:?ERROR: set DATA_ROOT to the assembled scaling dataset path}"
WARM_START="${WARM_START:-$SCRIPT_DIR/best_models/phase8/b4_seed42/best_model.pth}"
SEED="${SEED:-42}"
SKIP_MISSING="${SKIP_MISSING:-0}"
START_FROM=${1:-1}

WANDB_FLAG="${WANDB_FLAG:-True}"
WANDB_PROJECT="${WANDB_PROJECT:-hn_phase13_data_scaling}"

# Optional: override the dataset's augmentation_multiplier (default 75 in
# configs/demo/datasets/data_scaling.yaml). Useful for big-N runs where
# 75× multiplier is overkill — set AUG_MULT=1 for N≥1216.
AUG_MULT="${AUG_MULT:-}"

read -r -a SWEEP_TS <<< "${SWEEP_TS:-0.20 0.25 0.30 0.35 0.40 0.50 0.60 0.70}"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="/tmp/phase13_${TS}"
SWEEP_DIR="$SCRIPT_DIR/output/phase13_eval_${TS}"
mkdir -p "$LOG_DIR" "$SWEEP_DIR"

# All training-set sizes the data-prep team produced.
SIZES=(19 38 76 152 304 608 1216 2432 full)

# Fixed validation paths — used for both per-epoch (cheap) and
# full-size stitched (honest) evaluation.
VAL_CROP_CSV="$DATA_ROOT/val/herdnet_format_512_0_crops.csv"
VAL_CROP_ROOT="$DATA_ROOT/val/crops_512_numNone_overlap0"
VAL_FULL_CSV="$DATA_ROOT/val/herdnet_format.csv"
VAL_FULL_IMAGES="$DATA_ROOT/val/Default"

echo "============================================================"
echo "  Phase 13 — data scaling: train + eval per N"
echo "  Data root:     $DATA_ROOT"
echo "  Warm-start:    $WARM_START"
echo "  Sizes:         ${SIZES[*]}"
echo "  Eval sweep ts: ${SWEEP_TS[*]}"
echo "  Seed:          $SEED"
echo "  Wandb:         flag=$WANDB_FLAG project=$WANDB_PROJECT (eval → ${WANDB_PROJECT}_Test)"
echo "  Logs:          $LOG_DIR"
echo "  Eval sweep:    $SWEEP_DIR"
echo "  START_FROM:    $START_FROM"
echo "============================================================"

# Pre-flight: warm-start + val both required.
[ -e "$WARM_START" ] || { echo "  ERROR: warm-start checkpoint not found: $WARM_START"; exit 1; }
[ -f "$VAL_CROP_CSV" ] || { echo "  ERROR: missing $VAL_CROP_CSV"; exit 1; }
[ -d "$VAL_CROP_ROOT" ] || { echo "  ERROR: missing $VAL_CROP_ROOT"; exit 1; }
[ -f "$VAL_FULL_CSV" ] || { echo "  ERROR: missing $VAL_FULL_CSV"; exit 1; }
[ -d "$VAL_FULL_IMAGES" ] || { echo "  ERROR: missing $VAL_FULL_IMAGES"; exit 1; }

# Pre-flight: filter SIZES to splits that actually exist on disk.
KEPT_SIZES=()
for N in "${SIZES[@]}"; do
  if [ "$N" = "full" ]; then DIR="$DATA_ROOT/train_Nfull_s${SEED}"
  else                       DIR="$DATA_ROOT/train_N${N}_s${SEED}"
  fi
  ok=1
  [ -d "$DIR" ] || ok=0
  [ -f "$DIR/herdnet_format.csv" ] || ok=0
  [ -d "$DIR/Default" ] || ok=0
  if [ "$ok" -eq 1 ]; then
    KEPT_SIZES+=("$N")
  else
    if [ "$SKIP_MISSING" = "1" ]; then
      echo "  WARN: skipping N=$N (missing $DIR or its contents)"
    else
      echo "  ERROR: missing $DIR (or its herdnet_format.csv / Default)"
      echo "         set SKIP_MISSING=1 to proceed without this split"
      exit 1
    fi
  fi
done
SIZES=("${KEPT_SIZES[@]}")
echo "  Pre-flight OK — ${#SIZES[@]} splits will run: ${SIZES[*]}"
echo ""

# Aggregate sweep CSV across all N (one row per (N, threshold)).
SUMMARY_CSV="$SWEEP_DIR/summary.csv"
echo "N,seed,adapt_ts,f1,precision,recall,mae,rmse,ap" > "$SUMMARY_CSV"

TRAIN_RESULTS=()
EVAL_RESULTS=()
TRAIN_FAILED=0
EVAL_FAILED=0

# Helper: full-size stitched eval at one threshold (writes to wandb too).
run_eval_at_threshold() {
  local N=$1; local MODEL=$2; local ATS=$3
  local SUB="$SWEEP_DIR/N${N}/ts_${ATS}"
  local LOG="$LOG_DIR/N${N}_eval_ts${ATS}.log"
  mkdir -p "$SUB"
  echo "    [$(date '+%H:%M:%S')] N=$N @ ts=$ATS"
  (cd "$SUB" && python "$SCRIPT_DIR/tools/infer.py" \
      --config-dir "$CONFIG_DIR" \
      --config-name "$CONFIG" \
      --images "$VAL_FULL_IMAGES" \
      --model "$MODEL" \
      --evaluate \
      --overrides \
        "wandb_flag=$WANDB_FLAG" \
        "wandb_project=$WANDB_PROJECT" \
        "wandb_run=phase13_eval_N${N}_ts${ATS}" \
        "datasets.test.csv_file=$VAL_FULL_CSV" \
        "datasets.test.root_dir=$VAL_FULL_IMAGES" \
        "training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=$ATS") \
      > "$LOG" 2>&1 || echo "    failed at ts=$ATS"
  local M="$SUB/metrics_results.csv"
  if [ -f "$M" ]; then
    awk -F, -v n="$N" -v s="$SEED" -v ats="$ATS" '$1=="binary" {
      printf "%s,%s,%s,%.4f,%.4f,%.4f,%.2f,%.2f,%.4f\n",
             n, s, ats, $6, $5, $4, $8, $11, $12 }' "$M" >> "$SUMMARY_CSV"
  fi
}

# ---------------------------------------------------------------
# Main loop — train then eval, per N.
# ---------------------------------------------------------------
for i in "${!SIZES[@]}"; do
  num=$((i + 1))
  N=${SIZES[$i]}

  if [ "$num" -lt "$START_FROM" ]; then
    echo ""
    echo "  SKIP [$num/${#SIZES[@]}] N=$N"
    TRAIN_RESULTS+=("SKIP  N=$N")
    EVAL_RESULTS+=("SKIP  N=$N")
    continue
  fi

  if [ "$N" = "full" ]; then
    TRAIN_DIR="$DATA_ROOT/train_Nfull_s${SEED}"
    OUT_DIR="phase13_Nfull_s${SEED}"
    RUN_NAME="phase13_Nfull_s${SEED}"
  else
    TRAIN_DIR="$DATA_ROOT/train_N${N}_s${SEED}"
    OUT_DIR="phase13_N${N}_s${SEED}"
    RUN_NAME="phase13_N${N}_s${SEED}"
  fi

  TRAIN_CSV="$TRAIN_DIR/herdnet_format.csv"
  TRAIN_ROOT="$TRAIN_DIR/Default"
  TRAIN_LOG="$LOG_DIR/${num}_N${N}_train.log"

  echo ""
  echo "============================================================"
  echo "  [$num/${#SIZES[@]}] N=$N — train then eval"
  echo "  Train:    $TRAIN_DIR (warm-start from $WARM_START)"
  echo "  Out:      output/$OUT_DIR/<date>/<time>/best_model.pth"
  echo "  Log:      $TRAIN_LOG"
  echo "  Started:  $(date '+%Y-%m-%d %H:%M:%S')"
  echo "============================================================"

  # ----- Training -----
  echo ""
  echo "  --- TRAIN N=$N ---"

  # Optional augmentation_multiplier override (Phase 13 Option C: drop
  # to 1 for large N where the multiplier becomes overkill).
  TRAIN_EXTRA=()
  if [ -n "$AUG_MULT" ]; then
    TRAIN_EXTRA+=("datasets.train.augmentation_multiplier=$AUG_MULT")
    echo "  AUG_MULT override: $AUG_MULT (default in config is 75)"
  fi

  if python tools/train.py \
      --config-path "$CONFIG_DIR" \
      --config-name "$CONFIG" \
      "seed=$SEED" \
      "datasets.train.csv_file=$TRAIN_CSV" \
      "datasets.train.root_dir=$TRAIN_ROOT" \
      "datasets.validate.csv_file=$VAL_CROP_CSV" \
      "datasets.validate.root_dir=$VAL_CROP_ROOT" \
      "datasets.test.csv_file=$VAL_FULL_CSV" \
      "datasets.test.root_dir=$VAL_FULL_IMAGES" \
      "model.load_from=$WARM_START" \
      "wandb_flag=$WANDB_FLAG" \
      "wandb_project=$WANDB_PROJECT" \
      "wandb_run=$RUN_NAME" \
      "+wandb_tags=[phase13,data_scaling,N${N}]" \
      "${TRAIN_EXTRA[@]}" \
      "hydra.run.dir=./output/${OUT_DIR}/\${now:%Y-%m-%d}/\${now:%H-%M-%S}" \
      2>&1 | tee "$TRAIN_LOG"; then
    echo "  => TRAIN PASSED [$num/${#SIZES[@]}] N=$N $(date '+%H:%M:%S')"
    TRAIN_RESULTS+=("PASS  N=$N")
  else
    echo "  => TRAIN FAILED [$num/${#SIZES[@]}] N=$N $(date '+%H:%M:%S')"
    TRAIN_RESULTS+=("FAIL  N=$N")
    EVAL_RESULTS+=("SKIP  N=$N (train failed)")
    TRAIN_FAILED=$((TRAIN_FAILED + 1))
    continue
  fi

  # ----- Eval (full-size stitched threshold sweep, all logged to wandb) -----
  MODEL=$(find "$SCRIPT_DIR/output/$OUT_DIR" -name best_model.pth -printf '%T@ %p\n' 2>/dev/null \
            | sort -rn | head -1 | cut -d' ' -f2-)
  if [ -z "$MODEL" ] || [ ! -f "$MODEL" ]; then
    echo "  => EVAL SKIPPED [$num/${#SIZES[@]}] N=$N: no best_model.pth"
    EVAL_RESULTS+=("MISS  N=$N (no checkpoint)")
    continue
  fi

  echo ""
  echo "  --- EVAL N=$N (model: $MODEL) ---"
  echo "  Threshold sweep: ${SWEEP_TS[*]}"
  for ATS in "${SWEEP_TS[@]}"; do
    run_eval_at_threshold "$N" "$MODEL" "$ATS"
  done

  echo ""
  echo "  --- N=$N best operating points (from this run's sweep) ---"
  python3 - "$N" "$SWEEP_DIR/N${N}" <<'PY'
import sys, csv, os, glob
n, base = sys.argv[1], sys.argv[2]
rows = []
for d in sorted(glob.glob(os.path.join(base, "ts_*"))):
    p = os.path.join(d, "metrics_results.csv")
    if not os.path.exists(p): continue
    ts = d.split("ts_")[-1]
    for r in csv.DictReader(open(p)):
        if r["class"] == "binary":
            rows.append({"ats": ts, **{k: float(v) for k, v in r.items() if k not in ("class","species")}})
if not rows: print(f"    N={n}: no rows"); sys.exit(0)
def f(r,k): return r[k]
bf1 = max(rows, key=lambda r: f(r,'f1_score'))
bma = min(rows, key=lambda r: f(r,'mae'))
print(f"    best F1   : F1={f(bf1,'f1_score'):.4f}  MAE={f(bf1,'mae'):.2f}  P={f(bf1,'precision'):.4f}  R={f(bf1,'recall'):.4f}  ts={bf1['ats']}")
print(f"    best MAE  : F1={f(bma,'f1_score'):.4f}  MAE={f(bma,'mae'):.2f}  P={f(bma,'precision'):.4f}  R={f(bma,'recall'):.4f}  ts={bma['ats']}")
PY

  EVAL_RESULTS+=("PASS  N=$N")
done

# ---------------------------------------------------------------
# Final summary across all N.
# ---------------------------------------------------------------
echo ""
echo "============================================================"
echo "  Phase 13 — overall summary"
echo "============================================================"
echo "  Training:"
for r in "${TRAIN_RESULTS[@]}"; do echo "    $r"; done
echo "  Eval:"
for r in "${EVAL_RESULTS[@]}"; do echo "    $r"; done
echo ""
echo "============================================================"
echo "  Phase 13 — scaling curve (best F1 / best MAE per N)"
echo "============================================================"
python3 - "$SUMMARY_CSV" <<'PY'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows: print("  no rows"); sys.exit(0)
by_N = {}
for r in rows:
    by_N.setdefault(r['N'], []).append(r)
def f(r,k): return float(r[k])
print(f"  {'N':<8} {'best F1':<10} {'@ ts':<6}  {'best MAE':<10} {'@ ts':<6}  {'P@bestF1':<10}  {'R@bestF1':<10}")
for n in ['19','38','76','152','304','608','1216','2432','full']:
    if n not in by_N: continue
    bf1 = max(by_N[n], key=lambda r: f(r,'f1'))
    bma = min(by_N[n], key=lambda r: f(r,'mae'))
    print(f"  {n:<8} {f(bf1,'f1'):<10.4f} {bf1['adapt_ts']:<6}  {f(bma,'mae'):<10.2f} {bma['adapt_ts']:<6}  {f(bf1,'precision'):<10.4f}  {f(bf1,'recall'):<10.4f}")
PY

echo ""
echo "  Logs:           $LOG_DIR"
echo "  Eval sweep:     $SWEEP_DIR"
echo "  Summary CSV:    $SUMMARY_CSV"
echo "  Wandb projects: $WANDB_PROJECT (training) + ${WANDB_PROJECT}_Test (eval)"
echo ""
echo "  TEST set evaluation is intentionally NOT included here."
echo "  Run it ONCE manually after picking the production N + threshold:"
echo ""
echo "    python tools/infer.py --config-dir configs/demo --config-name data_scaling_b4 \\"
echo "      --images $DATA_ROOT/test/Default \\"
echo "      --model output/phase13_N<best>_s${SEED}/<date>/<time>/best_model.pth \\"
echo "      --evaluate \\"
echo "      --overrides datasets.test.csv_file=$DATA_ROOT/test/herdnet_format.csv \\"
echo "                  datasets.test.root_dir=$DATA_ROOT/test/Default \\"
echo "                  training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=<best from V>"
echo "============================================================"

if [ "$TRAIN_FAILED" -gt 0 ]; then exit 1; fi
