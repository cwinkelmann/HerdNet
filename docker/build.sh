#!/bin/bash
# Build the Phase-13 training container.
#
# Stages the necessary files (code, configs, warm-start checkpoint, one
# training subset + val + test) into a temporary build context, then
# runs `docker build`. The temp dir is cleaned up on exit.
#
# Usage:
#   bash docker/build.sh                 # bundle N=152, default tag
#   TRAIN_N=304 bash docker/build.sh     # bundle a different size
#   IMAGE_TAG=ifa-phase13:rp1 bash docker/build.sh
#
# Environment variables:
#   TRAIN_N    : which training subset to bundle (default 152)
#   SEED       : data-prep seed (default 42)
#   DATA_SRC   : path to the assembled scaling dataset
#                (default: /home/christian/data/training_data/2026_05_08_data_scaling)
#   IMAGE_TAG  : docker tag (default herdnet-phase13-N${TRAIN_N}:latest)

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

TRAIN_N="${TRAIN_N:-152}"
SEED="${SEED:-42}"
DATA_SRC="${DATA_SRC:-/home/christian/data/training_data/2026_05_08_data_scaling}"
IMAGE_TAG="${IMAGE_TAG:-herdnet-phase13-N${TRAIN_N}:latest}"

# Pre-flight: source data + warm-start must exist on the host.
TRAIN_DIR="${DATA_SRC}/train_N${TRAIN_N}_s${SEED}"
VAL_DIR="${DATA_SRC}/val"
TEST_DIR="${DATA_SRC}/test"
WARM_START="${REPO_DIR}/best_models/phase8/b4_seed42/best_model.pth"

for p in "$TRAIN_DIR" "$VAL_DIR" "$TEST_DIR"; do
  [ -d "$p" ] || { echo "ERROR: missing $p"; exit 1; }
done
if [ ! -e "$WARM_START" ]; then
  echo "ERROR: missing warm-start checkpoint $WARM_START"; exit 1
fi
# Resolve in case it's a symlink.
WARM_START_REAL=$(readlink -f "$WARM_START")
[ -f "$WARM_START_REAL" ] || { echo "ERROR: $WARM_START (resolved: $WARM_START_REAL) is not a real file"; exit 1; }

echo "============================================================"
echo "  Building Phase-13 training image"
echo "  TRAIN_N:    $TRAIN_N"
echo "  SEED:       $SEED"
echo "  Data src:   $DATA_SRC"
echo "  Warm-start: $WARM_START_REAL ($(du -h "$WARM_START_REAL" | cut -f1))"
echo "  Image tag:  $IMAGE_TAG"
echo "============================================================"
echo ""

# Stage everything into a temp directory that becomes the docker build context.
STAGE=$(mktemp -d -t phase13_docker_ctx_XXXXXX)
trap "echo 'cleaning up $STAGE'; rm -rf $STAGE" EXIT

echo "Staging build context at $STAGE ..."

# Code + configs (slim copies of the repo).
mkdir -p "$STAGE"
cp pyproject.toml "$STAGE/"
[ -f README.md ] && cp README.md "$STAGE/"
cp -r animaloc "$STAGE/"
cp -r configs  "$STAGE/"
cp -r tools    "$STAGE/"
cp run_phase13.sh "$STAGE/"

# Strip __pycache__ / .pytest_cache to keep the image lean.
find "$STAGE" -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".ipynb_checkpoints" \) \
  -exec rm -rf {} + 2>/dev/null || true

# Warm-start: copy only the b4_seed42 checkpoint (other 5 ensemble members
# aren't needed for fine-tuning and add ~2 GB).
mkdir -p "$STAGE/best_models/phase8/b4_seed42"
cp "$WARM_START_REAL" "$STAGE/best_models/phase8/b4_seed42/best_model.pth"

# Data. Use cp -L to dereference any symlinks. Rename train_N<N>_s<seed>
# to plain "train" inside the image (the entrypoint expects /app/data/train).
mkdir -p "$STAGE/data"
echo "  copying training subset N=$TRAIN_N ..."
cp -rL "$TRAIN_DIR" "$STAGE/data/train"
echo "  copying val ..."
cp -rL "$VAL_DIR" "$STAGE/data/val"
echo "  copying test ..."
cp -rL "$TEST_DIR" "$STAGE/data/test"

# Dockerfile + entrypoint scripts go under docker/ to mirror the repo layout.
mkdir -p "$STAGE/docker"
cp docker/Dockerfile.train     "$STAGE/docker/"
cp docker/train_entrypoint.sh  "$STAGE/docker/"
cp docker/upload_model.py      "$STAGE/docker/"

# Show the build-context size so user knows what's about to be sent to docker.
echo ""
echo "Build context size: $(du -sh "$STAGE" | cut -f1)"
echo ""
echo "Running docker build ..."
docker build \
  --tag "$IMAGE_TAG" \
  --file "$STAGE/docker/Dockerfile.train" \
  "$STAGE"

echo ""
echo "============================================================"
echo "  Built: $IMAGE_TAG"
echo "============================================================"
echo ""
echo "Run example:"
echo ""
echo "  docker run --rm --gpus all \\"
echo "    -e WANDB_API_KEY=\$WANDB_API_KEY \\"
echo "    -e TRAIN_N=$TRAIN_N \\"
echo "    -e WANDB_PROJECT=hn_phase13_data_scaling \\"
echo "    -e UPLOAD_MODEL=1 \\"
echo "    $IMAGE_TAG"
echo ""
echo "Or, to also pull the trained model out as a file:"
echo ""
echo "  docker run --rm --gpus all \\"
echo "    -v \$(pwd)/output:/app/output \\"
echo "    -e WANDB_API_KEY=\$WANDB_API_KEY \\"
echo "    -e TRAIN_N=$TRAIN_N \\"
echo "    $IMAGE_TAG"
