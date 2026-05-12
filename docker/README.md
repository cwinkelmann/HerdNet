# Phase-13 training container

A self-contained Docker image that bundles **code + warm-start checkpoint + one training subset + the fixed val/test sets**, runs a single Phase-13 fine-tune, and uploads the resulting best model as a wandb artifact.

Intended for remote GPU machines (RunPod, AWS EC2 g4/g5, etc.) where you push the image, run one container, and walk away.

## What's inside

| Path inside image | Source | Approx. size |
|---|---|---|
| `/app/animaloc/`, `/app/configs/`, `/app/tools/` | Repo | ~30 MB |
| `/app/best_models/phase8/b4_seed42/best_model.pth` | The Phase-8 production warm-start | ~400 MB |
| `/app/data/train/` | One `train_N<N>_s<seed>/` subset (chosen at build time) | varies (10 MB → 4 GB) |
| `/app/data/val/` | Fixed validation set (Phase 13) | ~125 MB |
| `/app/data/test/` | Fixed test set (Phase 13) | ~1.1 GB |
| `/app/docker/train_entrypoint.sh` | The entry script | ~3 KB |
| Base image (`pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime`) | DockerHub | ~6 GB |

Total image size for `TRAIN_N=152`: roughly **8 GB**. For `TRAIN_N=full` (~6900 training frames): roughly **20 GB**.

If you'd rather not bake the data into the image, mount it instead — see "Alternative: mount data" below.

## Build

```bash
# bundle N=152 (smallest practical size, ~8 GB image)
bash docker/build.sh

# bundle a bigger N
TRAIN_N=608 bash docker/build.sh

# custom image tag
TRAIN_N=304 IMAGE_TAG=ifa-phase13:rp1 bash docker/build.sh
```

The build script stages everything into a temp directory so the `docker build` context doesn't include the entire repo (with its 100+ GB `output/`). It cleans up the tempdir after.

## Run

Single training, with wandb integration:

```bash
docker run --rm --gpus all \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e WANDB_PROJECT=hn_phase13_data_scaling \
  -e TRAIN_N=152 \
  herdnet-phase13-N152:latest
```

To also pull the trained model out as a file on the host:

```bash
docker run --rm --gpus all \
  -v $(pwd)/output:/app/output \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e TRAIN_N=152 \
  herdnet-phase13-N152:latest
```

Best model lands at `output/phase13_N152_s42_docker/<date>/<time>/best_model.pth` on the host.

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `WANDB_API_KEY` | unset | **Required** for wandb upload; if unset, the run drops to wandb offline mode and the artifact upload is skipped |
| `WANDB_PROJECT` | `hn_phase13_data_scaling` | wandb project for training metrics |
| `WANDB_FLAG` | `True` | Set `False` to disable wandb entirely |
| `TRAIN_N` | `152` (build-time default; used for naming only) | Training-set tag; data is already baked in |
| `SEED` | `42` | Random seed |
| `AUG_MULT` | unset (config default 75) | Override `augmentation_multiplier`; set to `1` for big-N runs to save compute |
| `UPLOAD_MODEL` | `1` | Set `0` to skip the post-training wandb artifact upload |
| `ARTIFACT_NAME` | `phase13_best_model` | wandb artifact name |

## Alternative: mount data instead of baking it in

If you want a leaner image (~2 GB plus PyTorch base), build with a stub training subset (e.g. `TRAIN_N=19`) and then mount the real data at runtime:

```bash
# build once with the smallest subset just so the image structure is valid
TRAIN_N=19 bash docker/build.sh

# then run with the real data mounted, overriding /app/data
docker run --rm --gpus all \
  -v /path/to/2026_05_08_data_scaling/train_N608_s42:/app/data/train \
  -v /path/to/2026_05_08_data_scaling/val:/app/data/val \
  -v /path/to/2026_05_08_data_scaling/test:/app/data/test \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  -e TRAIN_N=608 \
  -e AUG_MULT=1 \
  herdnet-phase13-N19:latest
```

The volume mounts mask `/app/data/{train,val,test}` from the baked image, so the running container uses the host data instead. Same image, any N.

## What gets uploaded to wandb

If `UPLOAD_MODEL=1` and `WANDB_API_KEY` is set, the entrypoint:

1. Runs training; per-epoch metrics + final SUMMARY go to `WANDB_PROJECT` (the training run is named `phase13_N<N>_s<seed>_docker`).
2. After training, opens a second wandb run named `<run>_artifact` and uploads `best_model.pth` as a wandb `Artifact` with metadata `{train_n, seed, best_f1, mae, best_val, epochs, ...}` parsed from the final SUMMARY log line.

Two runs (one for training metrics, one for the model artifact) instead of one is intentional — keeps the artifact upload from interfering with the training run's metric history.

## Caveats

- **Image size**: the data layer is the biggest contributor. `TRAIN_N=152` → ~8 GB; `TRAIN_N=full` → ~20 GB. Push to a registry that handles large images well (DockerHub Pro, AWS ECR, etc.) or use the volume-mount route.
- **PyTorch base image** is `~6 GB` by itself. If you need it smaller, switch to the slim variant in `Dockerfile.train` — but you'll need to install PyTorch + CUDA bits yourself, which is fiddly.
- **GPU required**: `--gpus all` is mandatory. The base image expects an NVIDIA GPU and CUDA driver on the host.
- **wandb storage**: each artifact is ~400 MB; if you run many of these, mind the storage quota on your wandb account.
- **Reproducibility**: image bundles `pyproject.toml` but the exact CUDA / cuDNN versions come from the PyTorch base tag. Pin to a specific tag if you want bit-exact reproducibility months later.
