"""
Post-training: attach best_model.pth as a wandb Artifact to the training
run that just finished.

Training pushes per-epoch metrics to wandb live (see train_entrypoint.sh),
so the run already exists when this script runs. We resume that exact
run by ID — wandb writes a `wandb/run-<timestamp>-<id>/` directory under
the training's working directory, and we read the ID from there. Result:
one wandb run per training, with the per-epoch curves AND the final model
attached to it.

If we can't find the wandb run ID for any reason (e.g. wandb ran offline),
we fall back to creating a sidecar run with the same name suffixed by
`_model` so the artifact upload still happens.

Reads everything from environment variables — meant to be invoked from
train_entrypoint.sh after training succeeds.

Required env vars:
  WANDB_PROJECT, WANDB_API_KEY, ARTIFACT_NAME, RUN_NAME, MODEL_PATH

Optional env vars:
  OUT_DIR    : training run's hydra.run.dir; we look for wandb/run-*
               under here to recover the run ID
  TRAIN_LOG  : path to the training log; final SUMMARY line is parsed
               and attached as artifact metadata
  TRAIN_N    : training-set size tag (artifact metadata)
  SEED       : training seed (artifact metadata)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import wandb


def _parse_summary(log_path: str) -> dict:
    """Extract the final SUMMARY line emitted by animaloc.utils.train.

    Pattern matches: best_f1, best_f2, recall, precision, mae, rmse,
    best_val, epochs, model — the keys logged near line ~503 of train.py.
    """
    if not log_path or not os.path.exists(log_path):
        return {}
    pattern = (
        r"\bmodel=(\S+).*?\bbest_f1=([\d.]+).*?\bbest_f2=([\d.]+).*?"
        r"\brecall=([\d.]+).*?\bprecision=([\d.]+).*?\bmae=([\d.]+).*?"
        r"\brmse=([\d.]+).*?\bbest_val=([\d.]+).*?\bepochs=(\d+)"
    )
    summary = {}
    try:
        with open(log_path, "r") as f:
            content = f.read()
        for m in re.finditer(pattern, content):
            summary = {
                "model": m.group(1),
                "best_f1": float(m.group(2)),
                "best_f2": float(m.group(3)),
                "recall": float(m.group(4)),
                "precision": float(m.group(5)),
                "mae": float(m.group(6)),
                "rmse": float(m.group(7)),
                "best_val": float(m.group(8)),
                "epochs": int(m.group(9)),
            }
    except (OSError, ValueError):
        return {}
    return summary


def _find_training_run_id(out_dir: str) -> str | None:
    """Recover the wandb run ID from the training's wandb cache directory.

    During training, wandb (when `wandb_flag=True`) writes its state to
    `<cwd>/wandb/run-<YYYYMMDD>_<HHMMSS>-<id>/`. With Hydra's
    `hydra.run.dir=<out_dir>`, the cwd is <out_dir>, so we look there.

    Returns None if no wandb directory or no run-* sub-directory exists
    (e.g. wandb was offline or disabled).
    """
    if not out_dir:
        return None
    wandb_dir = Path(out_dir) / "wandb"
    if not wandb_dir.is_dir():
        return None
    runs = sorted(
        (p for p in wandb_dir.glob("run-*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        return None
    name = runs[-1].name  # e.g. "run-20260512_091500-abc12def"
    # Trailing hyphen-separated chunk is the run ID.
    parts = name.rsplit("-", 1)
    if len(parts) != 2:
        return None
    return parts[1] or None


def main() -> int:
    model_path = os.environ.get("MODEL_PATH")
    project = os.environ.get("WANDB_PROJECT", "phase13-docker")
    artifact_name = os.environ.get("ARTIFACT_NAME", "phase13_best_model")
    run_name = os.environ.get("RUN_NAME", "phase13-docker-run")
    out_dir = os.environ.get("OUT_DIR", "")
    train_log = os.environ.get("TRAIN_LOG", "")
    train_n = os.environ.get("TRAIN_N", "unknown")
    seed = os.environ.get("SEED", "unknown")

    if not model_path or not Path(model_path).is_file():
        print(f"ERROR: MODEL_PATH not set or missing: {model_path!r}",
              file=sys.stderr)
        return 1
    if not os.environ.get("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY not set — cannot upload artifact.",
              file=sys.stderr)
        return 1

    summary = _parse_summary(train_log)
    if summary:
        print(f"  parsed final SUMMARY: best_f1={summary['best_f1']:.4f} "
              f"mae={summary['mae']:.2f} epochs={summary['epochs']}")
    else:
        print("  (no SUMMARY line found in training log)")

    # Try to resume the training run so the artifact lands on the same
    # wandb row as the per-epoch curves. Fall back to a sidecar run if
    # the ID can't be recovered.
    training_run_id = _find_training_run_id(out_dir)
    if training_run_id:
        print(f"  found training wandb run id: {training_run_id} — "
              f"resuming to attach artifact")
        run = wandb.init(
            project=project,
            id=training_run_id,
            resume="must",
        )
    else:
        sidecar_name = f"{run_name}_model"
        print(f"  no training wandb run id found under {out_dir!r} — "
              f"creating sidecar run '{sidecar_name}'")
        run = wandb.init(
            project=project,
            name=sidecar_name,
            tags=["phase13", "data_scaling", "docker", f"N{train_n}",
                  "model_artifact_sidecar"],
            notes=(
                f"Sidecar artifact upload for {run_name} (N={train_n}, "
                f"seed={seed}). Created because the training run's wandb "
                f"ID could not be recovered (training was probably offline)."
            ),
            reinit=True,
        )

    metadata = {
        "train_n": train_n,
        "seed": seed,
        "model_path_in_container": model_path,
        "model_file_size_mb": round(
            Path(model_path).stat().st_size / (1024 ** 2), 1
        ),
        **summary,
    }

    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=(
            f"HerdNet Phase-13 best_model.pth, N={train_n}, seed={seed}. "
            f"Warm-started from best_models/phase8/b4_seed42 and fine-tuned "
            f"on Phase-13 train_N{train_n}_s{seed} for {summary.get('epochs', '?')} epochs."
        ),
        metadata=metadata,
    )
    artifact.add_file(model_path)

    # Refresh run.summary with the final metrics (the per-epoch run does
    # this internally too, but doing it here ensures the artifact run has
    # the same headline numbers visible in the run header).
    if summary:
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                wandb.summary[k] = v
        wandb.summary["model_file_size_mb"] = metadata["model_file_size_mb"]

    print(f"  uploading {model_path} as wandb Artifact "
          f"'{artifact_name}' in project '{project}' ...")
    run.log_artifact(artifact)
    run.finish()
    print("  done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
