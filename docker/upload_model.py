"""
Post-training: open one wandb run for this training, log the final
training-summary metrics, and attach best_model.pth as an artifact.

This is the ONLY wandb push for the whole training. Training itself runs
with wandb_flag=False (no per-epoch chatter) — see train_entrypoint.sh.
You get exactly one row per training in the wandb UI, with the final
metrics in `run.summary` and the model attached as an Artifact.

Reads everything from environment variables — meant to be invoked from
train_entrypoint.sh after training succeeds.

Required env vars:
  WANDB_PROJECT, WANDB_API_KEY, ARTIFACT_NAME, RUN_NAME, MODEL_PATH

Optional env vars:
  TRAIN_LOG  : path to the training log to extract the final SUMMARY line
  TRAIN_N    : training-set size tag (recorded as artifact metadata)
  SEED       : training seed (artifact metadata)
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import wandb


def _parse_summary(log_path: str) -> dict:
    """Extract the final SUMMARY line emitted by animaloc.utils.train (line ~503).

    Looks for: best_f1=, best_f2=, recall=, precision=, mae=, rmse=, best_val=,
               epochs=, model=
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
        # SUMMARY line is near the end of the log
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


def main() -> int:
    model_path = os.environ.get("MODEL_PATH")
    project = os.environ.get("WANDB_PROJECT", "phase13-docker")
    artifact_name = os.environ.get("ARTIFACT_NAME", "phase13_best_model")
    run_name = os.environ.get("RUN_NAME", "phase13-docker-run")
    train_log = os.environ.get("TRAIN_LOG", "")
    train_n = os.environ.get("TRAIN_N", "unknown")
    seed = os.environ.get("SEED", "unknown")

    if not model_path or not Path(model_path).is_file():
        print(f"ERROR: MODEL_PATH not set or missing: {model_path!r}", file=sys.stderr)
        return 1

    # WANDB_API_KEY check (let wandb itself complain if missing).
    if not os.environ.get("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY not set — cannot upload artifact.",
              file=sys.stderr)
        return 1

    summary = _parse_summary(train_log)
    if summary:
        print(f"  parsed final SUMMARY: best_f1={summary.get('best_f1'):.4f} "
              f"mae={summary.get('mae'):.2f} epochs={summary.get('epochs')}")
    else:
        print("  (no SUMMARY line found in training log)")

    # Open the single wandb run for this whole training. Training itself
    # ran with wandb_flag=False, so this run is fresh and contains only
    # the final summary metrics + the model artifact (no per-epoch
    # history). Exactly one row per training in the wandb UI.
    run = wandb.init(
        project=project,
        name=run_name,
        tags=["phase13", "data_scaling", "docker", f"N{train_n}"],
        notes=(
            f"Phase-13 training, N={train_n}, seed={seed}. "
            f"Warm-started from best_models/phase8/b4_seed42. "
            f"Training metrics were not streamed per-epoch — the final "
            f"summary + model artifact are submitted here at the end."
        ),
        reinit=True,
    )

    metadata = {
        "train_n": train_n,
        "seed": seed,
        "model_path_in_container": model_path,
        "model_file_size_mb": round(Path(model_path).stat().st_size / (1024 ** 2), 1),
        **summary,
    }

    artifact = wandb.Artifact(
        name=artifact_name,
        type="model",
        description=(
            f"HerdNet Phase-13 best_model.pth, N={train_n}, seed={seed}. "
            f"Warm-started from best_models/phase8/b4_seed42 and fine-tuned on "
            f"the Phase-13 train_N{train_n}_s{seed} split."
        ),
        metadata=metadata,
    )
    artifact.add_file(model_path)

    # Also log the final summary as run metrics so the artifact run has
    # context in the wandb dashboard.
    if summary:
        for k, v in summary.items():
            if isinstance(v, (int, float)):
                wandb.summary[k] = v
        wandb.summary["model_file_size_mb"] = metadata["model_file_size_mb"]

    print(f"  uploading {model_path} as wandb artifact "
          f"'{artifact_name}' in project '{project}' ...")
    run.log_artifact(artifact)
    run.finish()
    print("  done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
