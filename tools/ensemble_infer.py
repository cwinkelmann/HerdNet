#!/usr/bin/env python
"""
Ensemble inference for HerdNet.

Loads N model checkpoints (same architecture, different training seeds),
averages their pre-LMDS heatmap+classification outputs per tile, then runs
the standard HerdNetStitcher / LMDS / metrics pipeline.

Usage:
    python tools/ensemble_infer.py \
        --config-dir configs/demo \
        --config-name fmo03_full_v2_bifpn \
        --images data_fmo03/val/Default \
        --models <pth1> <pth2> <pth3> \
        --evaluate \
        --overrides "wandb_flag=False" \
                    "training_settings.evaluator.kwargs.lmds_kwargs.adapt_ts=0.35"
"""

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

import animaloc.models  # registers model classes
import animaloc.utils.inference as inf_mod
from animaloc.models.utils import LossWrapper


class EnsembleHerdNet(nn.Module):
    """Average pre-LMDS heatmap + classification outputs across N models.

    All models must share the same architecture and produce the same
    (heatmap, cls_out) tuple from `forward(x)`. Aux outputs (P3/P4) are
    ignored — they're training-only on V2/V3 backbones.
    """

    def __init__(self, models: List[nn.Module]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x: torch.Tensor):
        # Each member returns (heatmap, cls_out) or longer tuple for V2/V3.
        outs = [m(x) for m in self.models]
        h = torch.stack([o[0] for o in outs]).mean(dim=0)
        c = torch.stack([o[1] for o in outs]).mean(dim=0)
        return h, c

    def reshape_classes(self, num_classes: int) -> None:
        for m in self.models:
            if hasattr(m, "reshape_classes"):
                m.reshape_classes(num_classes)


def _build_ensemble_factory(checkpoint_paths: List[str]):
    """Returns a `_build_model` replacement that ignores cfg.model.load_from
    and instead loads the supplied checkpoint paths into an EnsembleHerdNet.

    Cross-architecture support: if a checkpoint contains its own ``config``
    entry (HerdNet checkpoints saved via the codebase do), each member is
    instantiated from its own config — so B3 (HerdNetConvNeXt) and B4
    (CamouflageHerdNetConvNeXtV2) checkpoints can be ensembled together
    even though their classes differ. Otherwise, the runtime cfg is used
    as a fallback (single-architecture mode).
    """

    def _build_member_from_ckpt(pth, fallback_cfg):
        ckpt = torch.load(pth, map_location="cpu", weights_only=False)
        ckpt_cfg = ckpt.get("config")
        if ckpt_cfg is not None and "model" in ckpt_cfg:
            name = ckpt_cfg["model"]["name"]
            mkwargs = dict(ckpt_cfg["model"].get("kwargs", {}) or {})
            num_classes = ckpt_cfg.get("datasets", {}).get(
                "num_classes", fallback_cfg.datasets.num_classes
            )
        else:
            name = fallback_cfg.model.name
            mkwargs = dict(fallback_cfg.model.kwargs)
            num_classes = fallback_cfg.datasets.num_classes
        if name not in animaloc.models.__dict__:
            raise KeyError(f"Model class '{name}' not registered in animaloc.models")
        mkwargs.pop("num_classes", None)
        member = animaloc.models.__dict__[name](**mkwargs, num_classes=num_classes)

        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        # Some HerdNet checkpoints save the LossWrapper, so keys look like
        # 'model.<...>'. Strip a single leading 'model.' if every key has it.
        if all(k.startswith("model.") for k in sd.keys()):
            sd = {k[len("model.") :]: v for k, v in sd.items()}
        missing, unexpected = member.load_state_dict(sd, strict=False)
        if missing:
            print(f"    [warn] missing keys: {len(missing)} (first: {missing[0]})")
        if unexpected:
            print(f"    [warn] unexpected keys: {len(unexpected)} (first: {unexpected[0]})")
        return member, name

    def _build_ensemble(cfg):
        members = []
        seen_archs = []
        for pth in checkpoint_paths:
            print(f"  loading ensemble member: {pth}")
            member, name = _build_member_from_ckpt(pth, cfg)
            print(f"    arch: {name}")
            members.append(member)
            seen_archs.append(name)

        if len(set(seen_archs)) > 1:
            print(f"  cross-architecture ensemble: {seen_archs}")

        ensemble = EnsembleHerdNet(members)
        # Wrap in LossWrapper to match the rest of the inference plumbing.
        return LossWrapper(ensemble, [])

    return _build_ensemble


def parse_args():
    p = argparse.ArgumentParser(prog="ensemble_infer")
    p.add_argument("--config-dir", type=str, required=True)
    p.add_argument("--config-name", type=str, default="config")
    p.add_argument("--images", type=str, required=True)
    p.add_argument("--models", type=str, nargs="+", required=True,
                   help="Two or more best_model.pth paths to average.")
    p.add_argument("--vis", action="store_true")
    p.add_argument("--evaluate", action="store_true")
    p.add_argument("--overrides", nargs="*", default=[])
    return p.parse_args()


def main():
    args = parse_args()

    if len(args.models) < 2:
        raise ValueError("Ensemble needs at least 2 checkpoints.")

    GlobalHydra.instance().clear()
    config_dir = str(Path(args.config_dir).resolve())

    overrides = list(args.overrides)
    overrides.append(f"datasets.test.root_dir={args.images}")
    # cfg.model.load_from is unused once we monkey-patch _build_model, but
    # set it so other code that prints it (e.g. wandb metadata) doesn't crash.
    overrides.append(f"model.load_from={args.models[0]}")

    with initialize_config_dir(config_dir=config_dir, version_base="1.1"):
        cfg = compose(config_name=args.config_name, overrides=overrides)

    # Replace _build_model with the ensemble builder. Restore on exit so a
    # later import in the same Python process isn't affected.
    original_builder = inf_mod._build_model
    inf_mod._build_model = _build_ensemble_factory(args.models)
    try:
        inf_mod.inference(
            cfg,
            plain_inference=not args.evaluate,
            vis_detections=args.vis,
        )
    finally:
        inf_mod._build_model = original_builder


if __name__ == "__main__":
    main()
