"""
Inference Hyperparameter Search for HerdNet Models

Runs inference ONCE per model (disk-cached), then sweeps LMDS
detection parameters to find optimal settings for three objectives:

1. DETECTION: Best F1 score (balanced precision/recall)
2. COUNTING: Lowest counting error (MAE) - minimize |predicted - actual| per image
3. SCREENING: High recall (≥90%) with best achievable precision (Eikelboom-style)

Usage:
    python tools/inference_hparam_search.py
"""

import itertools
import json
import warnings
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from loguru import logger

import animaloc
from animaloc.eval.lmds import HerdNetLMDS
from animaloc.eval.metrics import PointsMetrics

warnings.filterwarnings("ignore", category=UserWarning)

DATA_ROOT = Path(__file__).parent.parent
VAL_CSV = DATA_ROOT / "data_fmo03/val/herdnet_format_512_0_crops.csv"
VAL_DIR = DATA_ROOT / "data_fmo03/val/crops_512"

MODELS = {
    "ConvNeXt_clean": "best_models/fmo03_new_clean_convnext/best_model.pth",
    "ConvNeXt_full": "best_models/fmo03_new_full_convnext/best_model.pth",
    "ConvNeXt_objcrop_e15": "output/fmo03_new_objcrop_e15_convnext/2026-04-08/05-31-14/best_model.pth",
    "ConvNeXt_augplus": "best_models/fmo03_new_objcrop_augplus_convnext/best_model.pth",
    "ConvNeXt_finetune": "best_models/fmo03_new_objcrop_finetune_convnext/best_model.pth",
    "Hybrid_clean": "best_models/fmo03_new_clean_hybrid/best_model.pth",
    "DLA34_clean": "best_models/fmo03_new_clean_dla34/best_model.pth",
}

ADAPT_TS_VALUES = [0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7]
KERNEL_SIZES = [(3, 3), (5, 5), (9, 9)]
NEG_TS_VALUES = [0.05, 0.1]

# Screening target: find params where recall >= this threshold
SCREENING_RECALL_TARGET = 0.90


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def load_model_from_checkpoint(pth_path: str, device: torch.device):
    checkpoint = torch.load(pth_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    model_name = cfg["model"]["name"]
    model_kwargs = cfg["model"].get("kwargs", {})
    model_cls = animaloc.models.__dict__[model_name]
    model = model_cls(**model_kwargs)
    model.reshape_classes(cfg["datasets"]["num_classes"])
    state_dict = checkpoint["model_state_dict"]
    cleaned = {k.replace("model.", "", 1) if k.startswith("model.") else k: v
               for k, v in state_dict.items()}
    model.load_state_dict(cleaned, strict=False)
    return model.to(device).eval(), cfg


CACHE_DIR = DATA_ROOT / "output" / "hparam_search" / "cache"


def _model_hash(pth_path: str) -> str:
    import hashlib
    p = Path(pth_path)
    key = f"{p.name}_{p.stat().st_size}_{p.stat().st_mtime_ns}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def cache_model_outputs(model, dataloader, device, pth_path: str = ""):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{_model_hash(pth_path)}.pt"

    if cache_file.exists():
        logger.info(f"  Loading from disk cache: {cache_file.name}")
        return torch.load(cache_file, weights_only=False)

    cached = []
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                heatmap, clsmap = model(images)

            gt_coords = [p[::-1] for p in targets["points"].squeeze(0).tolist()]
            gt_labels = targets["labels"].squeeze(0).tolist()
            cached.append({
                "heatmap": heatmap.cpu(), "clsmap": clsmap.cpu(),
                "gt_coords": gt_coords, "gt_labels": gt_labels,
            })

    torch.save(cached, cache_file)
    logger.info(f"  Saved to disk cache: {cache_file.name}")
    return cached


def evaluate_cached(cached, adapt_ts, kernel_size, neg_ts, num_classes) -> Dict:
    """Evaluate with specific LMDS params. Returns detection + counting metrics."""
    sf = cached[0]["heatmap"].shape[2] // cached[0]["clsmap"].shape[2]
    lmds = HerdNetLMDS(
        kernel_size=kernel_size, adapt_ts=adapt_ts,
        neg_ts=neg_ts, up=True, scale_factor=sf,
    )
    metrics = PointsMetrics(radius=100, num_classes=num_classes)

    # Per-image counting for MAE/RMSE
    count_errors = []  # (gt_count, pred_count) per image

    for sample in cached:
        outputs = (sample["heatmap"], sample["clsmap"])
        counts, locs, labels, scores, dscores = lmds(outputs)

        gt = {"loc": sample["gt_coords"], "labels": sample["gt_labels"]}
        preds = {
            "loc": locs[0], "labels": labels[0],
            "scores": scores[0], "dscores": dscores[0],
        }

        # GT count for this image (class 1 only for binary)
        gt_count = len([l for l in sample["gt_labels"] if l == 1])
        pred_count = sum(counts[0]) if isinstance(counts[0], list) else counts[0]
        count_errors.append((gt_count, pred_count))

        if len(preds["loc"]) > 0 and len(gt["loc"]) > 0:
            metrics.feed(gt=gt, preds=preds, est_count=counts[0])
        elif len(gt["loc"]) > 0:
            for c in range(1, num_classes):
                n_gt = len([l for l in sample["gt_labels"] if l == c])
                metrics.fn[c - 1] += n_gt
            # Still feed counting error
            metrics._update_errors(
                [len([l for l in sample["gt_labels"] if l == c]) for c in range(1, num_classes)],
                counts[0] if isinstance(counts[0], list) else [counts[0]]
            )
            metrics._total_calls += 1
        elif len(preds["loc"]) > 0:
            # False positives, no GT
            metrics._update_errors([0], counts[0] if isinstance(counts[0], list) else [counts[0]])
            metrics._total_calls += 1

    # Compute counting metrics
    import math
    gt_total = sum(g for g, p in count_errors)
    pred_total = sum(p for g, p in count_errors)
    abs_errors = [abs(g - p) for g, p in count_errors]
    sq_errors = [(g - p) ** 2 for g, p in count_errors]
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else 0
    rmse = math.sqrt(sum(sq_errors) / len(sq_errors)) if sq_errors else 0
    count_bias = pred_total - gt_total  # positive = overcounting

    tp = sum(metrics.tp)
    fp = sum(metrics.fp)
    fn = sum(metrics.fn)

    return {
        "f1": metrics.fbeta_score(c=1, beta=1),
        "recall": metrics.recall(),
        "precision": metrics.precision(),
        "tp": tp, "fp": fp, "fn": fn,
        "mae": mae,
        "rmse": rmse,
        "gt_total": gt_total,
        "pred_total": pred_total,
        "count_bias": count_bias,
        "count_error_pct": abs(count_bias) / gt_total * 100 if gt_total > 0 else 0,
    }


def main():
    logger.info("=" * 80)
    logger.info("Inference Hyperparameter Search — Detection / Counting / Screening")
    logger.info("=" * 80)

    device = get_device()
    logger.info(f"Device: {device}")

    val_df = pd.read_csv(str(VAL_CSV))
    albu_transforms = [__import__("albumentations").Normalize()]

    all_results = []
    best_detection = {}   # Best F1
    best_counting = {}    # Lowest MAE
    best_screening = {}   # Best precision at recall >= 90%

    for model_name, pth_rel in MODELS.items():
        pth_path = str(DATA_ROOT / pth_rel)
        if not Path(pth_path).exists():
            logger.warning(f"Skipping {model_name}: not found")
            continue

        logger.info(f"\n{'─' * 60}")
        logger.info(f"Model: {model_name}")

        model, cfg = load_model_from_checkpoint(pth_path, device)
        num_classes = cfg["datasets"]["num_classes"]
        down_ratio = cfg["model"]["kwargs"].get("down_ratio", 4)

        end_transforms = animaloc.data.transforms.DownSample(
            down_ratio=down_ratio, anno_type="point"
        )
        val_dataset = animaloc.datasets.CSVDataset(
            csv_file=val_df, root_dir=str(VAL_DIR),
            albu_transforms=albu_transforms, end_transforms=[end_transforms],
        )
        dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False)

        logger.info(f"  Caching {len(dataloader)} outputs (down_ratio={down_ratio})...")
        cached = cache_model_outputs(model, dataloader, device, pth_path=pth_path)
        logger.info(f"  Cached. Heatmap: {cached[0]['heatmap'].shape}")
        del model
        torch.cuda.empty_cache()

        # Track best per objective
        det_best_f1 = 0
        det_best = {}
        cnt_best_mae = float("inf")
        cnt_best = {}
        scr_best_prec = 0
        scr_best = {}

        for adapt_ts, kernel_size, neg_ts in itertools.product(
            ADAPT_TS_VALUES, KERNEL_SIZES, NEG_TS_VALUES
        ):
            r = evaluate_cached(cached, adapt_ts, kernel_size, neg_ts, num_classes)
            row = {
                "model": model_name,
                "loss": "fidt" if "fidt" in model_name else "focal",
                "adapt_ts": adapt_ts,
                "kernel": f"{kernel_size[0]}x{kernel_size[1]}",
                "neg_ts": neg_ts,
                **r,
            }
            all_results.append(row)

            # 1. Detection: best F1
            if r["f1"] > det_best_f1:
                det_best_f1 = r["f1"]
                det_best = {"adapt_ts": adapt_ts, "kernel": kernel_size, "neg_ts": neg_ts, **r}

            # 2. Counting: lowest MAE
            if r["mae"] < cnt_best_mae:
                cnt_best_mae = r["mae"]
                cnt_best = {"adapt_ts": adapt_ts, "kernel": kernel_size, "neg_ts": neg_ts, **r}

            # 3. Screening: best precision where recall >= target
            if r["recall"] >= SCREENING_RECALL_TARGET and r["precision"] > scr_best_prec:
                scr_best_prec = r["precision"]
                scr_best = {"adapt_ts": adapt_ts, "kernel": kernel_size, "neg_ts": neg_ts, **r}

        best_detection[model_name] = det_best
        best_counting[model_name] = cnt_best
        best_screening[model_name] = scr_best if scr_best else {"note": "no config achieves 90% recall"}

        logger.info(f"  DETECTION: F1={det_best.get('f1', 0):.4f} "
                     f"(ts={det_best.get('adapt_ts')}, ks={det_best.get('kernel')}) "
                     f"P={det_best.get('precision', 0):.3f} R={det_best.get('recall', 0):.3f}")
        logger.info(f"  COUNTING:  MAE={cnt_best.get('mae', 0):.3f} "
                     f"(ts={cnt_best.get('adapt_ts')}, ks={cnt_best.get('kernel')}) "
                     f"bias={cnt_best.get('count_bias', 0):+.0f} err={cnt_best.get('count_error_pct', 0):.1f}%")
        if scr_best:
            logger.info(f"  SCREENING: R={scr_best.get('recall', 0):.3f} P={scr_best.get('precision', 0):.3f} "
                         f"(ts={scr_best.get('adapt_ts')}, ks={scr_best.get('kernel')}) "
                         f"TP={scr_best.get('tp', 0):.0f} FP={scr_best.get('fp', 0):.0f}")
        else:
            logger.info(f"  SCREENING: No config achieves {SCREENING_RECALL_TARGET*100:.0f}% recall")

        del cached

    # Save CSV
    output_dir = DATA_ROOT / "output" / "hparam_search"
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    csv_path = output_dir / "inference_hparam_results.csv"
    df.to_csv(csv_path, index=False)

    # ═══════════════════════════════════════════════════════════════════
    # SUMMARY TABLES
    # ═══════════════════════════════════════════════════════════════════

    def _header(title):
        logger.info("\n" + "=" * 100)
        logger.info(title)
        logger.info("=" * 100)

    # ── 1. DETECTION (Best F1) ──
    _header("OBJECTIVE 1: DETECTION — Best F1 Score")
    logger.info(f"{'Model':<22} {'Loss':<7} {'F1':>6} {'Prec':>6} {'Rec':>6} "
                f"{'ts':>5} {'ks':>5} {'neg':>5} {'TP':>5} {'FP':>5} {'FN':>5}")
    logger.info("─" * 90)
    for name, p in best_detection.items():
        if not p: continue
        loss = "fidt" if "fidt" in name else "focal"
        logger.info(f"{name:<22} {loss:<7} {p['f1']:6.3f} {p['precision']:6.3f} {p['recall']:6.3f} "
                     f"{p['adapt_ts']:5.2f} {str(p['kernel']):>5} {p['neg_ts']:5.2f} "
                     f"{p['tp']:5.0f} {p['fp']:5.0f} {p['fn']:5.0f}")

    # ── 2. COUNTING (Lowest MAE) ──
    _header("OBJECTIVE 2: COUNTING — Lowest Mean Absolute Error")
    logger.info(f"{'Model':<22} {'Loss':<7} {'MAE':>6} {'RMSE':>6} {'Bias':>6} {'Err%':>6} "
                f"{'GT':>6} {'Pred':>6} {'ts':>5} {'ks':>5}")
    logger.info("─" * 90)
    for name, p in best_counting.items():
        if not p: continue
        loss = "fidt" if "fidt" in name else "focal"
        logger.info(f"{name:<22} {loss:<7} {p['mae']:6.3f} {p['rmse']:6.3f} "
                     f"{p['count_bias']:+6.0f} {p['count_error_pct']:5.1f}% "
                     f"{p['gt_total']:6.0f} {p['pred_total']:6.0f} "
                     f"{p['adapt_ts']:5.2f} {str(p['kernel']):>5}")

    # ── 3. SCREENING (Recall ≥ 90%, best precision) ──
    _header(f"OBJECTIVE 3: SCREENING — Best Precision at Recall ≥ {SCREENING_RECALL_TARGET*100:.0f}%")
    logger.info(f"{'Model':<22} {'Loss':<7} {'Rec':>6} {'Prec':>6} {'F1':>6} "
                f"{'ts':>5} {'ks':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'FP/img':>7}")
    logger.info("─" * 100)
    n_images = len(pd.read_csv(str(VAL_CSV))["images"].unique())
    for name, p in best_screening.items():
        if "note" in p:
            logger.info(f"{name:<22} {'—':<7} {'N/A':>6}  {p['note']}")
            continue
        loss = "fidt" if "fidt" in name else "focal"
        fp_per_img = p['fp'] / n_images if n_images > 0 else 0
        logger.info(f"{name:<22} {loss:<7} {p['recall']:6.3f} {p['precision']:6.3f} {p['f1']:6.3f} "
                     f"{p['adapt_ts']:5.2f} {str(p['kernel']):>5} "
                     f"{p['tp']:5.0f} {p['fp']:5.0f} {p['fn']:5.0f} {fp_per_img:7.1f}")

    # ── Cross-objective comparison ──
    _header("CROSS-OBJECTIVE: Same model, different inference goals")
    for arch in ["DLA34", "ConvNeXt", "Hybrid"]:
        for loss in ["focal", "fidt"]:
            name = f"{arch}_{loss}"
            if name not in best_detection: continue
            d = best_detection.get(name, {})
            c = best_counting.get(name, {})
            s = best_screening.get(name, {})
            if not d: continue
            logger.info(f"\n  {name}:")
            logger.info(f"    Detection:  F1={d.get('f1',0):.3f}  (ts={d.get('adapt_ts')})")
            logger.info(f"    Counting:   MAE={c.get('mae',0):.3f} bias={c.get('count_bias',0):+.0f}  (ts={c.get('adapt_ts')})")
            if "note" not in s:
                logger.info(f"    Screening:  R={s.get('recall',0):.3f} P={s.get('precision',0):.3f}  (ts={s.get('adapt_ts')})")
            else:
                logger.info(f"    Screening:  {s['note']}")

    # Save JSON
    summary = {
        "detection": {n: {k: str(v) if isinstance(v, tuple) else v for k, v in p.items()}
                      for n, p in best_detection.items()},
        "counting": {n: {k: str(v) if isinstance(v, tuple) else v for k, v in p.items()}
                     for n, p in best_counting.items()},
        "screening": {n: {k: str(v) if isinstance(v, tuple) else v for k, v in p.items()}
                      for n, p in best_screening.items()},
    }
    with open(output_dir / "best_params.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nResults: {csv_path}")
    logger.info(f"Best params: {output_dir / 'best_params.json'}")


if __name__ == "__main__":
    main()
