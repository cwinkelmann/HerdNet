"""
Generate diagnostic plots for the production stack.

Reads the Phase-6 (B4×3 same-arch) and Phase-8 (B3×3+B4×3 cross-arch)
sweep CSVs, plus the per-image detections at the best operating points,
and produces:

  - pr_curve.png         : precision–recall curves across the full
                           threshold sweep, both ensembles overlaid
  - mae_vs_threshold.png : MAE as adapt_ts is varied
  - me_vs_threshold.png  : signed Mean Error (predicted_total - gt_total)
                           per image as adapt_ts is varied
  - error_pct.png        : same MAE and ME expressed as % of mean per-image
                           GT count (i.e. "avg per-frame counting error
                           in % of true count")
  - per_image_error.png  : per-frame predicted vs GT count and signed
                           bias, for the production best-MAE config

Saves figures to docs/benchmarks/assets/plots/.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/home/christian/hnee/HerdNet")
GT_CSV = ROOT / "data_fmo03/val/herdnet_format.csv"
OUT_DIR = ROOT / "docs/benchmarks/assets/plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Phase 6 sweep CSVs (B4×3 same-architecture ensemble).
P6_NO_TTA = ROOT / "output/phase6_sweep_20260507_124034/T2_ensemble_no_tta.csv"
P6_TTA    = ROOT / "output/phase6_sweep_20260507_124034/T3_ensemble_tta.csv"

# Phase 8 sweep CSVs (B3×3 + B4×3 cross-architecture ensemble).
P8_NO_TTA = ROOT / "output/phase8_sweep_20260507_215325/T1_cross_arch_no_tta.csv"
P8_TTA    = ROOT / "output/phase8_sweep_20260507_215325/T2_cross_arch_tta.csv"

# Detections at the production best-MAE config (Phase-8 8a, ts=0.25).
P8_BEST_MAE_DETS = ROOT / "output/phase8_sweep_20260507_215325/ENS6_CROSS/ts_0.25_ttaFalse/detections.csv"


def _load(csv_path):
    df = pd.read_csv(csv_path)
    df["adapt_ts"] = df["adapt_ts"].astype(float)
    df = df.sort_values("adapt_ts").reset_index(drop=True)
    return df


def plot_pr():
    p6n = _load(P6_NO_TTA)
    p6t = _load(P6_TTA)
    p8n = _load(P8_NO_TTA)
    p8t = _load(P8_TTA)

    fig, ax = plt.subplots(figsize=(7, 6))
    series = [
        ("Phase 6 — B4×3 (same-arch), no TTA",     p6n, "tab:gray", "o"),
        ("Phase 6 — B4×3 + TTA",                    p6t, "gray",     "s"),
        ("Phase 8 — B3×3+B4×3 (cross-arch), no TTA", p8n, "tab:blue", "o"),
        ("Phase 8 — B3×3+B4×3 + TTA",               p8t, "tab:red",  "s"),
    ]
    for label, df, color, marker in series:
        ax.plot(df["recall"], df["precision"], marker=marker, color=color, label=label, lw=1.6, ms=6)
        # Annotate selected operating points
        for _, row in df.iterrows():
            ts = row["adapt_ts"]
            if ts in (0.25, 0.30, 0.35, 0.50, 0.70):
                ax.annotate(f"{ts:.2f}", (row["recall"], row["precision"]),
                            xytext=(4, 4), textcoords="offset points",
                            fontsize=7, color=color, alpha=0.7)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("PR curves — Phase 6 vs Phase 8 ensembles\n(annotated points are adapt_ts values)")
    ax.set_xlim(0.85, 1.005)
    ax.set_ylim(0.85, 1.005)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "pr_curve.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_mae_me():
    series = [
        ("Phase 6 B4×3, no TTA",      _load(P6_NO_TTA), "tab:gray"),
        ("Phase 6 B4×3 + TTA",         _load(P6_TTA),    "black"),
        ("Phase 8 B3×3+B4×3, no TTA", _load(P8_NO_TTA), "tab:blue"),
        ("Phase 8 B3×3+B4×3 + TTA",   _load(P8_TTA),    "tab:red"),
    ]

    # GT count for percentage axis (mean per-frame GT).
    gt = pd.read_csv(GT_CSV)
    gt_per_frame = gt.groupby("images").size().mean()
    print(f"  mean GT iguanas per frame: {gt_per_frame:.2f}")

    # --- MAE absolute ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, df, color in series:
        ax.plot(df["adapt_ts"], df["mae"], marker="o", color=color, label=label, lw=1.6)
    ax.set_xlabel("LMDS threshold (adapt_ts)")
    ax.set_ylabel("MAE  (iguanas / frame)")
    ax.set_title("Counting error vs threshold\n(MAE = mean absolute per-frame count error)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "mae_vs_threshold.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")

    # --- ME (signed) ---
    # Sweep CSVs don't carry signed ME, so compute per-eval from
    # detections.csv where available; fallback to deriving from
    # mae and direction.
    # The metrics_results.csv files in each ts subfolder DO carry 'me'.
    # For now, recover a signed ME per (config, ts) by reading the per-ts
    # metrics_results.csv files.
    def collect_me(sweep_csv: Path, base: Path):
        df = _load(sweep_csv)
        out = []
        for _, row in df.iterrows():
            ts = row["adapt_ts"]
            ts_str = f"{ts:.2f}"
            # Try both potential patterns for the per-ts subdir:
            tta_str = "True" if str(row.get("tta", "False")) == "True" else "False"
            cands = list(base.glob(f"*/ts_{ts_str}_tta{tta_str}/metrics_results.csv")) \
                  + list(base.glob(f"*/ts_{ts_str}/metrics_results.csv"))
            me_val = None
            for c in cands:
                try:
                    m = pd.read_csv(c)
                    me_val = float(m[m["class"] == "binary"]["me"].iloc[0])
                    break
                except Exception:
                    continue
            out.append({"adapt_ts": ts, "me": me_val,
                        "mae": row["mae"], "tta": tta_str})
        return pd.DataFrame(out)

    # Phase 6 / Phase 8 sweep base dirs
    P6_BASE = ROOT / "output/phase6_sweep_20260507_124034"
    P8_BASE = ROOT / "output/phase8_sweep_20260507_215325"

    me_series = [
        ("Phase 6 B4×3, no TTA",      collect_me(P6_NO_TTA, P6_BASE), "tab:gray"),
        ("Phase 6 B4×3 + TTA",         collect_me(P6_TTA,    P6_BASE), "black"),
        ("Phase 8 B3×3+B4×3, no TTA", collect_me(P8_NO_TTA, P8_BASE), "tab:blue"),
        ("Phase 8 B3×3+B4×3 + TTA",   collect_me(P8_TTA,    P8_BASE), "tab:red"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, df, color in me_series:
        df_ok = df.dropna(subset=["me"])
        ax.plot(df_ok["adapt_ts"], df_ok["me"], marker="o", color=color, label=label, lw=1.6)
    ax.axhline(0, color="black", lw=0.7, alpha=0.5)
    ax.set_xlabel("LMDS threshold (adapt_ts)")
    ax.set_ylabel("Mean Error  (signed; iguanas / frame)\n+ = over-count    − = under-count")
    ax.set_title("Counting bias vs threshold\n(ME = mean signed per-frame count error)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    out = OUT_DIR / "me_vs_threshold.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")

    # --- Percentage version: MAE% and ME% relative to per-frame GT count ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for label, df, color in series:
        ax1.plot(df["adapt_ts"], 100 * df["mae"] / gt_per_frame, marker="o",
                 color=color, label=label, lw=1.6)
    ax1.set_xlabel("LMDS threshold (adapt_ts)")
    ax1.set_ylabel("MAE  (% of mean per-frame GT count)")
    ax1.set_title(f"Counting error % vs threshold\n(absolute, normalized by GT={gt_per_frame:.1f}/frame)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    for label, df, color in me_series:
        df_ok = df.dropna(subset=["me"])
        ax2.plot(df_ok["adapt_ts"], 100 * df_ok["me"] / gt_per_frame, marker="o",
                 color=color, label=label, lw=1.6)
    ax2.axhline(0, color="black", lw=0.7, alpha=0.5)
    ax2.set_xlabel("LMDS threshold (adapt_ts)")
    ax2.set_ylabel("ME  (% of mean per-frame GT count;  + over,  − under)")
    ax2.set_title("Counting bias % vs threshold")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    out = OUT_DIR / "error_pct.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")


def plot_per_image():
    """Per-frame predicted vs GT count and signed bias for the
    production best-MAE config (Phase 8a, ts=0.25)."""
    if not P8_BEST_MAE_DETS.exists():
        print(f"  skip per-image: missing {P8_BEST_MAE_DETS}")
        return

    gt = pd.read_csv(GT_CSV)
    det = pd.read_csv(P8_BEST_MAE_DETS)

    # Per-image counts. GT: count of rows per image. Detections: count of
    # detections per image (each row = one detection).
    g = gt[gt["labels"] == 1].groupby("images").size().rename("gt_count")
    d = det.groupby("images").size().rename("pred_count")
    df = pd.concat([g, d], axis=1).fillna(0).astype(int)
    df["bias"] = df["pred_count"] - df["gt_count"]
    df["abs_err"] = df["bias"].abs()
    df["bias_pct"] = 100 * df["bias"] / df["gt_count"].replace(0, np.nan)
    df = df.sort_index()
    print(df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(df))
    w = 0.4
    ax1.bar(x - w/2, df["gt_count"], width=w, color="tab:green", label="GT")
    ax1.bar(x + w/2, df["pred_count"], width=w, color="tab:blue", label="Predicted")
    ax1.set_xticks(x)
    ax1.set_xticklabels([n.replace(".JPG", "").replace("FMO05___DJI_", "") for n in df.index],
                        rotation=45, ha="right")
    ax1.set_ylabel("count (iguanas)")
    ax1.set_title("Per-frame counts — GT vs predicted\n"
                  "(B3×3+B4×3 cross-arch ensemble, adapt_ts=0.25, no TTA)")
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.legend()

    colors = ["tab:red" if b > 0 else "tab:purple" if b < 0 else "tab:gray"
              for b in df["bias"]]
    ax2.bar(x, df["bias"], color=colors)
    ax2.axhline(0, color="black", lw=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels([n.replace(".JPG", "").replace("FMO05___DJI_", "") for n in df.index],
                        rotation=45, ha="right")
    ax2.set_ylabel("Signed error  (predicted − GT)")
    ax2.set_title(f"Per-frame counting bias\n"
                  f"sum(bias) = {int(df['bias'].sum())}, "
                  f"mean|bias| = {df['abs_err'].mean():.2f},  "
                  f"mean(bias) = {df['bias'].mean():+.2f}")
    ax2.grid(True, axis="y", alpha=0.3)

    # annotate each bar with the signed value
    for xi, b in zip(x, df["bias"]):
        ax2.text(xi, b + (0.05 * np.sign(b) if b != 0 else 0), f"{b:+d}",
                 ha="center", va="bottom" if b >= 0 else "top", fontsize=9)

    fig.tight_layout()
    out = OUT_DIR / "per_image_error.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  wrote {out}")

    # also save the per-frame table
    out_csv = OUT_DIR / "per_image_error.csv"
    df.to_csv(out_csv)
    print(f"  wrote {out_csv}")


if __name__ == "__main__":
    print("Generating PR curve...")
    plot_pr()
    print("Generating MAE & ME vs threshold...")
    plot_mae_me()
    print("Generating per-image error...")
    plot_per_image()
    print("Done.")
