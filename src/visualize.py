"""Report figures: accuracy-vs-SNR curves, per-class AP bars, image grids.

Reads results/metrics/summary_long.csv (from tables.py) and the per-(task,variant)
metric JSONs. Saves PNGs under results/figures/.

Usage:
    python -m src.visualize --config configs/config.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from src.config import load_config, variant_image_dir, variant_tag


def _figdir(cfg: Dict) -> Path:
    d = Path(cfg["paths"]["figures_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_acc_vs_snr(cfg: Dict) -> None:
    """One figure per task: metric vs SNR, one line per distortion, distorted vs enhanced."""
    long_path = Path(cfg["paths"]["metrics_dir"]) / "summary_long.csv"
    if not long_path.exists():
        print("[viz] no summary_long.csv; run tables first")
        return
    df = pd.read_csv(long_path)
    figdir = _figdir(cfg)

    for task in df["task"].unique():
        sub = df[df["task"] == task]
        clean_val = sub[sub["variant"] == "clean"]["value"]
        plt.figure(figsize=(9, 5))
        for dist in sorted(sub["distortion"].dropna().unique()):
            for variant, style in [("distorted", "-o"), ("enhanced", "--s"), ("finetuned", ":^")]:
                d = sub[(sub["distortion"] == dist) & (sub["variant"] == variant)]
                d = d.dropna(subset=["snr_db"]).sort_values("snr_db")
                if d.empty:
                    continue
                plt.plot(d["snr_db"], d["value"], style, label=f"{dist} ({variant})")
        if not clean_val.empty:
            plt.axhline(float(clean_val.iloc[0]), color="k", ls="--", alpha=0.6,
                        label=f"clean baseline = {float(clean_val.iloc[0]):.3f}")
        plt.gca().invert_xaxis()   # lower SNR = more distortion, to the right
        plt.xlabel("SNR (dB)  <- more distortion")
        plt.ylabel(sub["metric"].iloc[0])
        plt.title(f"{task}: performance vs SNR")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3)
        out = figdir / f"acc_vs_snr_{task}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"[viz] {out}")


def plot_per_class_ap(cfg: Dict) -> None:
    """Per-class AP bar chart for the clean detection baseline (top/bottom classes)."""
    mp = Path(cfg["paths"]["metrics_dir"]) / "detection__clean.json"
    if not mp.exists():
        return
    with open(mp) as f:
        res = json.load(f)
    pcl = res.get("per_class_ap")
    if not pcl:
        return
    items = sorted(((v, k) for k, v in pcl.items() if not np.isnan(v)), reverse=True)
    vals = [v for v, _ in items]
    names = [k for _, k in items]
    plt.figure(figsize=(14, 5))
    plt.bar(range(len(vals)), vals)
    plt.axhline(res["mAP"], color="r", ls="--", label=f"mAP = {res['mAP']:.3f}")
    plt.xticks(range(len(names)), names, rotation=90, fontsize=6)
    plt.ylabel("AP@[.5:.95]")
    plt.title("Per-class AP — clean detection baseline")
    plt.legend()
    out = _figdir(cfg) / "per_class_ap_clean.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[viz] {out}")


def plot_per_class_comparison(cfg: Dict, top_n: int = 15) -> None:
    """Grouped per-class AP bars: clean vs distorted vs enhanced vs finetuned,
    on the fine-tune distortion/severity (worst-case cell)."""
    ft = cfg.get("finetune", {})
    dtype, sev = ft.get("distortion"), ft.get("severity")
    if not dtype:
        return
    mdir = Path(cfg["paths"]["metrics_dir"])
    variants = {
        v: mdir / f"detection__{variant_tag(v, dtype, sev)}.json"
        for v in ("clean", "distorted", "enhanced", "finetuned")
    }
    per_class = {}
    for name, p in variants.items():
        if p.exists():
            with open(p) as f:
                per_class[name] = json.load(f).get("per_class_ap") or {}
    if "clean" not in per_class or len(per_class) < 2:
        return

    top = sorted(((v, k) for k, v in per_class["clean"].items() if not np.isnan(v)),
                 reverse=True)[:top_n]
    names = [k for _, k in top]
    x = np.arange(len(names))
    group_w = 0.8
    width = group_w / len(per_class)
    plt.figure(figsize=(14, 5))
    for i, (variant, pcl) in enumerate(per_class.items()):
        vals = [pcl.get(n, np.nan) for n in names]
        plt.bar(x + i * width, vals, width, label=variant)
    plt.xticks(x + (group_w - width) / 2, names, rotation=45, ha="right", fontsize=8)
    plt.ylabel("AP@[.5:.95]")
    plt.title(f"Per-class AP — clean vs {dtype}/{sev} (distorted / enhanced / fine-tuned)")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    out = _figdir(cfg) / f"per_class_ap_{dtype}_{sev}.png"
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"[viz] {out}")


def plot_image_grids(cfg: Dict, n: int = 4) -> None:
    """clean / distorted / enhanced grid for the first n subset images per distortion."""
    from pycocotools.coco import COCO
    from src.data import load_subset_ids

    ds = cfg["dataset"]
    if not Path(ds["val_subset_file"]).exists():
        print("[viz] no val subset yet; skipping image grids")
        return
    coco = COCO(ds["ann_instances_val"])
    ids = load_subset_ids(ds["val_subset_file"])[:n]
    names = [coco.loadImgs(i)[0]["file_name"] for i in ids]
    figdir = _figdir(cfg)

    for dist, spec in cfg["distortions"].items():
        sev = list(spec["severities"])[-1]   # highest severity
        fig, axes = plt.subplots(n, 3, figsize=(10, 3 * n))
        if n == 1:
            axes = axes[None, :]
        for row, fname in enumerate(names):
            for col, (variant, title) in enumerate(
                [("clean", "Clean"), ("distorted", f"Distorted ({dist}/{sev})"),
                 ("enhanced", "Enhanced")]
            ):
                d = variant_image_dir(cfg, variant, dist, sev)
                p = d / fname
                if p.exists():
                    axes[row, col].imshow(Image.open(p).convert("RGB"))
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(title, fontsize=10)
        out = figdir / f"grid_{dist}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=110)
        plt.close()
        print(f"[viz] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    plot_acc_vs_snr(cfg)
    plot_per_class_ap(cfg)
    plot_per_class_comparison(cfg)
    plot_image_grids(cfg)


if __name__ == "__main__":
    main()
