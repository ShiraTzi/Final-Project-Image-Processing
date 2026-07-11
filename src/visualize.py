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

from src.config import load_config, resolve_image_path, variant_image_dir, variant_tag


def _figdir(cfg: Dict) -> Path:
    d = Path(cfg["paths"]["figures_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


# Variant identity is fixed across every panel and figure: same color, marker,
# and linestyle everywhere (validated categorical slots 1-3; marker + linestyle
# double-encode identity so color never carries it alone).
VARIANT_STYLE = {
    "distorted": {"color": "#2a78d6", "marker": "o", "ls": "-"},
    "enhanced":  {"color": "#1baf7a", "marker": "s", "ls": "--"},
    "finetuned": {"color": "#eda100", "marker": "^", "ls": ":"},
}
INK = "#52514e"        # secondary ink: axis titles
INK_MUTED = "#898781"  # muted ink: ticks, baseline
GRID = "#e1e0d9"       # hairline grid


def plot_acc_vs_snr(cfg: Dict) -> None:
    """One figure per task, one panel per distortion: metric vs SNR with
    distorted / enhanced / finetuned as fixed-identity series."""
    long_path = Path(cfg["paths"]["metrics_dir"]) / "summary_long.csv"
    if not long_path.exists():
        print("[viz] no summary_long.csv; run tables first")
        return
    df = pd.read_csv(long_path)
    figdir = _figdir(cfg)

    for task in df["task"].unique():
        sub = df[df["task"] == task]
        clean_val = sub[sub["variant"] == "clean"]["value"]
        dists = sorted(sub["distortion"].dropna().unique())
        if not dists:
            continue
        fig, axes = plt.subplots(1, len(dists), figsize=(4.4 * len(dists), 4.2),
                                 sharey=True)
        axes = [axes] if len(dists) == 1 else list(axes)
        for ax, dist in zip(axes, dists):
            for variant, st in VARIANT_STYLE.items():
                d = sub[(sub["distortion"] == dist) & (sub["variant"] == variant)]
                d = d.dropna(subset=["snr_db"]).sort_values("snr_db")
                if d.empty:
                    continue
                ax.plot(d["snr_db"], d["value"], color=st["color"], ls=st["ls"],
                        marker=st["marker"], ms=7, lw=2, label=variant)
            if not clean_val.empty:
                ax.axhline(float(clean_val.iloc[0]), color=INK_MUTED, ls="--",
                           lw=1.5, label=f"clean = {float(clean_val.iloc[0]):.3f}")
            ax.invert_xaxis()   # lower SNR = more distortion, to the right
            ax.set_title(dist, fontsize=11, color=INK)
            ax.set_xlabel("SNR (dB)  <- more distortion", fontsize=9, color=INK)
            ax.grid(True, color=GRID, lw=0.8)
            ax.tick_params(colors=INK_MUTED, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(GRID)
        axes[0].set_ylabel(sub["metric"].iloc[0], fontsize=10, color=INK)
        axes[0].legend(fontsize=8, loc="lower left")   # identity is shared -> one legend
        fig.suptitle(f"{task}: performance vs SNR", fontsize=12)
        out = figdir / f"acc_vs_snr_{task}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=120)
        plt.close(fig)
        print(f"[viz] {out}")


def plot_recovery_bars(cfg: Dict) -> None:
    """Absolute metric values per distortion × severity cell, one panel per
    task: distorted vs enhanced vs finetuned bars against the clean-baseline
    line, with '% of the damage recovered' labels on repairs that helped —
    so both the depth of the damage and the size of the rescue are visible."""
    comp_path = Path(cfg["paths"]["metrics_dir"]) / "comparison.csv"
    if not comp_path.exists():
        return
    df = pd.read_csv(comp_path)
    tasks = list(df["task"].unique())
    if not tasks:
        return
    sev_order = {"low": 0, "med": 1, "high": 2}
    fig, axes = plt.subplots(len(tasks), 1, figsize=(11, 3.0 * len(tasks)),
                             sharex=True)
    axes = [axes] if len(tasks) == 1 else list(axes)
    for ax, task in zip(axes, tasks):
        sub = df[df["task"] == task].copy()
        sub["order"] = sub["severity"].map(sev_order)
        sub = sub.sort_values(["distortion", "order"]).reset_index(drop=True)
        labels = [f"{d}\n{s}" for d, s in zip(sub["distortion"], sub["severity"])]
        x = np.arange(len(sub))
        has_ft = sub["finetuned"].notna().any()
        series = ["distorted", "enhanced"] + (["finetuned"] if has_ft else [])
        width = 0.8 / len(series)
        for i, variant in enumerate(series):
            xpos = x - 0.4 + (i + 0.5) * width
            ax.bar(xpos, sub[variant], width,
                   color=VARIANT_STYLE[variant]["color"], label=variant)
            if variant == "distorted":
                continue
            # label repairs that helped with % of the damage they recovered
            rec = sub[f"recovery_{'enhance' if variant == 'enhanced' else 'finetune'}"]
            pct = rec / (-sub["degradation"])
            for xi, v, p, r in zip(xpos, sub[variant], pct, rec):
                if not np.isnan(v) and r > 0.02:
                    ax.text(xi, v + 0.01, f"+{p:.0%}", ha="center", va="bottom",
                            fontsize=7, color=INK)
        clean_v = float(sub["clean"].iloc[0])
        ax.axhline(clean_v, color=INK_MUTED, ls="--", lw=1.5,
                   label=f"clean = {clean_v:.3f}")
        ax.set_ylabel(f"{task}\n({'match ratio' if task == 'features' else 'mAP' if task in ('detection', 'keypoints') else 'PQ'})",
                      fontsize=9, color=INK)
        ax.set_ylim(0, clean_v * 1.18)
        ax.set_axisbelow(True)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=INK_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(GRID)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.legend(fontsize=7, loc="upper left", ncol=4)
    fig.suptitle("After repair: metric values vs the clean baseline "
                 "(labels = share of the damage recovered)", fontsize=12)
    out = _figdir(cfg) / "recovery_bars.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
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

    # GT instance counts in the subset — rare classes have high-variance AP,
    # so show n= under each class name.
    from src.data import get_coco, load_subset_ids
    coco = get_coco(cfg["dataset"]["ann_instances_val"])
    subset_ids = set(load_subset_ids(cfg["dataset"]["val_subset_file"]))
    name2id = {c["name"]: c["id"] for c in coco.loadCats(coco.getCatIds())}
    gt_n = {n: sum(1 for a in coco.loadAnns(coco.getAnnIds(catIds=[name2id[n]],
                                                            iscrowd=False))
                   if a["image_id"] in subset_ids) for n in names}
    tick_labels = [f"{n}\n(n={gt_n[n]})" for n in names]
    x = np.arange(len(names))
    group_w = 0.8
    width = group_w / len(per_class)
    bar_colors = {"clean": INK_MUTED,
                  **{v: st["color"] for v, st in VARIANT_STYLE.items()}}
    plt.figure(figsize=(14, 5))
    for i, (variant, pcl) in enumerate(per_class.items()):
        vals = [pcl.get(n, np.nan) for n in names]
        plt.bar(x + i * width, vals, width, label=variant, color=bar_colors[variant])
        for xi, v in zip(x + i * width, vals):
            if v == 0:   # measured zero, not missing data — make it visible
                plt.text(xi, 0.004, "0", ha="center", va="bottom",
                         fontsize=6, color=INK)
    plt.xticks(x + (group_w - width) / 2, tick_labels, rotation=45, ha="right",
               fontsize=8)
    plt.ylabel("AP@[.5:.95]")
    plt.title(f"Per-class AP — clean vs {dtype}/{sev} (distorted / enhanced / fine-tuned)")
    plt.legend()
    plt.gca().set_axisbelow(True)
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
                p = resolve_image_path(d, fname)
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


def plot_annotated_grids(cfg: Dict, n: int = 3, score_thr: float = 0.35) -> None:
    """Model predictions drawn on the images ("image with annotation"):
    ground-truth boxes in gray + detector boxes with class/score, for
    clean / distorted / enhanced (pretrained YOLO) and distorted (fine-tuned),
    one figure per distortion at its highest severity.  Pure drawing from the
    existing prediction JSONs — no model inference needed."""
    import matplotlib.patches as mpatches
    from src.data import get_coco, load_subset_ids

    ds = cfg["dataset"]
    if not Path(ds["val_subset_file"]).exists():
        print("[viz] no val subset yet; skipping annotated grids")
        return
    coco = get_coco(ds["ann_instances_val"])
    cat_names = {c["id"]: c["name"] for c in coco.loadCats(coco.getCatIds())}
    preds_dir = Path(cfg["paths"]["preds_dir"])
    figdir = _figdir(cfg)

    # pick the first n subset images with a healthy number of GT objects
    ids = [i for i in load_subset_ids(ds["val_subset_file"])
           if 3 <= len(coco.getAnnIds(imgIds=i, iscrowd=False)) <= 12][:n]
    if not ids:
        print("[viz] no suitable images for annotated grids")
        return

    def _load_preds(tag):
        p = preds_dir / f"detection__{tag}.json"
        if not p.exists():
            return None
        by_img: Dict[int, list] = {}
        with open(p) as f:
            for d in json.load(f):
                if d["score"] >= score_thr:
                    by_img.setdefault(d["image_id"], []).append(d)
        return by_img

    def _draw(ax, img_path, gt_anns, preds, color):
        ax.imshow(Image.open(img_path).convert("RGB"))
        for a in gt_anns:                                   # GT: gray, dashed
            x, y, w, h = a["bbox"]
            ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False,
                                            edgecolor=INK_MUTED, ls="--", lw=1.2))
        for d in preds or []:                               # predictions: variant color
            x, y, w, h = d["bbox"]
            ax.add_patch(mpatches.Rectangle((x, y), w, h, fill=False,
                                            edgecolor=color, lw=1.6))
            ax.text(x, max(y - 3, 0),
                    f"{cat_names.get(d['category_id'], '?')} {d['score']:.2f}",
                    fontsize=6, color="white",
                    bbox=dict(facecolor=color, pad=0.5, edgecolor="none"))
        ax.axis("off")

    clean_preds = _load_preds("clean")
    for dist, spec in cfg["distortions"].items():
        sev = list(spec["severities"])[-1]                   # highest severity
        cols = [
            ("clean", "clean", None, "Clean + YOLO", VARIANT_STYLE["distorted"]["color"]),
            ("distorted", "distorted", dist, f"Distorted ({dist}/{sev})",
             VARIANT_STYLE["distorted"]["color"]),
            ("enhanced", "enhanced", dist, "Enhanced", VARIANT_STYLE["enhanced"]["color"]),
            ("finetuned", "distorted", dist, "Distorted + fine-tuned YOLO",
             VARIANT_STYLE["finetuned"]["color"]),
        ]
        preds_by_col = {}
        for variant, _, d, _, _ in cols:
            tag = "clean" if variant == "clean" else variant_tag(variant, dist, sev)
            preds_by_col[variant] = clean_preds if variant == "clean" else _load_preds(tag)
        if preds_by_col["distorted"] is None:
            continue
        fig, axes = plt.subplots(len(ids), len(cols),
                                 figsize=(4.0 * len(cols), 3.1 * len(ids)))
        axes = axes[None, :] if len(ids) == 1 else axes
        for row, image_id in enumerate(ids):
            im = coco.loadImgs(image_id)[0]
            gt = coco.loadAnns(coco.getAnnIds(imgIds=image_id, iscrowd=False))
            for col, (variant, img_variant, d, title, color) in enumerate(cols):
                img_dir = variant_image_dir(cfg, img_variant, d, sev if d else None)
                path = resolve_image_path(img_dir, im["file_name"])
                preds = (preds_by_col[variant] or {}).get(image_id, [])
                _draw(axes[row, col], path, gt, preds, color)
                if row == 0:
                    axes[row, col].set_title(title, fontsize=10, color=INK)
        fig.suptitle(f"Detections on {dist}/{sev} — gray dashed = ground truth, "
                     f"solid = YOLO predictions (score ≥ {score_thr})", fontsize=11)
        out = figdir / f"annotated_{dist}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"[viz] {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    plot_acc_vs_snr(cfg)
    plot_recovery_bars(cfg)
    plot_per_class_ap(cfg)
    plot_per_class_comparison(cfg)
    plot_image_grids(cfg)
    plot_annotated_grids(cfg)


if __name__ == "__main__":
    main()
