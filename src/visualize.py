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


def plot_per_class_comparison(cfg: Dict, top_n: int = 15,
                              severity: str = "high") -> None:
    """Grouped per-class AP bars for every distortion at one severity."""
    mdir = Path(cfg["paths"]["metrics_dir"])
    clean_path = mdir / "detection__clean.json"
    if not clean_path.exists():
        return
    with open(clean_path) as f:
        clean_ap = json.load(f).get("per_class_ap") or {}

    top = sorted(((v, k) for k, v in clean_ap.items() if not np.isnan(v)),
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
    bar_colors = {"clean": INK_MUTED,
                  **{v: st["color"] for v, st in VARIANT_STYLE.items()}}

    for dtype in cfg.get("distortions", {}):
        variants = {"clean": clean_ap}
        for variant in ("distorted", "enhanced", "finetuned"):
            path = mdir / f"detection__{variant_tag(variant, dtype, severity)}.json"
            if path.exists():
                with open(path) as f:
                    variants[variant] = json.load(f).get("per_class_ap") or {}
        if len(variants) < 2:
            continue

        x = np.arange(len(names))
        group_w = 0.8
        width = group_w / len(variants)
        plt.figure(figsize=(14, 5))
        for i, (variant, pcl) in enumerate(variants.items()):
            vals = [pcl.get(n, np.nan) for n in names]
            positions = x + i * width
            plt.bar(positions, vals, width, label=variant,
                    color=bar_colors[variant])
            for xi, value in zip(positions, vals):
                if value == 0:  # measured zero, not missing data
                    plt.text(xi, 0.004, "0", ha="center", va="bottom",
                             fontsize=6, color=INK)
        plt.xticks(x + (group_w - width) / 2, tick_labels,
                   rotation=45, ha="right", fontsize=8)
        plt.ylabel("AP@[.5:.95]")
        plt.title(
            f"Per-class AP — clean vs {dtype}/{severity} "
            "(distorted / enhanced / fine-tuned)"
        )
        plt.legend()
        plt.gca().set_axisbelow(True)
        plt.grid(True, axis="y", alpha=0.3)
        out = _figdir(cfg) / f"per_class_ap_{dtype}_{severity}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=120)
        plt.close()
        print(f"[viz] {out}")


def plot_image_grids(cfg: Dict, n: int = 4,
                     exclude_ids: "set | None" = None) -> None:
    """clean / distorted / enhanced grid for n subset images per distortion.
    Pass exclude_ids to avoid showing images already used in annotation grids."""
    from pycocotools.coco import COCO
    from src.data import load_subset_ids

    ds = cfg["dataset"]
    if not Path(ds["val_subset_file"]).exists():
        print("[viz] no val subset yet; skipping image grids")
        return
    coco = COCO(ds["ann_instances_val"])
    _excl = exclude_ids or set()
    all_ids = load_subset_ids(ds["val_subset_file"])
    ids = [i for i in all_ids if i not in _excl][:n]
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
    return ids


def plot_keypoint_grids(cfg: Dict, n: int = 2, score_thr: float = 0.35) -> None:
    """Keypoint R-CNN predictions drawn on the images ("image with annotation"
    for the keypoints task): skeleton + joint dots for each detected person,
    clean / distorted (high severity) / enhanced, one figure per distortion.
    Drawn from the existing prediction JSONs — no model inference needed."""
    import matplotlib.patches as mpatches
    from src.data import get_coco, load_subset_ids

    # COCO 17-keypoint skeleton (0-indexed pairs)
    SKELETON = [
        (0, 1), (0, 2), (1, 3), (2, 4),          # head
        (5, 6),                                    # shoulders
        (5, 7), (7, 9), (6, 8), (8, 10),          # arms
        (5, 11), (6, 12), (11, 12),                # torso
        (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    ]
    KP_COLOR  = "#f5c518"
    LIMB_COLOR = "#e8e8e8"

    ds = cfg["dataset"]
    if not Path(ds["val_subset_file"]).exists():
        print("[viz] no val subset yet; skipping keypoint grids")
        return
    coco_kp = get_coco(ds["ann_keypoints_val"])
    preds_dir = Path(cfg["paths"]["preds_dir"])
    figdir = _figdir(cfg)

    # pick images with at least 2 persons with visible keypoints — richer scenes
    ids_all = load_subset_ids(ds["val_subset_file"])
    ids = [i for i in ids_all
           if sum(1 for a in coco_kp.loadAnns(
               coco_kp.getAnnIds(imgIds=i, iscrowd=False))
               if a.get("num_keypoints", 0) >= 5) >= 2][:n]
    if not ids:
        print("[viz] no suitable person images for keypoint grids; skipping")
        return

    def _load_kp_preds(tag):
        p = preds_dir / f"keypoints__{tag}.json"
        if not p.exists():
            return None
        by_img: Dict[int, list] = {}
        with open(p) as f:
            for d in json.load(f):
                if d.get("score", 0) >= score_thr:
                    by_img.setdefault(d["image_id"], []).append(d)
        return by_img

    def _draw_kps(ax, img_path, preds, gt_anns, color):
        ax.imshow(Image.open(img_path).convert("RGB"))
        for a in gt_anns:                              # GT skeleton: gray dashed
            kp = a["keypoints"]
            for i, j in SKELETON:
                xi, yi, vi = kp[3*i], kp[3*i+1], kp[3*i+2]
                xj, yj, vj = kp[3*j], kp[3*j+1], kp[3*j+2]
                if vi > 0 and vj > 0:
                    ax.plot([xi, xj], [yi, yj], color=INK_MUTED,
                            lw=1.0, ls="--", zorder=1)
        for pred in preds or []:                       # predictions: skeleton + dots
            kp = pred["keypoints"]
            for i, j in SKELETON:
                xi, yi, vi = kp[3*i], kp[3*i+1], kp[3*i+2]
                xj, yj, vj = kp[3*j], kp[3*j+1], kp[3*j+2]
                if vi > 0 and vj > 0:
                    ax.plot([xi, xj], [yi, yj], color=LIMB_COLOR,
                            lw=1.4, zorder=2)
            for k in range(17):
                x, y, v = kp[3*k], kp[3*k+1], kp[3*k+2]
                if v > 0:
                    ax.plot(x, y, "o", color=KP_COLOR, ms=4, zorder=3)
        ax.axis("off")

    clean_preds = _load_kp_preds("clean")
    coco_inst = get_coco(ds["ann_instances_val"])
    for dist, spec in cfg["distortions"].items():
        sev = list(spec["severities"])[-1]
        cols = [
            ("clean",     "clean",     None,  f"Clean + Keypoint R-CNN"),
            ("distorted", "distorted", dist,  f"Distorted ({dist}/{sev})"),
            ("enhanced",  "enhanced",  dist,  "Enhanced"),
        ]
        preds_by_col = {}
        for variant, _, d, _ in cols:
            tag = "clean" if variant == "clean" else variant_tag(variant, dist, sev)
            preds_by_col[variant] = (clean_preds if variant == "clean"
                                     else _load_kp_preds(tag))
        if preds_by_col["distorted"] is None:
            continue
        fig, axes = plt.subplots(len(ids), len(cols),
                                 figsize=(4.5 * len(cols), 3.4 * len(ids)))
        axes = axes[None, :] if len(ids) == 1 else axes
        for row, image_id in enumerate(ids):
            im = coco_kp.loadImgs(image_id)[0]
            gt = coco_kp.loadAnns(coco_kp.getAnnIds(imgIds=image_id, iscrowd=False))
            for col, (variant, img_variant, d, title) in enumerate(cols):
                img_dir = variant_image_dir(
                    cfg, img_variant,
                    d, sev if d else None)
                path = resolve_image_path(img_dir, im["file_name"])
                preds = (preds_by_col[variant] or {}).get(image_id, [])
                _draw_kps(axes[row, col], path, preds, gt, VARIANT_STYLE[
                    "distorted" if variant == "clean" else variant]["color"])
                if row == 0:
                    axes[row, col].set_title(title, fontsize=10, color=INK)
        fig.suptitle(
            f"Keypoints on {dist}/{sev} — gray dashed = GT skeleton, "
            f"white lines + yellow dots = predictions (score ≥ {score_thr})",
            fontsize=11)
        out = figdir / f"keypoints_{dist}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"[viz] {out}")
    return ids


def plot_orb_match_grids(cfg: Dict, n: int = 2, max_draw: int = 60) -> None:
    """ORB matching visualized ("image with annotation" for the features task):
    clean-image keypoints matched against the distorted and the enhanced
    variant, lines = good matches (same matcher/threshold as the metric), one
    figure per distortion at its highest severity.  The per-pair caption is
    the actual metric for that image: good matches / clean keypoints."""
    import cv2
    from src.data import get_coco, load_subset_ids

    ds = cfg["dataset"]
    if not Path(ds["val_subset_file"]).exists():
        print("[viz] no val subset yet; skipping ORB match grids")
        return
    coco = get_coco(ds["ann_instances_val"])
    # same image-selection rule as the annotated detection grids, so the
    # qualitative figures show the same scenes across tasks
    ids = [i for i in load_subset_ids(ds["val_subset_file"])
           if 3 <= len(coco.getAnnIds(imgIds=i, iscrowd=False)) <= 12][:n]
    names = [coco.loadImgs(i)[0]["file_name"] for i in ids]
    figdir = _figdir(cfg)

    nfeatures = int(cfg["orb"]["nfeatures"])
    dist_thr = int(cfg["orb"]["distance_threshold"])
    orb = cv2.ORB_create(nfeatures=nfeatures)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    clean_dir = variant_image_dir(cfg, "clean")

    def _hex_bgr(h):
        return tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))

    def _match_pair(clean_path, var_path, color_hex):
        """Side-by-side clean|variant image with good-match lines drawn."""
        img1 = cv2.imread(str(clean_path), cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(str(var_path), cv2.IMREAD_GRAYSCALE)
        kp1, d1 = orb.detectAndCompute(img1, None)
        kp2, d2 = orb.detectAndCompute(img2, None)
        good = []
        if d1 is not None and d2 is not None:
            good = sorted((m for m in matcher.match(d1, d2)
                           if m.distance <= dist_thr), key=lambda m: m.distance)
        vis = cv2.drawMatches(
            cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR), kp1,
            cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR), kp2,
            good[:max_draw], None, matchColor=_hex_bgr(color_hex),
            singlePointColor=(190, 190, 190),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        return cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), len(good), len(kp1)

    for dist, spec in cfg["distortions"].items():
        sev = list(spec["severities"])[-1]                   # highest severity
        cols = [("distorted", f"Clean ↔ Distorted ({dist}/{sev})",
                 VARIANT_STYLE["distorted"]["color"]),
                ("enhanced", "Clean ↔ Enhanced",
                 VARIANT_STYLE["enhanced"]["color"])]
        if not variant_image_dir(cfg, "distorted", dist, sev).is_dir():
            continue
        fig, axes = plt.subplots(len(names), len(cols),
                                 figsize=(7.4 * len(cols), 2.9 * len(names)))
        axes = axes[None, :] if len(names) == 1 else axes
        for row, fname in enumerate(names):
            for col, (variant, title, color) in enumerate(cols):
                var_dir = variant_image_dir(cfg, variant, dist, sev)
                vis, n_good, n_kp = _match_pair(
                    clean_dir / fname, resolve_image_path(var_dir, fname), color)
                ax = axes[row, col]
                ax.imshow(vis)
                ax.set_xlabel(f"{n_good} / {n_kp} clean keypoints matched "
                              f"(ratio {n_good / max(n_kp, 1):.2f})",
                              fontsize=9, color=INK)
                ax.set_xticks([]), ax.set_yticks([])
                for spine in ax.spines.values():
                    spine.set_visible(False)
                if row == 0:
                    ax.set_title(title, fontsize=11, color=INK)
        fig.suptitle(f"ORB feature matching under {dist}/{sev} — lines = good "
                     f"matches (Hamming ≤ {dist_thr}, cross-checked, "
                     f"best {max_draw} drawn)", fontsize=11)
        out = figdir / f"orb_matches_{dist}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"[viz] {out}")
    return ids


def plot_panoptic_grids(cfg: Dict, n: int = 2, alpha: float = 0.55) -> None:
    """Panoptic predictions drawn on the images: GT segments / clean pred /
    distorted pred / enhanced pred, one figure per distortion at its highest
    severity.  Colors are keyed by *category* (same class = same color in
    every panel, golden-ratio hue spacing); white lines = segment boundaries.
    Pure drawing from the saved panoptic PNGs + segments_info — no model."""
    import colorsys

    import cv2
    from src.data import get_coco, load_subset_ids

    ds = cfg["dataset"]
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    preds_dir = Path(cfg["paths"]["preds_dir"])
    gt_json = metrics_dir / "panoptic_gt_subset.json"
    gt_dir = Path(ds["panoptic_val_dir"])
    if not (Path(ds["val_subset_file"]).exists() and gt_json.exists()
            and gt_dir.is_dir()):
        print("[viz] panoptic GT/preds missing; skipping panoptic grids")
        return

    coco = get_coco(ds["ann_instances_val"])
    ids = [i for i in load_subset_ids(ds["val_subset_file"])
           if 3 <= len(coco.getAnnIds(imgIds=i, iscrowd=False)) <= 12][:n]
    figdir = _figdir(cfg)

    def _cat_color(cat_id: int):
        hue = (cat_id * 0.61803398875) % 1.0            # golden-ratio spacing
        return np.array(colorsys.hsv_to_rgb(hue, 0.62, 0.92))

    def _segments_by_image(json_path: Path):
        with open(json_path) as f:
            data = json.load(f)
        anns = data["annotations"] if isinstance(data, dict) else data
        return {a["image_id"]: a["segments_info"] for a in anns}

    def _overlay(ax, base_img_path: Path, mask_png: Path, segs: list, title=None):
        base = np.asarray(Image.open(base_img_path).convert("RGB"),
                          dtype=np.float64) / 255.0
        if mask_png.exists() and segs is not None:
            m = np.asarray(Image.open(mask_png).convert("RGB"), dtype=np.uint32)
            seg_id = m[..., 0] + 256 * m[..., 1] + 65536 * m[..., 2]
            if seg_id.shape != base.shape[:2]:       # enhanced/distorted PNGs match, GT is same size
                seg_id = cv2.resize(seg_id.astype(np.int32), base.shape[1::-1],
                                    interpolation=cv2.INTER_NEAREST).astype(np.uint32)
            painted = np.zeros_like(base)
            covered = np.zeros(seg_id.shape, dtype=bool)
            for s in segs:
                region = seg_id == s["id"]
                painted[region] = _cat_color(s["category_id"])
                covered |= region
            out = base.copy()
            out[covered] = (1 - alpha) * base[covered] + alpha * painted[covered]
            # white hairline boundaries between segments
            edges = (cv2.morphologyEx(seg_id.astype(np.float32), cv2.MORPH_GRADIENT,
                                      np.ones((3, 3), np.float32)) > 0) & covered
            out[edges] = 1.0
            ax.imshow(out)
        else:
            ax.imshow(base)
        ax.axis("off")
        if title:
            ax.set_title(title, fontsize=10, color=INK)

    gt_segs = _segments_by_image(gt_json)

    def _pq(tag):
        p = metrics_dir / f"segmentation__{tag}.json"
        if p.exists():
            with open(p) as f:
                return json.load(f).get("PQ")
        return None

    for dist, spec in cfg["distortions"].items():
        sev = list(spec["severities"])[-1]                   # highest severity
        tags = {"clean": "clean",
                "distorted": variant_tag("distorted", dist, sev),
                "enhanced": variant_tag("enhanced", dist, sev)}
        pred_segs = {}
        for v, tag in tags.items():
            j = preds_dir / f"segmentation__{tag}.json"
            if not j.exists():
                pred_segs = None
                break
            pred_segs[v] = _segments_by_image(j)
        if pred_segs is None:
            continue

        def _fmt_pq(tag):
            v = _pq(tag)
            return f" — PQ {v:.3f}" if v is not None else ""

        cols = [("gt", "clean", "Ground truth"),
                ("clean", "clean", f"Clean{_fmt_pq('clean')}"),
                ("distorted", "distorted",
                 f"Distorted ({dist}/{sev}){_fmt_pq(tags['distorted'])}"),
                ("enhanced", "enhanced", f"Enhanced{_fmt_pq(tags['enhanced'])}")]
        fig, axes = plt.subplots(len(ids), len(cols),
                                 figsize=(4.0 * len(cols), 3.1 * len(ids)))
        axes = axes[None, :] if len(ids) == 1 else axes
        for row, image_id in enumerate(ids):
            im = coco.loadImgs(image_id)[0]
            stem = Path(im["file_name"]).stem
            for col, (source, img_variant, title) in enumerate(cols):
                img_dir = variant_image_dir(
                    cfg, img_variant, dist if img_variant != "clean" else None,
                    sev if img_variant != "clean" else None)
                base = resolve_image_path(img_dir, im["file_name"])
                if source == "gt":
                    mask, segs = gt_dir / f"{stem}.png", gt_segs.get(image_id)
                else:
                    mask = preds_dir / f"segmentation__{tags[source]}" / f"{stem}.png"
                    segs = pred_segs[source].get(image_id)
                _overlay(axes[row, col], base, mask, segs,
                         title if row == 0 else None)
        fig.suptitle(f"Panoptic segmentation on {dist}/{sev} — color = category "
                     "(consistent across panels), white = segment boundaries",
                     fontsize=11)
        out = figdir / f"panoptic_{dist}.png"
        fig.tight_layout()
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"[viz] {out}")
    return ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    plot_acc_vs_snr(cfg)
    plot_recovery_bars(cfg)
    plot_per_class_ap(cfg)
    plot_per_class_comparison(cfg)
    # Run annotation grids first so we know which images they used,
    # then give the raw grid a disjoint set of images.
    ann_ids  = plot_annotated_grids(cfg) or []
    kp_ids   = plot_keypoint_grids(cfg) or []
    orb_ids  = plot_orb_match_grids(cfg) or []
    pan_ids  = plot_panoptic_grids(cfg) or []
    used_ids = set(ann_ids) | set(kp_ids) | set(orb_ids) | set(pan_ids)
    plot_image_grids(cfg, exclude_ids=used_ids)


if __name__ == "__main__":
    main()
