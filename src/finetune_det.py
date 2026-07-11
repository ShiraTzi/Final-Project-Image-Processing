"""Fine-tune YOLOv8 on corrupted train2017 images with REAL COCO GT.

Improvement approach #2 (the assignment's "fine-tune for DL methods").  The
reference pipeline fine-tuned YOLO on *pseudo-labels* from clean predictions;
we have real COCO annotations, so we convert them to YOLO-format labels and
train against true boxes.

Training distribution: a per-image seeded MIXTURE of clean + every
(distortion, severity) cell.  Fine-tuning on a single corruption cell was
measured to cause graded negative transfer — the model beat the pretrained
baseline in-domain but LOST on every motion-blur cell and every low-severity
cell (it learned to expect heavy i.i.d. noise and drifted away from clean
statistics).  The mixture keeps every evaluation cell in-domain.

A seeded 10% split of the train subset is held out as the YOLO val set so
best-checkpoint selection is honest (previously val == train, which selects
for memorization).  The benchmark's val2017 subset is never touched during
training — no leakage.

We then evaluate the fine-tuned detector on the *clean* val subset (does
robustness cost clean accuracy?), on all distorted cells (in-domain recovery),
and on all enhanced cells (are enhancement and fine-tuning additive?).

Usage:
    python -m src.finetune_det --mode train
    python -m src.finetune_det --mode eval     # infer + metrics on clean/distorted/enhanced val
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import yaml
from PIL import Image
from tqdm import tqdm

from src.config import load_config, ensure_dirs
from src.data import load_subset_ids
from src.distortions import apply_distortion, _rng_for
from src.models import COCO91_TO_80, get_yolo_model, yolo_device


def _workdir(cfg: Dict) -> Path:
    wd = Path(cfg["finetune"]["workdir"])
    return wd if wd.is_absolute() else Path(cfg["_root"]) / wd


def _checkpoint_path(cfg: Dict) -> Path:
    ck = Path(cfg["finetune"]["checkpoint"])
    return ck if ck.is_absolute() else Path(cfg["_root"]) / ck


def _mixture_variants(cfg: Dict) -> list:
    """Training variants: clean + every (distortion, severity) cell."""
    variants = [("clean", None)]
    for dt, spec in cfg["distortions"].items():
        for sv in spec["severities"]:
            variants.append((dt, sv))
    return variants


def build_yolo_dataset(cfg: Dict) -> Path:
    """Corrupt the train subset on the fly (per-image seeded mixture of clean +
    all distortion cells), write YOLO images+labels+data.yaml with a held-out
    val split for honest best-checkpoint selection."""
    from pycocotools.coco import COCO

    ds = cfg["dataset"]
    ft = cfg["finetune"]
    seed = cfg["seed"]
    variants = _mixture_variants(cfg)
    val_frac = float(ft.get("val_fraction", 0.1))

    src_dir = Path(cfg["paths"]["coco_root"]) / ds["train_split"]
    wd = _workdir(cfg)
    for split in ("train", "val"):
        (wd / "images" / split).mkdir(parents=True, exist_ok=True)
        (wd / "labels" / split).mkdir(parents=True, exist_ok=True)

    coco = COCO(ds["ann_instances_train"])
    img_ids = load_subset_ids(ds["train_subset_file"])
    # seeded held-out split (train images only — the benchmark's val2017 subset
    # must stay untouched by training)
    val_rng = np.random.default_rng(seed)
    val_ids = set(val_rng.choice(img_ids, size=int(len(img_ids) * val_frac),
                                 replace=False).tolist())

    n_kept = {"train": 0, "val": 0}
    mix_counts: Dict[str, int] = {}
    for image_id in tqdm(img_ids, desc="yolo-dataset", leave=False):
        im = coco.loadImgs(image_id)[0]
        W, H = im["width"], im["height"]
        lines = []
        for a in coco.loadAnns(coco.getAnnIds(imgIds=image_id, iscrowd=False)):
            x, y, w, h = a["bbox"]
            if w <= 1 or h <= 1 or a["category_id"] not in COCO91_TO_80:
                continue
            cls = COCO91_TO_80[a["category_id"]]
            cx, cy = (x + w / 2) / W, (y + h / 2) / H
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        if not lines:
            continue
        # per-image seeded variant pick + distortion (same RNG scheme as val)
        pick = _rng_for(seed, image_id, "mixture", "pick").integers(len(variants))
        dtype, severity = variants[int(pick)]
        clean = np.array(Image.open(src_dir / im["file_name"]).convert("RGB"))
        if dtype == "clean":
            out = clean
        else:
            params = cfg["distortions"][dtype]["severities"][severity]
            out, _ = apply_distortion(clean, dtype, params,
                                      _rng_for(seed, image_id, dtype, severity))
        split = "val" if image_id in val_ids else "train"
        # PNG: keep training images as lossless as the benchmark's val images
        out_name = Path(im["file_name"]).stem + ".png"
        Image.fromarray(out).save(wd / "images" / split / out_name)
        (wd / "labels" / split / (Path(im["file_name"]).stem + ".txt")).write_text(
            "\n".join(lines))
        n_kept[split] += 1
        key = "clean" if dtype == "clean" else f"{dtype}/{severity}"
        mix_counts[key] = mix_counts.get(key, 0) + 1

    names = get_yolo_model(cfg["yolo"]["weights"]).names   # {0: 'person', ...}
    data_yaml = wd / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.safe_dump({
            "path": str(wd.resolve()),
            "train": "images/train",
            "val": "images/val",            # held-out split -> honest best.pt selection
            "names": {int(k): v for k, v in names.items()},
        }, f, sort_keys=False)
    print(f"[finetune] YOLO dataset: {n_kept['train']} train / {n_kept['val']} val "
          f"-> {wd}  mixture: {mix_counts}")
    return data_yaml


def train(cfg: Dict) -> Path:
    from ultralytics import YOLO

    ft = cfg["finetune"]
    data_yaml = build_yolo_dataset(cfg)
    wd = _workdir(cfg)

    # deliberately NOT get_yolo_model(): .train() mutates the model in place,
    # which would poison the shared cached pretrained instance.
    model = YOLO(cfg["yolo"]["weights"])
    model.train(
        data=str(data_yaml),
        epochs=ft["epochs"],
        imgsz=ft["imgsz"],
        batch=ft["batch_size"],
        device=yolo_device(cfg["inference"]["device"]),
        # pinned explicitly: `optimizer=auto` silently overrides lr0, so the
        # recorded hyperparameters would not match what actually ran
        optimizer="AdamW",
        lr0=ft.get("lr0", 1.0e-4),
        lrf=0.01,
        cos_lr=True,
        patience=ft.get("patience", 10),
        project=str(wd / "runs"),
        name="finetune",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    best = Path(model.trainer.best)
    ckpt = _checkpoint_path(cfg)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, ckpt)
    print(f"[finetune] saved checkpoint -> {ckpt}")
    return ckpt


def evaluate_finetuned(cfg: Dict, force: bool = False) -> list:
    """Run the fine-tuned YOLO on the clean val subset + every distorted and
    enhanced val cell, then COCOeval.

    - clean:      does robustness training cost clean accuracy? (forgetting check)
    - distorted:  in-domain recovery on all 9 cells
    - enhanced:   is classical restoration + fine-tuning additive?

    All cells are held out — the model never sees a val2017 image in training.
    Resumable like the other stages: cells whose metric JSON exists are skipped,
    and one failed cell doesn't abort the rest.  Returns the failed tags."""
    import traceback

    from src.config import variant_tag
    from src.inference import run_inference
    from src.metrics import evaluate

    ckpt = str(_checkpoint_path(cfg))
    metrics_dir = Path(cfg["paths"]["metrics_dir"])
    preds_dir = Path(cfg["paths"]["preds_dir"])

    cells = [("finetuned", None, None)]                      # clean images
    for dtype, spec in cfg["distortions"].items():
        for severity in spec["severities"]:
            cells.append(("finetuned", dtype, severity))     # distorted images
            cells.append(("finetuned_enh", dtype, severity))  # enhanced images

    failures = []
    for variant, dtype, severity in cells:
        tag = variant_tag(variant, dtype, severity)
        if (metrics_dir / f"detection__{tag}.json").exists() and not force:
            print(f"[finetune] skip {tag}: metrics exist")
            continue
        try:
            if force or not (preds_dir / f"detection__{tag}.json").exists():
                run_inference(cfg, "detection", variant, dtype, severity,
                              checkpoint=ckpt)
            evaluate(cfg, "detection", variant, dtype, severity)
        except Exception:  # noqa: BLE001 — one bad cell shouldn't abort the rest
            traceback.print_exc()
            print(f"[finetune] FAILED eval {tag}")
            failures.append(f"finetune:detection/{tag}")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["train", "eval", "both"], default="both")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.mode in ("train", "both"):
        train(cfg)
    if args.mode in ("eval", "both"):
        failures = evaluate_finetuned(cfg)
        if failures:
            raise SystemExit(f"[finetune] {len(failures)} cell(s) failed: {failures}")


if __name__ == "__main__":
    main()
