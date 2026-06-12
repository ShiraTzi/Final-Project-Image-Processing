"""Fine-tune YOLOv8 on distorted train2017 images with REAL COCO GT.

Improvement approach #2 (the assignment's "fine-tune for DL methods").  The
reference pipeline fine-tuned YOLO on *pseudo-labels* from clean predictions;
we have real COCO annotations, so we convert them to YOLO-format labels and
train against true boxes.  We then evaluate the fine-tuned detector on the
*distorted val subset* (held out from training) to measure robustness recovery
without leakage.

Flow (mirrors reference §4 but with real GT):
  build YOLO dataset (distorted images + labels + data.yaml) -> train -> compare.

Usage:
    python -m src.finetune_det --mode train
    python -m src.finetune_det --mode eval     # infer + metrics on distorted val
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


def build_yolo_dataset(cfg: Dict) -> Path:
    """Distort the train subset on the fly, write YOLO images+labels+data.yaml."""
    from pycocotools.coco import COCO

    ds = cfg["dataset"]
    ft = cfg["finetune"]
    dtype, severity = ft["distortion"], ft["severity"]
    params = cfg["distortions"][dtype]["severities"][severity]
    seed = cfg["seed"]

    src_dir = Path(cfg["paths"]["coco_root"]) / ds["train_split"]
    wd = _workdir(cfg)
    img_out = wd / "images" / "train"
    lbl_out = wd / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    coco = COCO(ds["ann_instances_train"])
    img_ids = load_subset_ids(ds["train_subset_file"])
    n_kept = 0
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
        # distort on the fly (same RNG scheme as the val distortions)
        clean = np.array(Image.open(src_dir / im["file_name"]).convert("RGB"))
        dist = apply_distortion(clean, dtype, params, _rng_for(seed, image_id, dtype, severity))
        Image.fromarray(dist).save(img_out / im["file_name"])
        (lbl_out / (Path(im["file_name"]).stem + ".txt")).write_text("\n".join(lines))
        n_kept += 1

    names = get_yolo_model(cfg["yolo"]["weights"]).names   # {0: 'person', ...}
    data_yaml = wd / "data.yaml"
    with open(data_yaml, "w") as f:
        yaml.safe_dump({
            "path": str(wd.resolve()),
            "train": "images/train",
            "val": "images/train",          # real eval is separate (distorted val via COCOeval)
            "names": {int(k): v for k, v in names.items()},
        }, f, sort_keys=False)
    print(f"[finetune] YOLO dataset: {n_kept} images -> {wd}  ({dtype}/{severity})")
    return data_yaml


def train(cfg: Dict) -> Path:
    ft = cfg["finetune"]
    data_yaml = build_yolo_dataset(cfg)
    wd = _workdir(cfg)

    model = get_yolo_model(cfg["yolo"]["weights"])
    model.train(
        data=str(data_yaml),
        epochs=ft["epochs"],
        imgsz=ft["imgsz"],
        batch=ft["batch_size"],
        device=yolo_device(cfg["inference"]["device"]),
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


def evaluate_finetuned(cfg: Dict) -> None:
    """Run the fine-tuned YOLO on the distorted val subset, then COCOeval."""
    from src.inference import run_inference
    from src.metrics import evaluate

    ft = cfg["finetune"]
    dtype, severity = ft["distortion"], ft["severity"]
    ckpt = str(_checkpoint_path(cfg))
    run_inference(cfg, "detection", "finetuned", dtype, severity, checkpoint=ckpt)
    evaluate(cfg, "detection", "finetuned", dtype, severity)


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
        evaluate_finetuned(cfg)


if __name__ == "__main__":
    main()
