"""Fine-tune YOLOv8 on corrupted train2017 images with REAL COCO GT.

Improvement approach #2 (the assignment's "fine-tune for DL methods").  The
reference pipeline fine-tuned YOLO on *pseudo-labels* from clean predictions;
we have real COCO annotations, so we convert them to YOLO-format labels and
train against true boxes.

Training distribution (v3): a per-image seeded MIXTURE of
  - clean images with p = finetune.clean_fraction (0.25), else
  - one of the 9 (distortion, severity) cells uniformly, which is then passed
    through the matched classical restorer with p = finetune.restored_fraction.

Why each piece is there, measured on v2:
  - MIXTURE at all: fine-tuning on a single corruption cell caused graded
    negative transfer — better in-domain but WORSE than pretrained on every
    motion-blur and low-severity cell.
  - clean_fraction 0.25 (v2: 1/10 uniform): v2's fine-tune cost 0.068 mAP on
    clean images and therefore lost to the pretrained model on every
    low-severity (near-clean) cell.  A larger clean share both regularizes
    toward clean statistics and makes best.pt selection reward keeping them.
  - restored_fraction 0.5 (v2: never): v2 was evaluated (and would be
    deployed) on classically-restored images it never trained on — on salt &
    pepper, enhance->finetuned scored BELOW enhance->pretrained at every
    severity.  Training on the restored domain makes the two repairs additive.

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
import os
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


def _mixture_cells(cfg: Dict) -> list:
    """The 9 (distortion, severity) cells."""
    return [(dt, sv) for dt, spec in cfg["distortions"].items()
            for sv in spec["severities"]]


def _build_one(job) -> None:
    """Worker: generate one training image (module-level for multiprocessing).

    Re-derives the exact corruption from the stable per-image RNG, and for
    restored picks applies the matched classical restorer with the *sampled*
    degradation params — the same non-blind scheme as the val enhancement."""
    (src, dst, dtype, severity, params_spec, restore, seed, image_id,
     gauss_method) = job
    from src.enhancement import (restore_motionblur, restore_noise,
                                 restore_noise_bm3d, restore_saltpepper)

    clean = np.array(Image.open(src).convert("RGB"))
    if dtype == "clean":
        Image.fromarray(clean).save(dst)
        return
    out, sampled = apply_distortion(clean, dtype, params_spec,
                                    _rng_for(seed, image_id, dtype, severity))
    if restore:
        if dtype == "gauss_noise":
            out = (restore_noise(out) if gauss_method == "nlm"
                   else restore_noise_bm3d(out, float(np.sqrt(sampled["var"]))))
        elif dtype == "salt_pepper":
            out = restore_saltpepper(out)
        elif dtype == "motion_blur":
            out = restore_motionblur(out, sampled["ksize"], sampled["angle"])
    # PNG: keep training images as lossless as the benchmark's val images
    Image.fromarray(out).save(dst)


def build_yolo_dataset(cfg: Dict, workers: int = None) -> Path:
    """Corrupt (and 50% of the time restore) the train subset on the fly with
    the per-image seeded v3 mixture; write YOLO images+labels+data.yaml with a
    held-out val split for honest best-checkpoint selection.

    Resumable: if data.yaml exists and the image counts match, the existing
    dataset is reused (lets a CPU job prebuild it — BM3D restoration is the
    expensive part — and the GPU training job skip straight to training)."""
    from concurrent.futures import ProcessPoolExecutor

    from pycocotools.coco import COCO

    ds = cfg["dataset"]
    ft = cfg["finetune"]
    seed = cfg["seed"]
    cells = _mixture_cells(cfg)
    val_frac = float(ft.get("val_fraction", 0.1))
    clean_frac = float(ft.get("clean_fraction", 0.25))
    restored_frac = float(ft.get("restored_fraction", 0.5))
    gauss_method = cfg.get("enhancement", {}).get("gauss_method", "bm3d")
    if workers is None:
        workers = int(cfg.get("enhancement", {}).get("workers", 8))
    # respect the SLURM allocation, not the node's core count
    workers = max(1, min(workers, len(os.sched_getaffinity(0))))

    src_dir = Path(cfg["paths"]["coco_root"]) / ds["train_split"]
    wd = _workdir(cfg)
    data_yaml = wd / "data.yaml"

    coco = COCO(ds["ann_instances_train"])
    img_ids = load_subset_ids(ds["train_subset_file"])
    # seeded held-out split (train images only — the benchmark's val2017 subset
    # must stay untouched by training)
    val_rng = np.random.default_rng(seed)
    val_ids = set(val_rng.choice(img_ids, size=int(len(img_ids) * val_frac),
                                 replace=False).tolist())

    # resume check: a completed build records its image count in data.yaml
    if data_yaml.exists():
        with open(data_yaml) as f:
            meta = yaml.safe_load(f)
        n_disk = sum(1 for s in ("train", "val")
                     for _ in (wd / "images" / s).glob("*.png"))
        if meta.get("n_images") == n_disk and n_disk > 0:
            print(f"[finetune] reusing prebuilt YOLO dataset ({n_disk} images) -> {wd}")
            return data_yaml

    # clean slate: a previous build may have used a different subset size or
    # mixture — a stale copy of an image in the other split would leak into
    # best.pt selection
    for sub in ("images", "labels"):
        if (wd / sub).exists():
            shutil.rmtree(wd / sub)
    for split in ("train", "val"):
        (wd / "images" / split).mkdir(parents=True, exist_ok=True)
        (wd / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_kept = {"train": 0, "val": 0}
    mix_counts: Dict[str, int] = {}
    jobs = []
    for image_id in tqdm(img_ids, desc="yolo-labels", leave=False):
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
        # per-image seeded mixture pick (one RNG stream, three draws; the "v3"
        # key decorrelates from the archived v2 picks)
        r = _rng_for(seed, image_id, "mixture", "v3")
        restore = False
        if r.uniform() < clean_frac:
            dtype, severity, params = "clean", None, None
        else:
            dtype, severity = cells[int(r.integers(len(cells)))]
            params = cfg["distortions"][dtype]["severities"][severity]
            restore = bool(r.uniform() < restored_frac)
        split = "val" if image_id in val_ids else "train"
        stem = Path(im["file_name"]).stem
        (wd / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        jobs.append((str(src_dir / im["file_name"]),
                     str(wd / "images" / split / f"{stem}.png"),
                     dtype, severity, params, restore, seed, image_id,
                     gauss_method))
        n_kept[split] += 1
        key = ("clean" if dtype == "clean"
               else f"{dtype}/{severity}" + ("+enh" if restore else ""))
        mix_counts[key] = mix_counts.get(key, 0) + 1

    # image generation fans out per image — BM3D-restored picks cost ~11s each
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for _ in tqdm(ex.map(_build_one, jobs, chunksize=8),
                      total=len(jobs), desc="yolo-images", leave=False):
            pass

    names = get_yolo_model(cfg["yolo"]["weights"]).names   # {0: 'person', ...}
    with open(data_yaml, "w") as f:
        yaml.safe_dump({
            "path": str(wd.resolve()),
            "train": "images/train",
            "val": "images/val",            # held-out split -> honest best.pt selection
            "names": {int(k): v for k, v in names.items()},
            "n_images": n_kept["train"] + n_kept["val"],   # resume marker
        }, f, sort_keys=False)
    print(f"[finetune] YOLO dataset: {n_kept['train']} train / {n_kept['val']} val "
          f"-> {wd}  mixture: {dict(sorted(mix_counts.items()))}")
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
    ap.add_argument("--mode", choices=["build", "train", "eval", "both"], default="both",
                    help="build = only generate the YOLO dataset (CPU-heavy: "
                         "BM3D-restored picks; lets a CPU node prebuild it)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.mode == "build":
        build_yolo_dataset(cfg)
        return
    if args.mode in ("train", "both"):
        train(cfg)
    if args.mode in ("eval", "both"):
        failures = evaluate_finetuned(cfg)
        if failures:
            raise SystemExit(f"[finetune] {len(failures)} cell(s) failed: {failures}")


if __name__ == "__main__":
    main()
