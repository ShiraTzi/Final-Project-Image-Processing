# Final-Project-Image-Processing

Robustness benchmark for image-processing / vision methods under image
distortion, on the **COCO** dataset. For each task we measure a **clean
baseline**, the **degradation** under three corruptions at multiple severities,
and the **recovery** from two improvement strategies (classical **enhancement**
and model **fine-tuning**), reported per class / category and as a function of
**SNR**.

---

## Course-project pipeline (phases)

The project follows the course's required workflow. Each phase has a single
runnable entry point (see [How to run each phase](#how-to-run-each-phase)).

| Phase | What it does | Entry point |
|---|---|---|
| **0 — Setup & Data** | build the two virtualenvs; download COCO annotations + subset images (+ panoptic GT) | `src.data` |
| **1 — Clean baseline** | run all models on clean images, compute baseline metrics | `--only infer eval` (clean) |
| **2 — Distortion** | generate distorted images (3 corruptions × 3 severities) + SNR; measure degradation | `src.distortions` → infer → eval |
| **3 — Enhancement** | restore distorted images (matched denoise/deblur); measure recovery | `src.enhancement` → infer → eval |
| **4 — Fine-tuning** | fine-tune YOLOv8 on distorted data (real GT); re-evaluate | `src.finetune_det` |
| **5 — Report** | comparison tables, per-class plots, accuracy-vs-SNR curves | `src.tables`, `src.visualize` |

```
data ──► distort ──► enhance ──┐
  │          │          │       ├─► infer ─► eval ─► tables/figures
  └──────────┴──────────┴───────┘            ▲
                          finetune (YOLOv8) ──┘
```

## Tasks and models
Three tasks spanning low-level and high-level vision (the main-branch task set):

| Task | Model / algorithm | Stack | Metric |
|---|---|---|---|
| Object detection | **YOLOv8n** (COCO-pretrained) | ultralytics | COCOeval bbox mAP |
| Keypoint detection | **Keypoint R-CNN** ResNet50-FPN (COCO-pretrained) | torchvision | COCOeval keypoints (OKS) |
| Panoptic segmentation | **Panoptic FPN** R50 (COCO-pretrained) | detectron2 | Panoptic Quality (PQ/SQ/RQ) |

> Object detection is fine-tuned (Phase 4) as the deep-learning improvement.
> Panoptic FPN runs in a **separate virtualenv** (detectron2 needs a different
> torch) — the pipeline calls it as a subprocess; everything else is automatic.

## Distortions
Three corruptions (main-branch choice), each at **3 severities** (low/med/high),
applied to the fixed subset; deterministic per image (numpy/cv2). SNR (dB) is
recorded per image in `results/metrics/snr_index.csv`.

| Distortion | Models | Matched enhancement (Phase 3) |
|---|---|---|
| **Gaussian noise** | sensor / intensity noise | Non-Local Means + bilateral |
| **Salt-and-pepper** | impulsive pixel corruption | median filter |
| **Motion blur** | camera shake / object motion | unsharp masking (deblur proxy) |

## Dataset
COCO **val2017**, a **fixed seeded subset of 500 images** (single source of
truth: every variant runs on exactly these image-ids, sharing identical ground
truth). A seeded **1500-image train2017 subset** is used only for fine-tuning.
Only subset images are downloaded (via each image's `coco_url`); panoptic PQ
additionally needs the COCO panoptic GT (~821MB, fetched automatically when the
segmentation task is enabled).

---

## Environments
Two virtualenvs are required because detectron2 needs an older torch than the
main stack:

```bash
# (1) main env — detection (YOLOv8), keypoints (torchvision), distortions, eval, plots
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt
deactivate

# (2) detectron2 env — panoptic segmentation only
python3 -m venv .venv-det
source .venv-det/bin/activate
pip install -U pip wheel
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install "numpy<2" pyyaml tqdm pillow opencv-python pycocotools ninja
pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'
pip install 'git+https://github.com/cocodataset/panopticapi.git'
deactivate
```
Sanity checks (run on a **GPU node** for the `cuda` checks):
```bash
.venv/bin/python      -c "import torch,torchvision,cv2,ultralytics,pycocotools; print(torch.cuda.is_available())"
.venv-det/bin/python  -c "import torch,detectron2; from panopticapi.evaluation import pq_compute; print(torch.cuda.is_available())"
```
> The committed `venv.zip` is a Windows env and is **not** used on Linux.
> The detectron2 env builds from source; for GPU support build it on a node that
> has the CUDA toolkit (`nvcc`).

---

## How to run each phase

The orchestrator `scripts/run_pipeline.py` is **resumable** (skips work whose
output already exists; add `--force` to recompute). Phases map to stages:
`data, distort, enhance, infer, eval, finetune, report`.

### Everything at once
```bash
python scripts/run_pipeline.py                      # phases 0–5
sbatch slurm/pipeline.sbatch                         # same, on a GPU node
```

### Phase 0 — Setup & Data
```bash
python -m src.data                                   # annotations + subset images (+ panoptic GT)
# (env setup as above)
```

### Phase 1 — Clean baseline
```bash
python scripts/run_pipeline.py --only infer eval     # runs clean first (+ all variants)
# or just the clean models manually:
python -m src.inference --task detection --variant clean
python -m src.inference --task keypoints --variant clean
python -m src.metrics   --task segmentation --variant clean   # detectron2 subprocess
python -m src.metrics   --task detection    --variant clean
python -m src.metrics   --task keypoints    --variant clean
```

### Phase 2 — Distortion
```bash
python -m src.distortions                            # writes data/distorted/** + snr_index.csv
python -m src.inference --task detection --variant distorted --dtype motion_blur --severity high
python -m src.metrics   --task detection --variant distorted --dtype motion_blur --severity high
```

### Phase 3 — Enhancement
```bash
python -m src.enhancement                            # writes data/enhanced/**
python -m src.inference --task detection --variant enhanced --dtype gauss_noise --severity high
python -m src.metrics   --task detection --variant enhanced --dtype gauss_noise --severity high
```

### Phase 4 — Fine-tuning (YOLOv8)
```bash
python -m src.finetune_det --mode both               # train on distorted train2017 + evaluate
sbatch slurm/finetune.sbatch                          # on a GPU node
```

### Phase 5 — Report
```bash
python -m src.tables                                 # comparison.csv / comparison.md / summary_long.csv
python -m src.visualize                              # acc-vs-SNR curves, per-class AP bars, image grids
python scripts/run_pipeline.py --only report
```

---

## Repository structure
```
configs/config.yaml        # single source of truth: paths, seed, subset sizes,
                           #   distortions×severities, tasks, yolo/finetune/segmentation params
requirements.txt           # main venv deps
src/
  config.py                # config loader + variant/path helpers
  data.py                  # download annotations (+panoptic) + subset images; seeded splits
  distortions.py           # gaussian / salt-pepper / motion-blur + compute_snr
  enhancement.py           # NLM+bilateral / median / unsharp restorers
  models.py                # YOLOv8 loader (+COCO80<->91 map), Keypoint R-CNN loader
  inference.py             # detection (YOLO) + keypoints -> COCO prediction JSON
  segmentation.py          # Panoptic FPN inference + PQ  (runs under .venv-det)
  metrics.py               # COCOeval (bbox/keypoints); shells out for segmentation PQ
  tables.py                # comparison / degradation / recovery tables
  visualize.py             # acc-vs-SNR curves, per-class AP bars, image grids
  finetune_det.py          # fine-tune YOLOv8 on distorted train2017 (real GT)
scripts/run_pipeline.py    # resumable orchestrator over all phases
slurm/                     # pipeline / inference / finetune sbatch jobs
results/{preds,metrics,figures}/   # generated outputs
data/                      # coco/ images+annotations, distorted/, enhanced/ (gitignored)
```

## Configuration
All knobs live in [configs/config.yaml](configs/config.yaml):
- `dataset.val_subset_size` / `train_subset_size`, `seed`
- `distortions:` — the three corruptions × 3 severities
- `tasks:` — `[detection, keypoints, segmentation]`
- `yolo.weights` — `yolov8n.pt` (bump to `yolov8s/m.pt` for accuracy)
- `finetune:` — epochs / batch / imgsz / which distortion to fine-tune on
- `segmentation:` — detectron2 venv python path + Panoptic FPN config

## Outputs
- `results/preds/{task}__{variant}.json` — predictions (segmentation also writes PNG masks)
- `results/metrics/{task}__{variant}.json` — per-(task,variant) metrics
- `results/metrics/snr_index.csv` — per-image SNR for every distortion/severity
- `results/metrics/summary_long.csv`, `comparison.csv`, `comparison.md` — aggregated tables
- `results/figures/*.png` — acc-vs-SNR curves, per-class AP bars, image grids

## Metrics
- **Detection** (YOLOv8): COCOeval **bbox mAP**, mAP@.50, mAP@.75, per-class AP, small/med/large.
- **Keypoints** (Keypoint R-CNN): COCOeval **keypoints** (OKS) AP.
- **Segmentation** (Panoptic FPN): **PQ / SQ / RQ** via panopticapi (things & stuff).
- Tables report **degradation** (distorted − clean) and **recovery** (enhanced/fine-tuned − distorted).

## How fine-tuning works
`src/finetune_det.py` distorts the train2017 subset on the fly, writes the
**real** COCO boxes as YOLO labels, and continues training the **pretrained**
`yolov8n.pt` on that data (transfer learning — same architecture, updated
weights). The result is saved separately as `models/yolov8_finetuned.pt`
(the clean baseline model is untouched) and evaluated on the *distorted val*
subset (held out — no leakage). yolov8n is tiny and starts pretrained, so this
is light (~minutes on one GPU). Tune via `finetune.epochs`, `imgsz`,
`dataset.train_subset_size`.

---

## Committing and pushing
Generated data/outputs and environments are gitignored (`data/`,
`results/preds`, `results/figures`, `.venv/`, `.venv-det/`, `*.zip`, `*.pt`).

```bash
git add -A
git commit -m "your message"
git push -u origin <branch>      # e.g. ohads/phase0
```
If `git push` reports `could not read Username for 'https://github.com'`, the
machine has no stored credentials. Authenticate with one of:
```bash
gh auth login                                                # (a) GitHub CLI
git push https://<TOKEN>@github.com/ShiraTzi/Final-Project-Image-Processing.git <branch>   # (b) PAT
git remote set-url origin git@github.com:ShiraTzi/Final-Project-Image-Processing.git       # (c) SSH
```

## Deliverables
- This README as the detailed report (choices, methods, metrics, run instructions).
- Result tables (per class and per SNR) and figures under `results/`.
- Code, config, SLURM scripts, fine-tuned checkpoint.
- Final presentation (PPT/PDF) summarizing the README.
- Team registration (names, emails) and the GitHub repository URL.
