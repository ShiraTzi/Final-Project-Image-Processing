# Final-Project-Image-Processing

Robustness benchmark for image-processing / vision methods under image
distortion, on the **COCO** dataset. For each task we measure a **clean
baseline**, the **degradation** under three corruptions at multiple severities,
and the **recovery** from two improvement strategies (classical **enhancement**
and model **fine-tuning**), reported per class and as a function of **SNR**.

---

## 1. Dataset
We use the COCO **val2017** split as the benchmark, with a **fixed, seeded
subset of 500 images** (the single source of truth: every variant —
clean / distorted / enhanced / fine-tuned — runs on exactly these image-ids,
so all results share identical ground truth). A separate seeded **1500-image
train2017 subset** is used only for fine-tuning. Only the subset images are
downloaded (via each image's `coco_url`), so no multi-GB zips are needed.

## 2. Tasks and Models
Three tasks spanning low-level and high-level vision, each with a representative
model/algorithm:

| Task | Model / algorithm | Stack | Metric |
|---|---|---|---|
| Feature / corner detection (low-level) | **ORB** | OpenCV (classical) | match ratio vs clean |
| Object detection (high-level) | **YOLOv8n** (COCO-pretrained) | ultralytics | COCOeval bbox mAP |
| Keypoint detection (high-level) | **Keypoint R-CNN** ResNet50-FPN (COCO-pretrained) | torchvision | COCOeval keypoints (OKS) |

> Panoptic/instance segmentation is **out of scope** for this implementation
> (kept to three tasks); it can be added later as a 4th task (e.g. Mask R-CNN)
> without changing the pipeline structure.

## 3. Distortions
Three corruption types, each at **3 severities** (low / med / high), applied to
the fixed subset. They are implemented directly in numpy/cv2 (equivalent to the
common albumentations ops) so the benchmark is deterministic and
version-independent:

1. **Gaussian noise** — sensor / random intensity noise.
2. **Severe JPEG compression** — heavy block/ringing artifacts from low-quality encoding.
3. **Low light** — strong global darkening.

For every distorted image we record the **SNR (dB)** in `results/metrics/snr_index.csv`.

> The matched enhancement methods (Part 5) are tied to these three distortions.
> Salt-and-pepper and motion blur are easy optional extras — add them under
> `distortions:` in the config and provide a matching restorer.

## 4. Benchmarking Workflow
For each (task, variant):
1. **Baseline** — measure on clean images.
2. **Distortion** — apply each corruption at each severity, measure degradation.
3. **Compare** — `pycocotools` COCOeval for the high-level tasks (mAP, mAP@.50,
   mAP@.75, per-class AP, small/med/large), ORB match ratio for the low-level task.

Per-severity SNR is recorded, producing **accuracy-vs-SNR curves** in addition
to per-class results.

## 5. Improvement Strategies
Two approaches (the assignment asks for both):
1. **Enhancement (pre-processing)** — one matched restorer per distortion:
   Non-Local Means + bilateral (gaussian), Y-channel bilateral (severe JPEG),
   gamma + CLAHE (low light). Enhanced images flow through the same
   inference→eval path to measure recovery.
2. **Fine-tuning** — **YOLOv8** is fine-tuned on the *distorted* train2017
   subset using **real COCO ground truth** (converted to YOLO labels), then
   evaluated on the *distorted val* subset (held out — no leakage). See
   [How fine-tuning works](#how-fine-tuning-works).

## 6. Evaluation and Robustness Analysis
- **High-level tasks:** COCO metrics (mAP, mAP@.50, mAP@.75) + absolute
  degradation, per-class sensitivity, and small-vs-large object effects.
- **Low-level task:** ORB match ratio vs SNR.
- Outputs: per-class AP bars, accuracy-vs-SNR curves, and a wide
  clean / distorted / enhanced / fine-tuned comparison table with
  degradation and recovery deltas.

---

## Repository structure
```
configs/config.yaml        # single source of truth: paths, seed, subset sizes,
                           #   distortions×severities, tasks, yolo + finetune params
requirements.txt
src/
  config.py                # config loader + variant/path helpers
  data.py                  # download annotations + subset images; seeded splits
  distortions.py           # gaussian / severe-JPEG / low-light + compute_snr
  enhancement.py           # restore_noise / restore_jpeg / restore_lowlight
  models.py                # YOLOv8 loader (+COCO80<->91 map), Keypoint R-CNN, ORB helpers
  inference.py             # (task, variant) -> COCO-format prediction JSON
  metrics.py               # COCOeval (bbox/keypoints) + ORB match ratio
  tables.py                # comparison / degradation / recovery tables
  visualize.py             # acc-vs-SNR curves, per-class AP bars, image grids
  finetune_det.py          # fine-tune YOLOv8 on distorted train2017 (real GT)
scripts/run_pipeline.py    # resumable orchestrator over all stages
slurm/                     # pipeline / inference / finetune sbatch jobs
results/{preds,metrics,figures}/   # generated outputs
data/                      # coco/ images+annotations, distorted/, enhanced/ (gitignored)
```

## Environment setup
GPU is required for the detector inference and fine-tuning (run those on a GPU
node / via SLURM). CPU is fine for distortion, enhancement, and ORB.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -r requirements.txt

# sanity check (prints True on a GPU node)
python -c "import torch,torchvision,cv2,ultralytics,pycocotools; print(torch.cuda.is_available())"
```
> Note: the committed `venv.zip` is a Windows environment and is **not** used on
> Linux — always build a fresh `.venv` as above.

## How to run

### One command (resumable)
```bash
python scripts/run_pipeline.py            # data -> distort -> enhance -> infer -> eval -> finetune -> report
python scripts/run_pipeline.py --skip-finetune
python scripts/run_pipeline.py --only report      # rebuild tables/figures only
python scripts/run_pipeline.py --force            # recompute even if outputs exist
```
Every stage **skips work whose output already exists**, so it is safe to re-run
and resume after interruption.

### On the cluster (SLURM)
```bash
sbatch slurm/pipeline.sbatch              # full pipeline on one GPU
sbatch slurm/inference.sbatch             # infer + eval + report only
sbatch slurm/finetune.sbatch              # fine-tune YOLOv8 + evaluate
```

### Stage by stage (manual)
```bash
python -m src.data                                   # download + build subsets
python -m src.distortions                            # distorted sets + snr_index.csv
python -m src.enhancement                            # enhanced sets
python -m src.inference --task detection --variant clean
python -m src.inference --task keypoints --variant distorted --dtype gauss_noise --severity high
python -m src.metrics   --task orb       --variant distorted --dtype low_light  --severity high
python -m src.finetune_det --mode both               # train + evaluate fine-tuned YOLO
python -m src.tables
python -m src.visualize
```

## Configuration
All knobs live in [configs/config.yaml](configs/config.yaml):
- `dataset.val_subset_size` / `train_subset_size`, `seed`
- `distortions:` — the three corruptions and their 3 severities
- `tasks:` — `[orb, detection, keypoints]`
- `yolo.weights` — `yolov8n.pt` (bump to `yolov8s/m.pt` for more accuracy)
- `finetune:` — `epochs`, `batch_size`, `imgsz`, and which `distortion`/`severity`
  to fine-tune on

## Outputs
- `results/preds/{task}__{variant}.json` — COCO-format predictions
- `results/metrics/{task}__{variant}.json` — per-(task,variant) metrics
- `results/metrics/snr_index.csv` — per-image SNR for every distortion/severity
- `results/metrics/summary_long.csv`, `comparison.csv`, `comparison.md` — aggregated tables
- `results/figures/*.png` — acc-vs-SNR curves, per-class AP bars, image grids

## How fine-tuning works
`src/finetune_det.py` (1) distorts the train2017 subset on the fly and writes
the **real** COCO boxes as YOLO labels, then (2) continues training the
**pretrained** `yolov8n.pt` on that data (`model.train(...)`) — same
architecture, updated weights (transfer learning, not from scratch). The result
is saved as a **separate** checkpoint `models/yolov8_finetuned.pt`, leaving the
clean baseline model untouched. It is then evaluated on the *distorted val*
subset via COCOeval. Because yolov8n is tiny and starts pretrained, this is
light — roughly a few-to-15 minutes on one GPU. Tune cost via `finetune.epochs`,
`imgsz`, and `dataset.train_subset_size`.

---

## Committing and pushing
This repo is the submission, so keep it up to date. Generated data/outputs and
environments are gitignored (`data/`, `results/preds`, `results/figures`,
`.venv/`, `*.zip`, `*.pt`); commit code, config, and the report.

```bash
git add -A
git commit -m "your message"
git push -u origin <branch>      # e.g. ohads/phase0
```

If `git push` reports `could not read Username for 'https://github.com'`, the
machine has no stored GitHub credentials. Authenticate with one of:
```bash
# (a) GitHub CLI
gh auth login

# (b) Personal access token (Contents: write) over HTTPS
git push https://<TOKEN>@github.com/ShiraTzi/Final-Project-Image-Processing.git <branch>

# (c) SSH — after adding an SSH key to your GitHub account
git remote set-url origin git@github.com:ShiraTzi/Final-Project-Image-Processing.git
git push -u origin <branch>
```

## Deliverables
- This README as the detailed report (choices, methods, results, instructions).
- Result tables (per class and per SNR) and figures under `results/`.
- Code, config, SLURM scripts, and saved fine-tuned checkpoint.
- Final presentation (PPT/PDF) summarizing the README.
- Team registration (names, emails) and the GitHub repository URL.
