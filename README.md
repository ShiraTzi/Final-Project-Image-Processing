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
Four tasks spanning low-level and high-level vision:

| Task | Model / algorithm | Stack | Metric |
|---|---|---|---|
| Feature matching (**low-level**) | **ORB** + BFMatcher (Hamming, cross-check) | OpenCV | match ratio vs clean (good matches / clean keypoints) |
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

## Results

All numbers are on the fixed 500-image COCO val2017 subset. Full per-cell
table: [results/metrics/comparison.md](results/metrics/comparison.md) /
[`comparison.csv`](results/metrics/comparison.csv); long format with every
metric in [`summary_long.csv`](results/metrics/summary_long.csv).

### Clean baselines (Phase 1)

| Task | Metric | Clean baseline |
|---|---|---:|
| Feature matching (ORB) | match ratio | 1.000 |
| Object detection (YOLOv8n) | mAP@[.5:.95] | 0.363 |
| Keypoint detection (Keypoint R-CNN) | OKS AP | 0.649 |
| Panoptic segmentation (Panoptic FPN) | PQ | 0.400 (SQ 0.744 / RQ 0.489) |

### Clean vs distorted vs enhanced vs fine-tuned (Phases 2–4)

`degradation` = distorted − clean; `recovery_*` = improved − distorted.
Fine-tuning applies to the detection task only (the DL improvement); it was
**trained on gauss_noise/high** and evaluated on **all** distorted cells.

| task | distortion | severity | SNR (dB) | clean | distorted | enhanced | finetuned | degradation | recovery (enhance) | recovery (finetune) |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| detection | gauss_noise | low | 19.6 | 0.363 | 0.301 | 0.288 | 0.266 | −0.062 | −0.013 | −0.035 |
| detection | gauss_noise | med | 13.9 | 0.363 | 0.207 | 0.215 | 0.254 | −0.156 | +0.008 | **+0.047** |
| detection | gauss_noise | high | 10.1 | 0.363 | 0.102 | 0.117 | **0.218** | −0.260 | +0.014 | **+0.115** |
| detection | salt_pepper | low | 18.8 | 0.363 | 0.278 | 0.332 | 0.221 | −0.085 | +0.054 | −0.057 |
| detection | salt_pepper | med | 11.9 | 0.363 | 0.158 | **0.314** | 0.192 | −0.205 | **+0.156** | +0.035 |
| detection | salt_pepper | high | 8.3 | 0.363 | 0.077 | **0.288** | 0.163 | −0.286 | **+0.211** | +0.087 |
| detection | motion_blur | low | 21.1 | 0.363 | 0.300 | 0.292 | 0.203 | −0.062 | −0.009 | −0.098 |
| detection | motion_blur | med | 18.0 | 0.363 | 0.171 | 0.161 | 0.114 | −0.191 | −0.011 | −0.058 |
| detection | motion_blur | high | 15.9 | 0.363 | 0.065 | 0.061 | 0.044 | −0.297 | −0.004 | −0.021 |
| features | gauss_noise | low | 19.6 | 1.000 | 0.817 | 0.749 | — | −0.183 | −0.068 | — |
| features | gauss_noise | med | 13.9 | 1.000 | 0.718 | 0.704 | — | −0.282 | −0.014 | — |
| features | gauss_noise | high | 10.1 | 1.000 | 0.608 | 0.607 | — | −0.392 | −0.001 | — |
| features | salt_pepper | low | 18.8 | 1.000 | 0.553 | 0.699 | — | −0.447 | **+0.146** | — |
| features | salt_pepper | med | 11.9 | 1.000 | 0.408 | 0.682 | — | −0.592 | **+0.274** | — |
| features | salt_pepper | high | 8.3 | 1.000 | 0.330 | 0.649 | — | −0.670 | **+0.319** | — |
| features | motion_blur | low | 21.1 | 1.000 | 0.643 | 0.674 | — | −0.357 | +0.031 | — |
| features | motion_blur | med | 18.0 | 1.000 | 0.401 | 0.444 | — | −0.599 | +0.043 | — |
| features | motion_blur | high | 15.9 | 1.000 | 0.200 | 0.250 | — | −0.800 | +0.050 | — |
| keypoints | gauss_noise | low | 19.6 | 0.649 | 0.556 | 0.490 | — | −0.093 | −0.066 | — |
| keypoints | gauss_noise | med | 13.9 | 0.649 | 0.460 | 0.460 | — | −0.190 | −0.000 | — |
| keypoints | gauss_noise | high | 10.1 | 0.649 | 0.343 | 0.340 | — | −0.307 | −0.003 | — |
| keypoints | salt_pepper | low | 18.8 | 0.649 | 0.553 | 0.567 | — | −0.097 | +0.014 | — |
| keypoints | salt_pepper | med | 11.9 | 0.649 | 0.430 | **0.551** | — | −0.219 | **+0.121** | — |
| keypoints | salt_pepper | high | 8.3 | 0.649 | 0.317 | **0.515** | — | −0.332 | **+0.198** | — |
| keypoints | motion_blur | low | 21.1 | 0.649 | 0.546 | 0.541 | — | −0.103 | −0.005 | — |
| keypoints | motion_blur | med | 18.0 | 0.649 | 0.376 | 0.368 | — | −0.273 | −0.008 | — |
| keypoints | motion_blur | high | 15.9 | 0.649 | 0.167 | 0.161 | — | −0.482 | −0.007 | — |
| segmentation | gauss_noise | low | 19.6 | 0.400 | 0.343 | 0.291 | — | −0.057 | −0.052 | — |
| segmentation | gauss_noise | med | 13.9 | 0.400 | 0.281 | 0.239 | — | −0.120 | −0.041 | — |
| segmentation | gauss_noise | high | 10.1 | 0.400 | 0.196 | 0.182 | — | −0.204 | −0.014 | — |
| segmentation | salt_pepper | low | 18.8 | 0.400 | 0.284 | **0.359** | — | −0.116 | **+0.075** | — |
| segmentation | salt_pepper | med | 11.9 | 0.400 | 0.158 | **0.336** | — | −0.242 | **+0.178** | — |
| segmentation | salt_pepper | high | 8.3 | 0.400 | 0.111 | **0.303** | — | −0.289 | **+0.192** | — |
| segmentation | motion_blur | low | 21.1 | 0.400 | 0.335 | 0.332 | — | −0.065 | −0.003 | — |
| segmentation | motion_blur | med | 18.0 | 0.400 | 0.228 | 0.232 | — | −0.172 | +0.003 | — |
| segmentation | motion_blur | high | 15.9 | 0.400 | 0.116 | 0.118 | — | −0.284 | +0.003 | — |

### Key findings

- **Degradation tracks SNR monotonically for every task** — each severity step
  lowers SNR and performance together (see the per-SNR curves below).
  Motion blur is the most destructive corruption for localization-heavy tasks
  (keypoints −0.48, ORB −0.80 at high severity) even though its SNR is higher
  than the noise corruptions' — SNR alone does not fully predict task damage.
- **Classical enhancement is corruption-specific.** The median filter
  essentially rescues salt-and-pepper for every task (e.g. detection
  0.077 → 0.288, PQ 0.111 → 0.303, ORB 0.330 → 0.649 at high severity).
  NLM denoising barely helps (and slightly hurts low-severity cells) because it
  smooths away the textures/corners the models and ORB rely on; unsharp masking
  cannot undo motion blur (a genuine deconvolution problem).
- **Fine-tuning recovers in-domain and transfers within the corruption family.**
  Trained only on gauss_noise/high, YOLOv8n more than doubles its mAP on that
  cell (0.102 → 0.218) and improves the other noise-like cells
  (salt_pepper/high +0.087, gauss_noise/med +0.047), but *hurts* motion-blur
  cells — fine-tuning on one corruption does not buy robustness to a different
  corruption family.
- **Enhancement and fine-tuning are complementary:** enhancement wins where the
  corruption is classically invertible (impulse noise), fine-tuning wins where
  it is not (heavy Gaussian noise).

### Figures

Performance vs SNR (one line per distortion; distorted solid, enhanced dashed,
fine-tuned dotted; clean baseline as horizontal line):

![detection vs SNR](results/figures/acc_vs_snr_detection.png)
![features vs SNR](results/figures/acc_vs_snr_features.png)
![keypoints vs SNR](results/figures/acc_vs_snr_keypoints.png)
![segmentation vs SNR](results/figures/acc_vs_snr_segmentation.png)

Per-class AP (clean baseline, and clean vs distorted vs enhanced vs fine-tuned
on the fine-tune cell gauss_noise/high):

![per-class AP clean](results/figures/per_class_ap_clean.png)
![per-class AP comparison](results/figures/per_class_ap_gauss_noise_high.png)

Input/output examples — clean / distorted (high severity) / enhanced:

![gauss noise grid](results/figures/grid_gauss_noise.png)
![salt & pepper grid](results/figures/grid_salt_pepper.png)
![motion blur grid](results/figures/grid_motion_blur.png)

### Fine-tuning setup (Phase 4)

- Data: 1,500-image train2017 subset, distorted with **gauss_noise/high**
  (same deterministic per-image RNG as the val distortions), **real COCO boxes**
  converted to YOLO labels (no pseudo-labels).
- Training: `yolov8n.pt` continued for 20 epochs, imgsz 640, batch 16 —
  ~6 minutes on one NVIDIA L4 (SLURM job, 11:40 total including the 9-cell
  evaluation). Checkpoint: `models/yolov8_finetuned.pt` (the clean baseline
  model is untouched).
- Evaluation: the fine-tuned detector runs on **all 9 distorted val cells**
  (held out — no leakage), filling the `finetuned` column above.

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
python -m src.metrics   --task features     --variant clean   # ORB (CPU, no preds needed)
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
                           #   distortions×severities, tasks, orb/yolo/finetune/segmentation params
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
- **Features** (ORB): **match ratio** — good BFMatcher(Hamming, cross-check)
  matches (distance ≤ 64) between clean and variant descriptors / clean
  keypoints; clean vs clean = 1.0 by construction.
- **Detection** (YOLOv8): COCOeval **bbox mAP**, mAP@.50, mAP@.75, per-class AP, small/med/large.
- **Keypoints** (Keypoint R-CNN): COCOeval **keypoints** (OKS) AP.
- **Segmentation** (Panoptic FPN): **PQ / SQ / RQ** via panopticapi (things & stuff).
- Tables report **degradation** (distorted − clean) and **recovery** (enhanced/fine-tuned − distorted).

## How fine-tuning works
`src/finetune_det.py` distorts the train2017 subset on the fly, writes the
**real** COCO boxes as YOLO labels, and continues training the **pretrained**
`yolov8n.pt` on that data (transfer learning — same architecture, updated
weights). The result is saved separately as `models/yolov8_finetuned.pt`
(the clean baseline model is untouched) and evaluated on **all nine** distorted
val cells (held out — no leakage), so the comparison table shows both in-domain
recovery and cross-distortion generalization. yolov8n is tiny and starts
pretrained, so this is light (~6 minutes on one L4 GPU). Tune via
`finetune.epochs`, `imgsz`, `dataset.train_subset_size`.

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
