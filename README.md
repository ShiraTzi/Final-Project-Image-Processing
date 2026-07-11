# Final-Project-Image-Processing

**Team:** Yonatan Haba · Shira Tziony

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
| **4 — Fine-tuning** | fine-tune YOLOv8 on a clean+distorted mixture (real GT); re-evaluate on clean/distorted/enhanced | `src.finetune_det` |
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

**What each task does, and how to read its metric** (all metrics are 0–1,
higher is better):

- **Feature matching (ORB)** — the low-level question: does the raw *signal*
  still carry the same local structure? ORB detects corner-like keypoints with
  binary descriptors on the clean image and on the degraded/restored one; the
  **match ratio** is the fraction of clean keypoints that find a good match
  (Hamming distance ≤ 64, cross-checked) in the other image. 1.0 = every clean
  feature survives; 0 = local structure destroyed. Needs no ground truth — the
  clean image itself is the reference.
- **Object detection (YOLOv8n)** — find every object and its bounding box.
  Scored with COCO **mAP@[.5:.95]**: average precision, averaged over 10 IoU
  thresholds (0.50…0.95) and over the 80 classes. Deliberately strict — it
  rewards both *finding* the object and localizing it *tightly*, so it reacts
  to corruption earlier than a loose "did we detect something" measure.
- **Keypoint detection (Keypoint R-CNN)** — locate 17 human body joints per
  person. Scored with **OKS AP**: the same AP machinery, but box-IoU is
  replaced by Object Keypoint Similarity — a scale-normalized distance between
  predicted and true joints. Person-only, so it isolates how corruption
  affects *fine* spatial localization.
- **Panoptic segmentation (Panoptic FPN)** — assign every pixel a class and an
  instance id ("things" like cars *and* "stuff" like sky/road). Scored with
  **PQ = SQ × RQ**: RQ (recognition quality) is the F1 of correctly matched
  segments, SQ (segmentation quality) is the mean IoU of the matched ones — so
  PQ drops both when whole segments are missed and when masks get sloppy.

> Object detection is fine-tuned (Phase 4) as the deep-learning improvement.
> Panoptic FPN runs in a **separate virtualenv** (detectron2 needs a different
> torch) — the pipeline calls it as a subprocess; everything else is automatic.

## Distortions
Three corruptions (main-branch choice), each at **3 severities** (low/med/high),
applied to the fixed subset; deterministic per image (numpy/cv2, seeded via a
stable CRC32 digest so every rerun and every process reproduces the identical
corruption). Distorted and enhanced images are stored as **lossless PNG** — a
JPEG round-trip would re-shape the corruption itself (it partially denoises
Gaussian noise and smears salt-and-pepper impulses). Per image,
`results/metrics/snr_index.csv` records the SNR (dB) **and the exact sampled
degradation parameters** (noise variance / impulse fraction / blur kernel size
and angle) — the logged parameters are what make non-blind restoration possible.

| Distortion | Models | Matched enhancement (Phase 3) |
|---|---|---|
| **Gaussian noise** | sensor / intensity noise | sigma-adaptive Non-Local Means (strength follows the estimated noise level) |
| **Salt-and-pepper** | impulsive pixel corruption | median filter |
| **Motion blur** | camera shake / object motion | non-blind Wiener deconvolution with the logged blur kernel |

## Dataset
COCO **val2017**, a **fixed seeded subset of ~1500 images** (single source of
truth: every variant runs on exactly these image-ids, sharing identical ground
truth). The subset is a seeded random sample **topped up for class coverage**:
after sampling, images are added (in the same shuffled order) until every
category reaches ≥ `val_min_class_instances` GT instances (bounded by its
availability in val2017) — so per-class AP is not dominated by 5-instance
classes. A seeded **1500-image train2017 subset** is used only for fine-tuning.
Only subset images are downloaded (via each image's `coco_url`); panoptic PQ
additionally needs the COCO panoptic GT (~821MB, fetched automatically when the
segmentation task is enabled).

---

## Results

All numbers are on the fixed 1,521-image, class-coverage-balanced COCO val2017
subset. Full per-cell
table: [results/metrics/comparison.md](results/metrics/comparison.md) /
[`comparison.csv`](results/metrics/comparison.csv); long format with every
metric in [`summary_long.csv`](results/metrics/summary_long.csv).

### The headline: two repair strategies, and each wins somewhere else

The project's main result in one figure — the **actual metric values** after
each repair, per distortion × severity cell, one panel per task. Blue = the
damaged score, green/amber = after classical enhancement / fine-tuning, the
dashed line = the clean baseline, and each label is the **share of the damage
that the repair recovered**. Classical enhancement rescues **61–75%** of the
impulse-noise damage for *every* task (and gets stronger as the corruption
gets worse); fine-tuning recovers **42%** of the heavy-Gaussian-noise damage
for the detector; **nothing repairs motion blur** (0–9%):

![recovery per cell](results/figures/recovery_bars.png)

The same story per object class, on the fine-tuning cell (gauss_noise/high):
the fine-tuned detector (amber) beats the distorted model (blue) on **every**
class and recovers most of the clean AP (gray) — e.g. `bus` 0.07 → 0.42,
`microwave` 0.11 → 0.44 — while classical denoising (green) recovers far less:

![per-class AP comparison](results/figures/per_class_ap_gauss_noise_high.png)

### Key findings

- **Degradation tracks SNR monotonically for every task** — each severity step
  lowers SNR and performance together (see the per-SNR curves below).
  Motion blur is the most destructive corruption for localization-heavy tasks
  (keypoints −0.50, ORB −0.80 at high severity) even though its SNR is higher
  than the noise corruptions' — SNR alone does not fully predict task damage.
- **Classical enhancement is corruption-specific.** The median filter
  essentially rescues salt-and-pepper for every task (e.g. detection
  0.073 → 0.274, PQ 0.107 → 0.309, ORB 0.331 → 0.648 at high severity).
  NLM denoising barely helps (and slightly hurts low-severity cells) because it
  smooths away the textures/corners the models and ORB rely on; unsharp masking
  cannot undo motion blur (a genuine deconvolution problem).
- **Fine-tuning recovers in-domain and transfers within the corruption family.**
  Trained only on gauss_noise/high, YOLOv8n more than doubles its mAP on that
  cell (0.091 → 0.202) and improves the other noise-like cells
  (salt_pepper/high +0.076, gauss_noise/med +0.042), but *hurts* motion-blur
  cells — fine-tuning on one corruption does not buy robustness to a different
  corruption family.
- **Enhancement and fine-tuning are complementary:** enhancement wins where the
  corruption is classically invertible (impulse noise), fine-tuning wins where
  it is not (heavy Gaussian noise).

### Clean baselines (Phase 1)

| Task | Metric | Clean baseline |
|---|---|---:|
| Feature matching (ORB) | match ratio | 1.000 |
| Object detection (YOLOv8n) | mAP@[.5:.95] | 0.352 |
| Keypoint detection (Keypoint R-CNN) | OKS AP | 0.657 |
| Panoptic segmentation (Panoptic FPN) | PQ | 0.410 (SQ 0.775 / RQ 0.500) |

All three pretrained models land within ~0.02 of their published full-val2017
scores, which validates the measurement pipeline itself (GT handling, format
conversions, COCOeval) before any distortion enters the picture.

### Degradation per SNR (Phase 2)

Performance vs SNR — one panel per distortion, shared y-axis. Series identity is
fixed across all panels: distorted (blue, solid, ●), enhanced (green, dashed, ■),
fine-tuned (amber, dotted, ▲); the clean baseline is the gray dashed line. In
gauss_noise the fine-tuned line stays nearly flat and crosses above the others
as distortion grows, while in motion_blur it sits below the distorted line
(negative transfer):

![detection vs SNR](results/figures/acc_vs_snr_detection.png)
![features vs SNR](results/figures/acc_vs_snr_features.png)
![keypoints vs SNR](results/figures/acc_vs_snr_keypoints.png)
![segmentation vs SNR](results/figures/acc_vs_snr_segmentation.png)

Per-class AP of the clean detection baseline (strongest classes, with subset
GT counts):

![per-class AP clean](results/figures/per_class_ap_clean.png)

### The full numbers — clean vs distorted vs enhanced vs fine-tuned (Phases 2–4)

Each row is one experimental cell: the same model, the same 1,521 val images
and ground truth — only the image quality changes. **How to read the columns:**

- **SNR (dB)** — measured signal-to-noise ratio of the distorted images vs the
  clean ones (lower = more corrupted).
- **clean** — the task metric on the original images (the baseline; constant
  per task). Metrics: detection = bbox mAP@[.5:.95], features = ORB match
  ratio, keypoints = OKS AP, segmentation = PQ. All 0–1, higher is better.
- **distorted** — the same metric on the corrupted images.
- **enhanced** — the metric after the matched classical restoration (NLM for
  Gaussian noise, median filter for salt & pepper, unsharp for motion blur).
- **finetuned** — the metric of the fine-tuned YOLOv8n (detection only, the DL
  improvement; **trained on gauss_noise/high**, evaluated on all cells).
- **degradation** = distorted − clean (how much the corruption destroyed).
- **recovery (enhance / finetune)** = improved − distorted (how much each
  improvement strategy won back; negative = made it worse).

Example — `salt_pepper/high`: clean mAP 0.352 collapses to 0.073 (−0.280);
the median filter restores it to 0.274 (+0.202), the fine-tuned model to
0.148 (+0.076). Bold marks the notable wins.

| task | distortion | severity | SNR (dB) | clean | distorted | enhanced | finetuned | degradation | recovery (enhance) | recovery (finetune) |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| detection | gauss_noise | low | 19.6 | 0.352 | 0.285 | 0.265 | 0.244 | −0.067 | −0.020 | −0.041 |
| detection | gauss_noise | med | 13.9 | 0.352 | 0.193 | 0.203 | **0.235** | −0.159 | +0.010 | **+0.042** |
| detection | gauss_noise | high | 9.9 | 0.352 | 0.091 | 0.106 | **0.202** | −0.261 | +0.015 | **+0.111** |
| detection | salt_pepper | low | 18.7 | 0.352 | 0.265 | **0.318** | 0.207 | −0.088 | **+0.054** | −0.058 |
| detection | salt_pepper | med | 11.8 | 0.352 | 0.151 | **0.301** | 0.176 | −0.201 | **+0.150** | +0.026 |
| detection | salt_pepper | high | 8.2 | 0.352 | 0.073 | **0.274** | **0.148** | −0.280 | **+0.202** | **+0.076** |
| detection | motion_blur | low | 20.9 | 0.352 | 0.288 | 0.281 | 0.192 | −0.065 | −0.007 | −0.096 |
| detection | motion_blur | med | 17.7 | 0.352 | 0.161 | 0.152 | 0.098 | −0.191 | −0.009 | −0.063 |
| detection | motion_blur | high | 15.6 | 0.352 | 0.056 | 0.054 | 0.034 | −0.296 | −0.001 | −0.022 |
| features | gauss_noise | low | 19.6 | 1.000 | 0.818 | 0.754 | — | −0.182 | −0.064 | — |
| features | gauss_noise | med | 13.9 | 1.000 | 0.721 | 0.707 | — | −0.280 | −0.013 | — |
| features | gauss_noise | high | 9.9 | 1.000 | 0.609 | 0.607 | — | −0.391 | −0.001 | — |
| features | salt_pepper | low | 18.7 | 1.000 | 0.556 | **0.698** | — | −0.444 | **+0.142** | — |
| features | salt_pepper | med | 11.8 | 1.000 | 0.409 | **0.681** | — | −0.591 | **+0.271** | — |
| features | salt_pepper | high | 8.2 | 1.000 | 0.331 | **0.648** | — | −0.669 | **+0.316** | — |
| features | motion_blur | low | 20.9 | 1.000 | 0.641 | 0.672 | — | −0.359 | +0.031 | — |
| features | motion_blur | med | 17.7 | 1.000 | 0.403 | **0.444** | — | −0.598 | **+0.041** | — |
| features | motion_blur | high | 15.6 | 1.000 | 0.203 | **0.252** | — | −0.797 | **+0.050** | — |
| keypoints | gauss_noise | low | 19.6 | 0.657 | 0.570 | 0.500 | — | −0.088 | −0.070 | — |
| keypoints | gauss_noise | med | 13.9 | 0.657 | 0.474 | 0.453 | — | −0.183 | −0.021 | — |
| keypoints | gauss_noise | high | 9.9 | 0.657 | 0.352 | 0.353 | — | −0.306 | +0.001 | — |
| keypoints | salt_pepper | low | 18.7 | 0.657 | 0.557 | 0.582 | — | −0.101 | +0.025 | — |
| keypoints | salt_pepper | med | 11.8 | 0.657 | 0.433 | **0.566** | — | −0.224 | **+0.133** | — |
| keypoints | salt_pepper | high | 8.2 | 0.657 | 0.325 | **0.537** | — | −0.332 | **+0.212** | — |
| keypoints | motion_blur | low | 20.9 | 0.657 | 0.552 | 0.543 | — | −0.105 | −0.009 | — |
| keypoints | motion_blur | med | 17.7 | 0.657 | 0.374 | 0.364 | — | −0.283 | −0.010 | — |
| keypoints | motion_blur | high | 15.6 | 0.657 | 0.163 | 0.155 | — | −0.495 | −0.007 | — |
| segmentation | gauss_noise | low | 19.6 | 0.410 | 0.355 | 0.295 | — | −0.055 | −0.060 | — |
| segmentation | gauss_noise | med | 13.9 | 0.410 | 0.289 | 0.244 | — | −0.120 | −0.046 | — |
| segmentation | gauss_noise | high | 9.9 | 0.410 | 0.201 | 0.185 | — | −0.208 | −0.016 | — |
| segmentation | salt_pepper | low | 18.7 | 0.410 | 0.282 | **0.361** | — | −0.128 | **+0.080** | — |
| segmentation | salt_pepper | med | 11.8 | 0.410 | 0.159 | **0.344** | — | −0.251 | **+0.185** | — |
| segmentation | salt_pepper | high | 8.2 | 0.410 | 0.107 | **0.309** | — | −0.302 | **+0.202** | — |
| segmentation | motion_blur | low | 20.9 | 0.410 | 0.342 | 0.339 | — | −0.068 | −0.003 | — |
| segmentation | motion_blur | med | 17.7 | 0.410 | 0.230 | 0.229 | — | −0.179 | −0.001 | — |
| segmentation | motion_blur | high | 15.6 | 0.410 | 0.115 | 0.116 | — | −0.295 | +0.001 | — |

### Visual examples

Input/output examples — clean / distorted (high severity) / enhanced. These
show *why* the numbers behave as they do: the median filter visibly removes
impulse pixels, while unsharp masking cannot bring back edges that motion blur
smeared away:

![gauss noise grid](results/figures/grid_gauss_noise.png)
![salt & pepper grid](results/figures/grid_salt_pepper.png)
![motion blur grid](results/figures/grid_motion_blur.png)

### Validity: is the 1,521-image subset representative and class-balanced?

The subset is a **seeded random sample of 1,500 images topped up with 21
images for class coverage** (deterministic, same shuffled order), giving
11,129 GT instances. Checks:

- Class distribution matches full val2017 almost exactly: class-proportion
  correlation **r = 0.999**.
- Clean detection mAP (0.352) is within 0.02 of YOLOv8n's published
  full-val2017 mAP (~0.373); keypoints rest on **3,273 person instances**.
- **Every one of the 80 classes has ≥ 20 GT instances** (median 81), except
  `toaster` (9) and `hair drier` (11) — for which the subset already contains
  **100% of their instances in all of val2017**, so no sample can do better.
- All comparisons are **paired** (identical images and GT across every
  variant), so degradation/recovery deltas are not affected by sampling.

With the coverage floor, per-class AP now rests on ≥ 20 objects per class
(the per-class figure still shows the exact n= under each class name).

### Fine-tuning setup (Phase 4)

- Data: 1,500-image train2017 subset, distorted with **gauss_noise/high**
  (same deterministic per-image RNG as the val distortions), **real COCO boxes**
  converted to YOLO labels (no pseudo-labels).
- Training: `yolov8n.pt` continued for 20 epochs, imgsz 640, batch 16 —
  ~6 minutes on one NVIDIA L4. Checkpoint: `models/yolov8_finetuned.pt`
  (the clean baseline model is untouched).
- Evaluation: the fine-tuned detector runs on **all 9 distorted val cells** of
  the 1,521-image subset (held out — the model never sees any val2017 image
  during training, so there is no leakage), filling the `finetuned` column above.

### Do the results serve the project goals?

The course defines four measurable outcomes; each is met with a quantified
answer rather than a demo:

1. **Baseline performance (vs GT)** — met and *validated*: all four baselines
   land at their expected values (mAP 0.352 vs ~0.373 published, OKS AP 0.657,
   PQ 0.410, ORB 1.000 by construction). This matters because every later
   number is a delta against this baseline — if the baseline were off, the
   whole study would measure pipeline bugs instead of robustness.
2. **Performance on distorted images — per distortion, per class, per SNR** —
   met: 36 (task × distortion × severity) cells, all degrading monotonically
   with SNR. Beyond the requirement, the data shows **SNR alone does not
   predict task damage**: motion blur has a *higher* SNR than the noise
   corruptions yet destroys localization tasks the most (ORB −0.80, keypoints
   −0.50) — spatial structure matters more than pixel-wise error energy.
3. **Performance on enhanced images** — met, with a sharp engineering
   conclusion: classical restoration works exactly where the corruption
   matches the filter's model (median ↔ impulse noise: +0.08…+0.32 across all
   four tasks) and is neutral-to-harmful elsewhere (NLM smooths away the
   texture the models need; unsharp masking cannot invert a blur kernel).
   The negative results are reported deliberately — they delimit *when*
   enhancement is worth deploying.
4. **Fine-tuned model on distorted images** — met: +0.111 in-domain (mAP more
   than doubles), positive transfer within the noise family, and *negative*
   transfer to motion blur — evidence that fine-tuning buys corruption-specific
   robustness, not general robustness.

Bottom line: the four phases compose into a **decision matrix** — given a
corruption type, the results say whether to restore classically, fine-tune the
model, or fix the capture process (for blur, neither software repair works) —
which is exactly the trade-off the assignment asks the project to expose.

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
docs/                      # archived v1 result tables + the course reference pipeline
slides/                    # final presentation (PPTX + PDF)
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
- **Segmentation** (Panoptic FPN): **PQ / SQ / RQ** via panopticapi (things & stuff),
  plus the PQ_things / PQ_stuff split per cell.
- Tables report **degradation** (distorted − clean) and **recovery** (enhanced/fine-tuned − distorted).

> **Metric caveat (ORB match ratio).** The features score measures similarity
> to the *clean image's* ORB features, not task utility: any enhancement that
> alters texture — smoothing in particular — is penalized *by construction*,
> even when the same enhanced images improve every GT-scored task. (Salt &
> pepper shows this cleanly: the median-filtered images improve detection,
> keypoints and segmentation, while their ORB score stays well below the
> clean-reference 1.0.) Its clean baseline of exactly 1.0 is also "free", so
> degradation magnitudes are not directly comparable with the GT-based tasks.
> Cross-task conclusions in this report therefore lean on the three GT-based
> metrics; ORB answers the narrower low-level question "does the raw signal
> still carry the same local structure?"

## How fine-tuning works
`src/finetune_det.py` corrupts the train2017 subset on the fly with a
**per-image seeded mixture** of clean + all 9 (distortion × severity) cells,
writes the **real** COCO boxes as YOLO labels, and continues training the
**pretrained** `yolov8n.pt` on that data (AdamW, lr0 pinned to 1e-4, cosine
schedule — `optimizer=auto` silently overrides lr0, so it is set explicitly).
A seeded 10% split of the *train* subset is held out as the YOLO val set so
best-checkpoint selection is honest (selecting on the training images selects
for memorization). The result is saved separately as
`models/yolov8_finetuned.pt` (the clean baseline model is untouched) and
evaluated on the **clean** val subset (does robustness cost clean accuracy?),
all **nine distorted** val cells, and all **nine enhanced** val cells (is
classical restoration on top of fine-tuning additive?) — all held out, the
model never sees a val2017 image during training.

Why the mixture matters: an earlier iteration fine-tuned on a *single* cell
(gauss_noise/high only) and showed textbook graded negative transfer — it more
than doubled in-domain mAP but *lost* to the pretrained model on every
motion-blur cell and every low-severity cell (recovery −0.02…−0.10; archived
in [docs/archive/](docs/archive/)). Training on the mixture keeps every
evaluation cell in-domain.

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
