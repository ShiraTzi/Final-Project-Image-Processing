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

### The headline: matched repairs recover most of the damage — when the repair actually matches

The project's main result in one figure — the **actual metric values** after
each repair, per distortion × severity cell, one panel per task. Blue = the
damaged score, green/amber = after classical enhancement / fine-tuning, the
dashed line = the clean baseline, and each label is the **share of the damage
that the repair recovered**. The median filter rescues **80–91%** of the
impulse-noise damage for detection (74–86% for segmentation); non-blind
Wiener deconvolution — possible because the benchmark logs each image's blur
kernel — recovers **25–68%** of the motion-blur damage that a blind sharpening
filter couldn't touch (0–9% in the archived v1 run); and the mixture-trained
detector recovers **35%** of the heavy-Gaussian-noise damage, the corruption
classical filtering repairs worst:

![recovery per cell](results/figures/recovery_bars.png)

The same story per object class, on the gauss_noise/high cell: the fine-tuned
detector (amber) beats the distorted model (blue) across the board and
recovers a large share of the clean AP (gray), while classical denoising
(green) recovers less:

![per-class AP comparison](results/figures/per_class_ap_gauss_noise_high.png)

### Key findings

- **Degradation tracks SNR monotonically for every task** — each severity step
  lowers SNR and performance together (see the per-SNR curves below).
  Motion blur is the most destructive corruption for localization-heavy tasks
  (keypoints −0.50, ORB −0.80 at high severity) even though its SNR is *higher*
  than the noise corruptions' — SNR alone does not fully predict task damage.
- **Classical enhancement works exactly where the corruption matches the
  filter's model.** The median filter essentially rescues salt-and-pepper for
  every task (detection 0.079 → 0.321, PQ 0.105 → 0.348, OKS AP 0.348 → 0.598
  at high severity). Wiener deconvolution with the *logged* kernel repairs
  motion blur (detection 0.157 → 0.257 at med severity; ORB 0.20 → 0.48 at
  high) — but this is **known-degradation restoration**: a blind kernel guess
  measured *worse than no restoration at all*. Gaussian noise is the negative
  result: sigma-adaptive NLM helps only box-level detection (+0.056 at high),
  while the pixel-precise tasks (keypoints, panoptic) score *lower* on
  denoised images at every severity — smoothing trades exactly the fine
  texture they depend on.
- **Fine-tuning on a corruption *mixture* is robust across the board.**
  Trained on a per-image mix of clean + all 9 cells, YOLOv8n improves every
  med/high cell (gauss_noise/high 0.097 → 0.186, salt_pepper/high +0.124,
  motion_blur/high +0.056) with only small dips on near-clean low-severity
  cells (−0.011…−0.035). The v1 single-cell training (archived in
  [docs/archive/](docs/archive/)) had shown textbook negative transfer —
  worse than the pretrained model on 5 of 9 cells.
- **Robustness costs clean accuracy, and now it's measured:** the fine-tuned
  model scores **0.284 on clean images vs 0.352 pretrained** (−0.068). This
  trade-off — invisible in v1, which never evaluated the fine-tuned model on
  clean data — is why the low-severity cells dip.
- **Stacking the two repairs helps only where each is partial.** At
  motion_blur/high, enhancement→fine-tuned is the best cell (0.174, a 39%
  recovery vs 25%/19% for either repair alone). For salt & pepper the median
  filter alone beats the stack — the filtered images resemble clean data more
  than the fine-tuned model's training distribution.

### Clean baselines (Phase 1)

| Task | Metric | Clean baseline |
|---|---|---:|
| Feature matching (ORB) | match ratio | 1.000 |
| Object detection (YOLOv8n) | mAP@[.5:.95] | 0.352 |
| Keypoint detection (Keypoint R-CNN) | OKS AP | 0.657 |
| Panoptic segmentation (Panoptic FPN) | PQ | 0.410 (SQ 0.775 / RQ 0.500; things 0.475 / stuff 0.312) |

All three pretrained models land within ~0.02 of their published full-val2017
scores, which validates the measurement pipeline itself (GT handling, format
conversions, COCOeval) before any distortion enters the picture.

### Degradation per SNR (Phase 2)

Performance vs SNR — one panel per distortion, shared y-axis. Series identity is
fixed across all panels: distorted (blue, solid, ●), enhanced (green, dashed, ■),
fine-tuned (amber, dotted, ▲); the clean baseline is the gray dashed line. The
repaired lines flatten the SNR slope — the harder the corruption, the larger
the gap they open over the distorted line — and in gauss_noise the amber
fine-tuned line crosses above everything as distortion grows:

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
- **enhanced** — the metric after the matched classical restoration
  (sigma-adaptive NLM for Gaussian noise, median filter for salt & pepper,
  non-blind Wiener deconvolution for motion blur).
- **finetuned** — the fine-tuned YOLOv8n (detection only, the DL improvement;
  **trained on a per-image mixture of clean + all 9 cells**, evaluated on all
  cells). On clean images it scores **0.284** vs the pretrained 0.352 — the
  measured price of robustness.
- **finetuned+enh** — the fine-tuned detector running on the *enhanced*
  images: both repairs stacked.
- **degradation** = distorted − clean (how much the corruption destroyed).
- **recovery (enhance / finetune / combined)** = improved − distorted (how
  much each improvement strategy won back; negative = made it worse).

Example — `salt_pepper/high`: clean mAP 0.352 collapses to 0.079 (−0.274);
the median filter restores it to 0.321 (+0.242, 89% of the damage), the
fine-tuned model to 0.203 (+0.124), both together to 0.263. Bold marks
recoveries ≥ +0.04.

| task | distortion | severity | SNR (dB) | clean | distorted | enhanced | finetuned | finetuned+enh | degradation | recovery (enhance) | recovery (finetune) | recovery (combined) |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| detection | gauss_noise | low | 19.5 | 0.352 | 0.293 | 0.288 | 0.266 | 0.244 | −0.059 | −0.004 | −0.026 | −0.048 |
| detection | gauss_noise | med | 13.9 | 0.352 | 0.201 | 0.213 | 0.233 | 0.207 | −0.151 | +0.013 | +0.032 | +0.006 |
| detection | gauss_noise | high | 10.0 | 0.352 | 0.097 | **0.154** | **0.186** | **0.164** | −0.255 | **+0.056** | **+0.089** | **+0.067** |
| detection | motion_blur | low | 20.8 | 0.352 | 0.294 | 0.334 | 0.259 | 0.275 | −0.058 | +0.039 | −0.035 | −0.020 |
| detection | motion_blur | med | 17.7 | 0.352 | 0.157 | **0.257** | **0.202** | **0.240** | −0.195 | **+0.100** | **+0.044** | **+0.083** |
| detection | motion_blur | high | 15.6 | 0.352 | 0.059 | **0.132** | **0.114** | **0.174** | −0.293 | **+0.073** | **+0.056** | **+0.116** |
| detection | salt_pepper | low | 18.7 | 0.352 | 0.284 | **0.339** | 0.273 | 0.269 | −0.068 | **+0.055** | −0.011 | −0.015 |
| detection | salt_pepper | med | 11.8 | 0.352 | 0.167 | **0.336** | **0.245** | **0.267** | −0.185 | **+0.169** | **+0.078** | **+0.100** |
| detection | salt_pepper | high | 8.2 | 0.352 | 0.079 | **0.321** | **0.203** | **0.263** | −0.274 | **+0.242** | **+0.124** | **+0.184** |
| features | gauss_noise | low | 19.5 | 1.000 | 0.834 | 0.800 | — | — | −0.166 | −0.034 | — | — |
| features | gauss_noise | med | 13.9 | 1.000 | 0.731 | 0.670 | — | — | −0.269 | −0.061 | — | — |
| features | gauss_noise | high | 10.0 | 1.000 | 0.616 | 0.538 | — | — | −0.384 | −0.079 | — | — |
| features | motion_blur | low | 20.8 | 1.000 | 0.643 | **0.781** | — | — | −0.357 | **+0.138** | — | — |
| features | motion_blur | med | 17.7 | 1.000 | 0.399 | **0.626** | — | — | −0.601 | **+0.227** | — | — |
| features | motion_blur | high | 15.6 | 1.000 | 0.197 | **0.476** | — | — | −0.803 | **+0.280** | — | — |
| features | salt_pepper | low | 18.7 | 1.000 | 0.554 | **0.705** | — | — | −0.446 | **+0.151** | — | — |
| features | salt_pepper | med | 11.8 | 1.000 | 0.404 | **0.692** | — | — | −0.596 | **+0.288** | — | — |
| features | salt_pepper | high | 8.2 | 1.000 | 0.326 | **0.662** | — | — | −0.674 | **+0.336** | — | — |
| keypoints | gauss_noise | low | 19.5 | 0.657 | 0.590 | 0.558 | — | — | −0.068 | −0.031 | — | — |
| keypoints | gauss_noise | med | 13.9 | 0.657 | 0.489 | 0.441 | — | — | −0.168 | −0.048 | — | — |
| keypoints | gauss_noise | high | 10.0 | 0.657 | 0.369 | 0.316 | — | — | −0.288 | −0.053 | — | — |
| keypoints | motion_blur | low | 20.8 | 0.657 | 0.578 | **0.620** | — | — | −0.080 | **+0.043** | — | — |
| keypoints | motion_blur | med | 17.7 | 0.657 | 0.386 | **0.524** | — | — | −0.272 | **+0.138** | — | — |
| keypoints | motion_blur | high | 15.6 | 0.657 | 0.159 | **0.387** | — | — | −0.499 | **+0.229** | — | — |
| keypoints | salt_pepper | low | 18.7 | 0.657 | 0.590 | 0.623 | — | — | −0.068 | +0.034 | — | — |
| keypoints | salt_pepper | med | 11.8 | 0.657 | 0.438 | **0.614** | — | — | −0.219 | **+0.176** | — | — |
| keypoints | salt_pepper | high | 8.2 | 0.657 | 0.348 | **0.598** | — | — | −0.310 | **+0.250** | — | — |
| segmentation | gauss_noise | low | 19.5 | 0.410 | 0.359 | 0.317 | — | — | −0.050 | −0.043 | — | — |
| segmentation | gauss_noise | med | 13.9 | 0.410 | 0.294 | 0.240 | — | — | −0.116 | −0.055 | — | — |
| segmentation | gauss_noise | high | 10.0 | 0.410 | 0.209 | 0.163 | — | — | −0.201 | −0.046 | — | — |
| segmentation | motion_blur | low | 20.8 | 0.410 | 0.354 | 0.385 | — | — | −0.056 | +0.031 | — | — |
| segmentation | motion_blur | med | 17.7 | 0.410 | 0.231 | **0.311** | — | — | −0.179 | **+0.081** | — | — |
| segmentation | motion_blur | high | 15.6 | 0.410 | 0.123 | **0.204** | — | — | −0.287 | **+0.081** | — | — |
| segmentation | salt_pepper | low | 18.7 | 0.410 | 0.290 | **0.378** | — | — | −0.120 | **+0.088** | — | — |
| segmentation | salt_pepper | med | 11.8 | 0.410 | 0.150 | **0.373** | — | — | −0.260 | **+0.223** | — | — |
| segmentation | salt_pepper | high | 8.2 | 0.410 | 0.105 | **0.348** | — | — | −0.304 | **+0.243** | — | — |

### Visual examples

**Predictions drawn on the images** (the course's "image with annotation"):
gray dashed boxes = COCO ground truth, solid boxes = YOLO detections with
scores, columns = clean / distorted (high severity) / enhanced / distorted
with the fine-tuned model. You can watch detections disappear under the
corruption and reappear after each repair:

![annotated gauss noise](results/figures/annotated_gauss_noise.png)
![annotated salt & pepper](results/figures/annotated_salt_pepper.png)
![annotated motion blur](results/figures/annotated_motion_blur.png)

Raw input/output examples — clean / distorted (high severity) / enhanced.
These show *why* the numbers behave as they do: the median filter visibly
removes impulse pixels, and the known-kernel Wiener filter visibly re-sharpens
what motion blur smeared:

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

- Data: 4,500-image train2017 subset, corrupted on the fly with a **per-image
  seeded mixture** of clean + all 9 (distortion × severity) cells (~412–490
  images per variant; same deterministic per-image RNG as the val
  distortions), **real COCO boxes** converted to YOLO labels (no
  pseudo-labels). 4,050 images train / 450 held out as the YOLO val split so
  best-checkpoint selection is honest.
- Training: `yolov8n.pt` continued with AdamW (lr0 = 1e-4 pinned, cosine
  schedule), imgsz 640, batch 16; early-stopped at epoch 15 of 30 (patience
  10) — ~12 minutes on one NVIDIA L4. Checkpoint:
  [`models/yolov8_finetuned.pt`](models/yolov8_finetuned.pt) (committed; the
  clean baseline model is untouched).
- Evaluation: the fine-tuned detector runs on the **clean** val subset
  (forgetting check: 0.284 vs 0.352 pretrained), on **all 9 distorted** cells,
  and on **all 9 enhanced** cells (the `finetuned+enh` column) of the
  1,521-image subset — all held out; the model never sees a val2017 image
  during training, so there is no leakage.

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
   with SNR, with the exact per-image degradation parameters logged. Beyond
   the requirement, the data shows **SNR alone does not predict task damage**:
   motion blur has a *higher* SNR than the noise corruptions yet destroys
   localization tasks the most (ORB −0.80, keypoints −0.50) — spatial
   structure matters more than pixel-wise error energy.
3. **Performance on enhanced images** — met, with a sharp engineering
   conclusion: classical restoration is only as good as its *degradation
   model*. Where the model is exact, recovery is large — median ↔ impulse
   noise (+0.03…+0.34 across all four tasks), logged-kernel Wiener ↔ motion
   blur (+0.03…+0.28) — and where no faithful inverse exists (i.i.d. Gaussian
   noise), denoising helps only the coarse box-level task and *hurts* the
   pixel-precise ones. The remaining negative results are reported
   deliberately — they delimit *when* enhancement is worth deploying.
4. **Fine-tuned model on distorted images** — met: mixture training improves
   every med/high cell (up to +0.124) with no catastrophic negative transfer
   (v1's single-cell training, archived, lost on 5 of 9 cells), and the
   robustness price is *measured*: −0.068 mAP on clean images, which is
   exactly where the small low-severity dips come from.

Bottom line: the four phases compose into a **decision matrix** — impulse
noise: restore classically (cheapest, best); motion blur with a known/
calibratable kernel: deconvolve, and stack fine-tuning if the blur is heavy;
Gaussian noise: fine-tune the model, don't filter; and if the deployment also
sees clean images, budget the measured −0.068 clean-accuracy cost or route
inputs by corruption type — exactly the trade-off the assignment asks the
project to expose.

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
  enhancement.py           # adaptive-NLM / median / non-blind Wiener restorers
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
