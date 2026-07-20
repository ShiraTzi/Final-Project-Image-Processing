# Presentation Slide Script
# Robustness Benchmark for Image-Processing and Vision Methods
# Team: Ohad Shpizhizen · Shira Tziony
# 24 slides

---

## Slide 1 — Title / Introduction

**Title:** Robustness Benchmark for Image-Processing and Vision Methods

**Subtitle / body bullets:**
- Evaluating how vision algorithms degrade under image distortion — and how well they recover
- Dataset: COCO val2017 (1,521-image balanced subset)
- 4 tasks · 3 distortions · 3 severities · 2 improvement strategies
- Team: Ohad Shpizhizen · Shira Tziony
- Course: Image Processing / Computer Vision — [Semester/Year]
- GitHub: https://github.com/ShiraTzi/Final-Project-Image-Processing

---

## Slide 2 — Table of Contents

**Title:** Table of Contents

**Body (numbered list):**
1. Project Pipeline
2. Dataset — COCO
3. Selected Tasks (4 tasks)
4. Selected Noises & Enhancements
5. Results
   - Recovery per cell (Plot 1)
   - Per-class AP — Gaussian noise high (Plot 2)
   - Metric vs SNR curves (Plot 3)
   - Clean baselines table + Per-class AP clean (Plot 4)
   - Visual examples — all tasks (Plots 5–8)
   - Raw noise examples (Plots 9–11)
6. Conclusions
7. Appendix — Previous Versions

---

## Slide 3 — Project Pipeline

**Title:** Project Pipeline

**Show the course pipeline diagram adapted to our project:**

```
data ──► distort ──► enhance ──┐
  │          │          │       ├─► infer ─► eval ─► tables/figures
  └──────────┴──────────┴───────┘            ▲
                          finetune (YOLOv8) ──┘
```

**Table below the diagram:**

| Phase | What it does |
|---|---|
| 0 — Setup & Data | Download COCO annotations + 1,521-image balanced val subset |
| 1 — Clean Baseline | Run all models on clean images; validate against published scores |
| 2 — Distortion | Apply 3 corruptions × 3 severities; log SNR + exact degradation params |
| 3 — Enhancement | Apply matched classical restorer per image (BM3D / median / Wiener) |
| 4 — Fine-tuning | Fine-tune YOLOv8n on 9,000-image clean+distorted+restored mixture |
| 5 — Report | Comparison tables, per-class AP bars, metric-vs-SNR curves, visual grids |

**Speaker note:** The pipeline follows the course structure exactly. The key design decision is logging the per-image degradation parameters at Step 2 — this is what enables non-blind restoration at Step 3.

---

## Slide 4 — The Dataset

**Title:** Dataset — COCO val2017

**Left column — What is COCO?**
- Common Objects in Context — large-scale benchmark by Microsoft
- 5,000 validation images, 80 object categories
- Unique: provides bounding boxes + keypoints + panoptic masks for the same images → one dataset serves all four tasks
- Official COCOeval toolkit → metrics are directly comparable to published results
- All chosen models are COCO-pretrained → baseline validation is straightforward

**Right column — Our Subset**
- Fixed seeded subset of **1,521 images** (single source of truth for every variant)
- Sampling strategy: random seed + coverage top-up (every class ≥ 20 GT instances)
- Class distribution correlation with full val2017: **r = 0.999**
- 11,129 GT instances; 3,273 person instances
- All experiments are **paired** (same images + GT across clean / distorted / enhanced)

**Optional image:** COCO sample grid showing multiple categories / panoptic overlay

---

## Slide 5 — Selected Tasks Overview

**Title:** Selected Tasks — Low-level and High-level Vision

**Table:**

| Task | Level | Algorithm | Metric |
|---|---|---|---|
| Feature matching | **Low-level** | ORB + BFMatcher | Match ratio vs clean |
| Object detection | **High-level** | YOLOv8n | mAP@[.5:.95] |
| Keypoint detection | **High-level** | Keypoint R-CNN | OKS AP |
| Panoptic segmentation | **High-level** | Panoptic FPN | PQ = SQ × RQ |

**Short justification bullets:**
- One low-level task (ORB) — tests raw signal integrity, no ground truth needed
- Three high-level tasks — span coarse localization (bbox) → fine localization (joints) → pixel-level (panoptic)
- Together they reveal *what aspect of the image* each corruption damages most
- All metrics are 0–1, higher is better

---

## Slide 6 — Task: Feature Matching (ORB)

**Title:** Task 1 — Feature Matching (ORB) · Low-level

**Algorithm:**
- ORB = Oriented FAST + Rotated BRIEF
- Detects corner-like keypoints with scale and orientation invariance
- Computes compact binary descriptors
- Matched with Brute-Force Matcher (Hamming distance, cross-check)

**Metric — Match Ratio:**
- (Good matches) / (Clean keypoints)
- Good match = Hamming distance ≤ 64, cross-checked
- 1.0 = every clean feature survives; 0 = local structure destroyed
- No ground truth needed — the clean image itself is the reference

**Why chosen:**
- Sole low-level task: answers "does the raw pixel signal preserve local structure?"
- Known sensitivity: degrades under noise, collapses under blur → strong contrast with high-level tasks
- Runs on CPU in seconds; scales to the full subset easily

**Optional image:** ORB keypoints drawn on a COCO image

---

## Slide 7 — Task: Object Detection (YOLOv8n)

**Title:** Task 2 — Object Detection (YOLOv8n) · High-level

**Algorithm:**
- YOLOv8n (nano variant) — single-stage real-time detector
- COCO-pretrained on 80 classes
- Predicts bounding boxes + class labels simultaneously

**Metric — mAP@[.5:.95]:**
- Average Precision averaged over 10 IoU thresholds (0.50 … 0.95)
- Further averaged over all 80 classes
- Deliberately strict: rewards tight localization, not just detection
- Clean baseline: **0.352** (published full-val2017: ~0.373)

**Why chosen:**
- Most canonical high-level robustness benchmark
- Fast enough to run full 1,521-image subset at scale
- The only task we fine-tune → focal point of enhancement vs fine-tuning comparison
- Metric reacts early to corruption (tight IoU thresholds)

**Optional image:** YOLO predictions drawn on a sample COCO image

---

## Slide 8 — Task: Keypoint Detection (Keypoint R-CNN)

**Title:** Task 3 — Keypoint Detection (Keypoint R-CNN) · High-level

**Algorithm:**
- Keypoint R-CNN with ResNet-50 + FPN backbone, COCO-pretrained (torchvision)
- Predicts 17 human body joints per detected person instance
- Joints: nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles

**Metric — OKS AP:**
- Object Keypoint Similarity: scale-normalized Euclidean distance between predicted and GT joints
- AP machinery (same as detection) but box-IoU → OKS
- Person-only → isolates fine spatial localization sensitivity
- Clean baseline: **0.657**

**Why chosen:**
- Tests pixel-accurate joint localization — fundamentally different sensitivity from bbox detection
- A corruption that barely hurts detection (e.g., mild blur) can dramatically shift joint predictions
- The detection-vs-keypoints contrast is one of the most informative axes of the benchmark

**Optional image:** Keypoint skeleton drawn on a person in a COCO image

---

## Slide 9 — Task: Panoptic Segmentation (Panoptic FPN)

**Title:** Task 4 — Panoptic Segmentation (Panoptic FPN) · High-level

**Algorithm:**
- Panoptic FPN with ResNet-50 backbone, COCO-pretrained (detectron2)
- Assigns every pixel a semantic class AND instance identity
- Handles "things" (cars, people — countable) AND "stuff" (sky, grass, road — amorphous)

**Metric — PQ = SQ × RQ:**
- **RQ** (Recognition Quality): F1 over correctly matched segments — penalizes missed/false segments
- **SQ** (Segmentation Quality): mean IoU of matched segments — penalizes sloppy masks
- PQ drops if segments are missed (RQ↓) OR if boundaries are imprecise (SQ↓)
- Clean baseline: **PQ 0.410** (SQ 0.775 / RQ 0.500)

**Why chosen:**
- Most demanding pixel-level task — requires detecting, classifying, and masking every pixel
- PQ decomposition into SQ/RQ provides a built-in diagnostic for *how* corruption hurts
- This diagnostic was essential in identifying the v2 NLM failure (see Appendix)
- Runs in a separate virtualenv (detectron2 requires different torch) via subprocess

**Optional image:** Panoptic segmentation overlay on a COCO image (colored masks + boundaries)

---

## Slide 10 — Selected Noises and Enhancements

**Title:** Selected Noises & Enhancement Methods

**Table:**

| Distortion | Models | SNR range | Enhancement | Why |
|---|---|---|---|---|
| **Gaussian noise** | Sensor/intensity noise | 10–19.5 dB | Non-blind BM3D (logged sigma) | AWGN is the textbook case; BM3D is the strongest classical denoiser |
| **Salt-and-pepper** | Impulsive pixel corruption | 8.2–18.7 dB | Median filter (3×3) | Median discards impulse outliers exactly; essentially optimal for this model |
| **Motion blur** | Camera shake / object motion | 15.6–20.8 dB | Non-blind Wiener (logged kernel) | Wiener with the true kernel is the optimal linear inverse for known blur |

**Key design decision — logging degradation parameters:**
- Noise sigma, impulse fraction, blur kernel length + angle logged per image at corruption time
- Enables **non-blind restoration** at enhancement phase — restorer inverts the known forward model
- A blind kernel guess for motion blur measured *worse than no restoration* → oracle knowledge is decisive

**Speaker note:** The three corruptions were chosen to be qualitatively different: continuous noise (AWGN), discrete outliers (impulse), and frequency-domain loss (blur). Together they test the full range of classical restoration techniques.

---

## Slide 11 — Results: Table of Contents

**Title:** Results — Road Map

**Body (numbered, matching slides to follow):**

1. **Plot 1:** Recovery share per distortion × severity × task
2. **Plot 2:** Per-class AP breakdown — Gaussian noise, high severity
3. **Plot 3:** Task metric vs SNR (all four tasks)
4. **Clean baseline table** + **Plot 4:** Per-class AP on clean images
5. **Plots 5–8:** Visual examples — predictions on images (all 4 tasks, all 3 distortions)
6. **Plots 9–11:** Raw noise and enhancement examples (pixel-level grids)

---

## Slide 12 — Plot 1: Recovery per Cell

**Title:** Recovery per Distortion × Severity Cell

**Image:** `results/figures/recovery_bars.png`

**Caption / description (no conclusions):**
- Each panel = one task; bar groups = one distortion × severity combination
- Blue = distorted metric; green = after classical enhancement; amber = after fine-tuning (detection only)
- Dashed horizontal line = clean baseline
- Each bar label = fraction of total damage recovered (e.g., "+89%" means 89% of the drop was undone)
- 9 bar groups × 4 tasks = 36 experimental cells, all shown in one figure

---

## Slide 13 — Plots 2 & 3

**Title:** Per-Class AP and Metric vs SNR

**Top half — Plot 2** (`results/figures/per_class_ap_gauss_noise_high.png`):
- Per-class detection AP for the gauss_noise / high cell
- Gray = clean baseline; blue = distorted; green = after BM3D; amber = fine-tuned model
- Sorted by clean AP (descending); n= label = GT instance count per class

**Bottom half — Plot 3** (representative SNR curve, e.g., detection):
- One of `acc_vs_snr_detection.png`, `acc_vs_snr_features.png`, `acc_vs_snr_keypoints.png`, `acc_vs_snr_segmentation.png`
- X-axis = measured SNR (dB); Y-axis = task metric
- Three series: distorted (blue ●), enhanced (green ■), fine-tuned (amber ▲; detection only)
- Gray dashed = clean baseline; lower SNR = more severe corruption
- Consider showing all 4 SNR figures or selecting the most illustrative one

---

## Slide 14 — Clean Baseline Table + Plot 4

**Title:** Clean Baselines and Per-Class AP Distribution

**Top — Clean baseline table:**

| Task | Metric | Clean baseline |
|---|---|---|
| Feature matching (ORB) | Match ratio | 1.000 |
| Object detection (YOLOv8n) | mAP@[.5:.95] | 0.352 |
| Keypoint detection (Keypoint R-CNN) | OKS AP | 0.657 |
| Panoptic segmentation (Panoptic FPN) | PQ | 0.410 |

- All pretrained models land within ~0.02 of published full-val2017 scores → pipeline validated

**Bottom — Plot 4** (`results/figures/per_class_ap_clean.png`):
- Per-class YOLOv8n AP on clean images, sorted descending
- Dashed line = mean AP; n= label = GT instance count per class
- Establishes the starting point from which all degradation and recovery deltas are measured

---

## Slide 15 — Plot 5: Object Detection Examples

**Title:** Visual Examples — Object Detection (YOLOv8n)

**Three images stacked or side by side:**
- `results/figures/annotated_gauss_noise.png`
- `results/figures/annotated_salt_pepper.png`
- `results/figures/annotated_motion_blur.png`

**Caption:**
- Gray dashed boxes = COCO ground-truth bounding boxes
- Solid colored boxes + labels = YOLOv8n predictions with confidence scores
- Columns within each image: clean / distorted (high severity) / enhanced / fine-tuned on distorted

---

## Slide 16 — Plot 6: Keypoint Detection Examples

**Title:** Visual Examples — Keypoint Detection (Keypoint R-CNN)

**Three images:**
- `results/figures/keypoints_gauss_noise.png`
- `results/figures/keypoints_salt_pepper.png`
- `results/figures/keypoints_motion_blur.png`

**Caption:**
- Gray dashed lines + dots = ground-truth skeleton from COCO keypoints annotation
- White skeleton + yellow joint dots = model predictions at score ≥ 0.35
- Columns: clean / distorted (high severity) / enhanced

---

## Slide 17 — Plot 7: Panoptic Segmentation Examples

**Title:** Visual Examples — Panoptic Segmentation (Panoptic FPN)

**Three images:**
- `results/figures/panoptic_gauss_noise.png`
- `results/figures/panoptic_salt_pepper.png`
- `results/figures/panoptic_motion_blur.png`

**Caption:**
- Color per semantic category (consistent palette across all panels)
- White hairlines = segment boundaries
- Column header = cell PQ value
- Columns: ground truth / clean / distorted (high severity) / enhanced

---

## Slide 18 — Plot 8: ORB Feature Matching Examples

**Title:** Visual Examples — ORB Feature Matching

**Three images:**
- `results/figures/orb_matches_gauss_noise.png`
- `results/figures/orb_matches_salt_pepper.png`
- `results/figures/orb_matches_motion_blur.png`

**Caption:**
- Each pair: clean image (left) matched against distorted or enhanced variant (right)
- Lines = good matches (Hamming ≤ 64, cross-checked — same criterion as the metric)
- Caption below each pair = actual per-image match ratio (good matches / clean keypoints)
- Line geometry reveals match quality: crossing lines = false matches, parallel lines = spatially consistent true matches

---

## Slide 19 — Plots 9–11: Gaussian Noise Examples

**Title:** Raw Examples — Gaussian Noise (High Severity)

**Image:** `results/figures/grid_gauss_noise.png`

**Caption:**
- Rows = different images from the subset; columns = clean / distorted / enhanced
- Shows what the distortion and enhancement look like at the pixel level before any model sees the image
- Heavy grain visible in distorted; BM3D removes it while preserving edges and texture detail

---

## Slide 20 — Salt-and-Pepper + Motion Blur Examples

**Title:** Raw Examples — Salt-and-Pepper and Motion Blur

**Top image:** `results/figures/grid_salt_pepper.png`
- Random black and white impulse pixels visible in the distorted column
- Median filter restores clean-looking images; impulse pixels are fully removed

**Bottom image:** `results/figures/grid_motion_blur.png`
- Directional smearing visible in distorted column
- Wiener deconvolution re-sharpens edges and detail in the enhanced column

---

## Slide 21 — All Three Noise Comparisons Side-by-Side (optional / summary visual)

**Title:** Raw Noise and Enhancement — All Three Distortions

**Show a single representative image processed through all three distortion pipelines:**
- Or a 3-panel comparison using one image from each grid
- This slide is optional; use if the previous two slides feel too sparse

**Caption:**
- Three qualitatively different corruptions: continuous noise / discrete outliers / frequency-domain loss
- Three qualitatively different restorations: BM3D / median / Wiener deconvolution

---

## Slide 22 — Conclusions: Enhancement vs Fine-tuning

**Title:** Discussion — Why Enhancement Outperforms Fine-tuning (7 of 9 cells)

**Four structural reasons:**

1. **Oracle knowledge:** Restorer gets the exact degradation parameters; fine-tuned model gets nothing
   - Evidence: blind kernel guess for motion blur scored *worse than no restoration*

2. **Distribution shift:** Enhancement pushes inputs back to the model's clean-image home; fine-tuning shifts weights away from it
   - Evidence: fine-tuned model pays −0.042 clean mAP; pretrained model is untouched

3. **Textbook best case for classical methods:** Synthetic, uniform, closed-form noise models
   - Real-world corruptions (mixed, spatially-varying, unknown) would close this gap

4. **Model-agnostic:** The same enhanced images improve all 4 tasks; fine-tuning only helped detection

**When fine-tuning wins:**
- Unknown corruption: fine-tuned model is safe (worst case −0.012 vs baseline)
- Partially irreversible damage: heavy motion blur destroys frequency bands no filter recovers; stacking wins

---

## Slide 23 — Conclusions: SNR, Negative Transfer, and the Mixture Fix

**Title:** Discussion — Additional Findings

**SNR vs task damage:**
- Degradation tracks SNR monotonically for every task ✓
- But motion blur at high severity (SNR ~15.6 dB) is *more* destructive than Gaussian noise at medium (SNR ~13.9 dB) for ORB and keypoints
- Spatial structure loss ≠ pixel-energy deviation

**The v2 NLM failure — the filter, not the concept:**
- NLM scored *below noisy images* on pixel-precise tasks
- Diagnosis: blotchy residual noise + texture loss → small objects and stuff segments disappeared
- BM3D fixed it: collaborative patch filtering → texture preserved, noise removed
- The archived negative result now delimits *which filter* to deploy

**Fine-tuning mixture design:**
- v1 (single-cell): textbook negative transfer → failed on 5/9 cells
- v2 (10% clean, no restored images): −0.068 clean mAP + domain mismatch on enhanced images
- v3 (25% clean + restored picks): −0.042 clean mAP, domain mismatch resolved, stack additive

---

## Slide 24 — Conclusion

**Title:** Conclusion

**Key takeaways:**

- Image distortion inflicts substantial but largely **recoverable** damage — when restoration matches the degradation model
- Classical enhancement with oracle knowledge of corruption parameters is the strongest single repair on **7 of 9 cells** and all four tasks
- Fine-tuning is the robust fallback for **unknown or irreversible** corruptions; stacking both is additive when training includes restored images
- The PQ decomposition (SQ/RQ) and size-stratified AP provided the diagnostic power to locate and fix a denoiser that actively hurt the system — turning a negative result into a validated design principle
- The result is a **repeatable robustness evaluation framework** applicable to any vision task, dataset, and corruption regime

**Closing line:**
Know your corruption → choose your weapon. When the forward model is known and invertible, image space is the cheaper and stronger battlefield. When it isn't, model robustness is your safety net.

---

*GitHub: https://github.com/ShiraTzi/Final-Project-Image-Processing*
