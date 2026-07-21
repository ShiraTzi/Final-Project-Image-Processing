# Final Project Report — Image Processing and Computer Vision

**Team Members:** 
Ohad Shpizhizen (oshpizhizen@nvidia.com)  
Shira Tziony (shira.tziony@gmail.com)  
**GitHub Repository:** [Final-Project-Image-Processing](https://github.com/ShiraTzi/Final-Project-Image-Processing)

---

## Results at a Glance

![Recovery summary and representative outputs from all four tasks](results/figures/readme_results_teaser.png)


---

## Overview

This project systematically benchmarks the robustness of modern image-processing and computer vision algorithms to realistic image corruptions, using the COCO dataset. We:

- Establish a **clean baseline** for four key vision tasks (feature matching, object detection, keypoint detection, panoptic segmentation)
- Quantify **performance degradation** when images are subjected to three types of distortions (Gaussian noise, salt-and-pepper, motion blur) at three levels of severity each
- Evaluate two recovery strategies: classical **image enhancement** (non-blind restoration via BM3D, median filter, Wiener deconvolution) and deep **model fine-tuning**
- Report results as a function of distortion type/severity, signal-to-noise ratio (SNR), and per-class and per-task performance

---

## Project Pipeline

The experimental workflow follows a five-phase structure:

| Phase | Description | Entry Point |
|---|---|---|
| 0 — Setup & Data      | Build environments, download COCO data/subset/panoptic GT | `src.data` |
| 1 — Clean Baseline    | Run all models on clean images, validate baseline metrics | `--only infer eval` |
| 2 — Distortion        | Generate distorted images (3 corruptions × 3 severities), log SNR | `src.distortions` → infer → eval |
| 3 — Enhancement       | Apply matched restoration filters, assess recovery          | `src.enhancement` → infer → eval |
| 4 — Fine-tuning       | Fine-tune YOLOv8n on a mixture of clean/distorted/restored data | `src.finetune_det` |
| 5 — Reporting         | Aggregate and visualize results, create tables/plots        | `src.tables`, `src.visualize` |

![Experimental design pipeline](results/figures/experimental_design_pipeline.png)

The fixed evaluation subset is processed as clean, distorted, and enhanced variants before inference with all four methods. A separate train2017 subset is used to fine-tune YOLOv8n, which is then evaluated on the same detection cells.

---

## The Dataset

### What is COCO?

**COCO (Common Objects in Context)** is a large-scale computer-vision benchmark [introduced by Lin et al.](https://arxiv.org/abs/1405.0312). It contains approximately 330,000 images, of which more than 200,000 are labeled, and covers 80 object categories. Its task-specific annotations include object bounding boxes, instance segmentation masks, 17-keypoint skeletons for people, and panoptic segmentation labels.

We use the **val2017** split (5,000 images) as our evaluation pool. It has public ground-truth annotations and is widely used for reporting benchmark results, allowing us to compare our clean baselines with published model scores.

### Why COCO?

- **Multiple tasks on the same image set.** Detection, person-keypoint, and panoptic annotations can be evaluated on one fixed image subset without cross-dataset alignment. ORB feature matching does not use COCO ground truth; it uses each clean image as its reference.
- **Standardized evaluation tools.** The official [COCO API](https://github.com/cocodataset/cocoapi) computes detection AP and keypoint OKS AP, while the official [Panoptic API](https://github.com/cocodataset/panopticapi) computes PQ. This makes our evaluation reproducible and uses the same metric definitions as published COCO results. Because we evaluate on a custom 1,521-image subset, absolute scores are not directly comparable with results reported on the full val2017 split.
- **All three learned models are COCO-pretrained.** YOLOv8n, Keypoint R-CNN, and Panoptic FPN ship with weights trained on COCO train2017. Their clean baselines can therefore be used as an approximate sanity check against published full-val2017 scores, rather than as an exact comparison.
- **Class diversity.** 80 categories across people, animals, vehicles, household objects, and outdoor scenes ensure our per-class degradation analysis is meaningful and not dominated by a single domain.

### Our Subset

A **fixed, seeded subset of 1,521 images** is the single source of truth for every experiment variant. It is a seeded random sample of 1,500 images **topped up with 21 images for class coverage**: after the initial sample, images are added (in the same shuffled order) until every category reaches a minimum GT instance count, bounded by what val2017 contains. This prevents per-class AP from being dominated by rare categories with only 2–3 instances.

**Exact evaluation dataset:** [the complete list of 1,521 COCO image IDs used in every experiment](data/splits/val_subset.json), sampled from the official [COCO val2017 dataset](https://cocodataset.org/#download).

| Property | Value |
|---|---|
| Subset size | 1,521 images |
| GT instances | 11,129 total; 3,273 person instances |
| Class correlation with full val2017 | r = 0.999 |
| Minimum instances per class | ≥ 20 (except `toaster` = 9, `hair drier` = 11 — both exhaust their entire val2017 supply) |
| All comparisons | Paired: identical images and GT across every variant |

A separate seeded **9,000-image train2017 subset** is used only for fine-tuning; its exact image IDs are listed in [data/splits/train_subset.json](data/splits/train_subset.json). Panoptic PQ additionally requires the COCO panoptic GT (~821 MB, fetched automatically).

---

## Selected Tasks

Four tasks covering both low-level signal analysis and high-level scene understanding:

| Task | Level | Model / Algorithm | Library | Metric |
|---|---|---|---|---|
| Feature matching | **Low-level** | [ORB](https://doi.org/10.1109/ICCV.2011.6126544) + [BFMatcher](https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html) (Hamming, cross-check) | [OpenCV](https://opencv.org/) | Match ratio vs clean |
| Object detection | **High-level** | [YOLOv8n](https://docs.ultralytics.com/models/yolov8/) (COCO-pretrained) | [Ultralytics](https://docs.ultralytics.com/) | [COCO mAP](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)@[.5:.95] |
| Keypoint detection | **High-level** | [Keypoint R-CNN ResNet50-FPN](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html) (COCO-pretrained) | [torchvision](https://pytorch.org/vision/stable/) | [OKS AP](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) |
| Panoptic segmentation | **High-level** | [Panoptic FPN R50](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-panoptic-segmentation-baselines-with-panoptic-fpn) (COCO-pretrained) | [detectron2](https://github.com/facebookresearch/detectron2) | [PQ = SQ × RQ](https://github.com/cocodataset/panopticapi) |

All metrics are in the range [0, 1]; higher is better.

### Feature Matching — ORB (Low-level)

**What it does:** ORB (Oriented FAST and Rotated BRIEF) detects corner-like keypoints and computes compact binary descriptors. We run ORB on the clean image and the degraded or restored variant, then match with a Brute-Force Matcher using Hamming distance and cross-check. For each image, the **match ratio** is the number of matches with Hamming distance ≤ 64 divided by the number of keypoints in the clean image; the reported score is the mean ratio across the evaluation subset. A score of 1.0 means every clean feature was matched; 0 means none were matched. No ground-truth annotations are needed — the clean image itself is the reference.

**Justification and caveat:** As the only low-level task, ORB measures whether local image structure survives corruption, complementing the three ground-truth-based semantic tasks. Its binary descriptors are sensitive to noise and blur, and the method runs quickly on CPU. However, because the clean image is the reference, an enhancement that changes texture can lower the ORB score even when it improves detection, keypoints, or segmentation.

### Object Detection — YOLOv8n (High-level)

**What it does:** YOLOv8n (nano variant) is a real-time, single-stage object detector. Given an image it predicts bounding boxes and class labels for all 80 COCO categories simultaneously. It is scored with **mAP@[.5:.95]**: the area under the interpolated precision–recall curve, averaged over all 80 classes and 10 bounding-box IoU thresholds from 0.50 to 0.95. This is deliberately strict — it rewards both detecting and tightly localizing objects.

**Justification:** Object detection is the most canonical high-level vision task and one of the most widely benchmarked for robustness. YOLOv8n is fast enough to run the full 1,521-image subset without excessive GPU time. Its published full-val2017 mAP (~0.373) provides an approximate sanity check for our clean baseline, not a direct comparison because we use a custom subset. Detection is the only task for which we apply fine-tuning, making it the focal point of the enhancement-vs-fine-tuning comparison.

### Keypoint Detection — Keypoint R-CNN (High-level)

**What it does:** Keypoint R-CNN (ResNet50-FPN backbone, COCO-pretrained) localizes 17 human body joints — shoulders, elbows, wrists, hips, knees, ankles, eyes, ears, and nose — per detected person instance. It is scored with **OKS AP**: the standard AP calculation with bounding-box IoU replaced by Object Keypoint Similarity. OKS measures the distance between predicted and ground-truth joints, normalizes it by person scale, includes only labeled joints, and applies a COCO-defined tolerance for each joint type. AP is averaged over 10 OKS thresholds from 0.50 to 0.95.

**Justification:** Keypoints probe a fundamentally different sensitivity than bounding-box detection. A corruption that merely blurs edges may not harm detection much but can cause predicted joint positions to drift significantly, because joint localization requires resolving subtle local appearance cues. The detection–vs–keypoints contrast is one of the most informative dimensions of this benchmark.

### Panoptic Segmentation — Panoptic FPN (High-level)

**What it does:** Panoptic FPN (Feature Pyramid Network with ResNet-50 backbone, COCO-pretrained via detectron2) assigns each evaluated pixel a semantic category and segment identity. It handles both "things" (countable objects: cars, people, …) and "stuff" (amorphous regions: sky, grass, road, …). It is scored with **PQ = SQ × RQ**. Predicted and ground-truth segments of the same category are matched when IoU > 0.5; **SQ** is their mean IoU, while **RQ = TP / (TP + 0.5 FP + 0.5 FN)** measures segment recognition. PQ decreases when segments are missed, falsely predicted, or imprecisely bounded.

**Justification:** Panoptic segmentation adds dense, pixel-level scene understanding to the benchmark, complementing bounding-box detection and person keypoints. It tests whether corruption damages object recognition, instance separation, semantic "stuff" classification, and mask boundaries, making it the broadest high-level task in the project.

**Diagnostic value:** The PQ decomposition separates recognition failures from boundary-quality failures: lower RQ indicates missing or false segments, while lower SQ indicates less accurate masks for correctly matched segments. This distinction was essential in diagnosing the v2 NLM failure (see Appendix).

**Implementation note:** Panoptic FPN runs in a separate virtual environment because detectron2 requires a different PyTorch version; the pipeline invokes it automatically as a subprocess.

---

## Selected Noises

Three corruption types are applied at **three severities** (low / medium / high). Each `(image, distortion, severity)` combination is reproducible: its random-number generator is seeded with a stable CRC32 digest of the global seed, image ID, distortion type, and severity. Distorted and enhanced images are stored as **lossless PNG** so JPEG compression does not alter the generated corruption.

For every image variant, `results/metrics/snr_index.csv` records the measured SNR (dB) and degradation parameters at the precision used by the restoration stage: Gaussian-noise variance, impulse amount, and blur-kernel size and angle. BM3D derives its noise sigma from the recorded variance, while Wiener deconvolution reconstructs the corresponding blur kernel. This enables matched, **non-blind restoration** instead of estimating the degradation from the corrupted image.

| Distortion | What it models | How it is applied | Enhancement method |
|---|---|---|---|
| [**Gaussian noise**](https://en.wikipedia.org/wiki/Gaussian_noise) | Sensor / intensity noise | Additive white Gaussian noise; variance sampled and recorded per image | Non-blind [**BM3D**](https://doi.org/10.1109/TIP.2007.901238) using sigma derived from the recorded variance |
| [**Salt-and-pepper**](https://en.wikipedia.org/wiki/Salt-and-pepper_noise) | Impulsive pixel corruption | Random pixel locations set to 0 or 255; configured amount recorded | [**Median filter**](https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html) (3×3) |
| [**Motion blur**](https://en.wikipedia.org/wiki/Motion_blur) | Global camera-motion blur | Whole-image convolution with a linear kernel; size and angle recorded | Non-blind [**Wiener deconvolution**](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.wiener) with the reconstructed kernel |

### Gaussian Noise

**What it is:** Additive white Gaussian noise (AWGN) approximates signal-independent noise from sources such as sensor electronics and analog-to-digital conversion. Each pixel channel receives an independent Gaussian perturbation. The variance is sampled per image and recorded; sigma is its square root. Mean measured SNR: ~19.5 dB (low) / ~13.9 dB (medium) / ~10.0 dB (high).

**Enhancement — Non-blind BM3D:** BM3D (Block-Matching 3D) is an established classical AWGN denoiser. It groups similar image patches, filters them collaboratively in a 3D transform domain, and aggregates the restored patches. We derive sigma from the recorded variance, allowing BM3D's strength to match the generated noise level instead of estimating it from the corrupted image. BM3D replaced the v2 NLM method after NLM reduced ORB, keypoint, and panoptic performance (see Appendix).

**Why chosen:** AWGN is a standard signal-processing noise model with a simple, controlled forward process. It provides a clear test of matched denoising and a useful contrast with sparse impulse noise and spatial blur.

### Salt-and-Pepper Noise

**What it is:** Impulsive corruption in which randomly selected pixel locations are replaced with either 0 (black) or 255 (white). It approximates effects such as sensor defects or transmission errors. The configured amount is fixed by severity (1% / 5% / 12%), while the affected locations are sampled independently for each image. Mean measured SNR: ~18.7 dB (low) / ~11.8 dB (medium) / ~8.2 dB (high).

**Enhancement — Median filter:** A 3×3 median filter replaces each channel value with the median of its local neighborhood. It suppresses isolated extreme values and generally preserves edges better than an averaging filter, although it can still remove fine detail.

**Why chosen:** Salt-and-pepper noise is qualitatively different from AWGN: its perturbations are sparse and extreme rather than continuous. The strong measured recovery from median filtering demonstrates how well a simple matched filter can work when the corruption model is known.

### Motion Blur

**What it is:** Global linear motion blur is produced by convolving the whole image with a directional kernel, approximating camera motion during exposure. Kernel size is fixed by severity (5 / 11 / 21 pixels), while its angle is sampled per image. The recorded size and angle allow the kernel to be reconstructed for restoration. Mean measured SNR: ~20.8 dB (low) / ~17.7 dB (medium) / ~15.6 dB (high).

**Enhancement — Non-blind Wiener deconvolution:** We reconstruct the blur kernel from its recorded parameters and apply frequency-domain Wiener deconvolution with fixed regularization (`NSR = 0.02`). This approximately inverts the blur while limiting amplification near frequencies suppressed by the kernel. In our earlier test, using an incorrect kernel angle performed worse than no restoration, showing that deconvolution is sensitive to kernel mismatch.

**Why chosen:** Motion blur occupies a fundamentally different regime from the noise corruptions: it destroys high-frequency spatial information (edges, fine texture, sharp boundaries) rather than adding noise. Despite its *higher* SNR than the noise corruptions at comparable severity, it causes the most damage to localization-sensitive tasks (ORB match ratio −0.80, OKS AP −0.50 at high severity) — demonstrating that SNR alone does not predict task damage.

---

## Results

All numbers are on the fixed 1,521-image, class-coverage-balanced COCO val2017 subset.  
Full per-cell tables: [results/metrics/comparison.md](results/metrics/comparison.md) / [comparison.csv](results/metrics/comparison.csv)  
Long format with every metric: [results/metrics/summary_long.csv](results/metrics/summary_long.csv)

### Clean Baselines — Phase 1

| Task | Metric | Our 1,521-image subset | Published full val2017 | Difference |
|---|---|---:|---:|---:|
| Feature matching (ORB) | Match ratio | 1.000 | — | — |
| Object detection (YOLOv8n) | mAP@[.5:.95] | 0.352 | [0.373](https://docs.ultralytics.com/models/yolov8/#performance-metrics) | −0.021 |
| Keypoint detection (Keypoint R-CNN) | OKS AP | 0.657 | [0.650](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.detection.keypointrcnn_resnet50_fpn.html) | +0.007 |
| Panoptic segmentation (Panoptic FPN) | PQ | 0.410 | [0.415](https://github.com/facebookresearch/detectron2/blob/main/MODEL_ZOO.md#coco-panoptic-segmentation-baselines-with-panoptic-fpn) | −0.005 |

Difference is our subset score minus the published full-val2017 score. ORB's clean score is 1.000 by construction because each clean image is matched with itself. The learned-model differences range from −0.021 to +0.007 (−2.1 to +0.7 metric points). Because the image sets differ, these values are an approximate sanity check—not a direct benchmark replication—and indicate no obvious issue in annotation handling, format conversion, or evaluation.

For the clean Panoptic FPN baseline, the additional category-averaged metrics are SQ = 0.775, RQ = 0.500, PQ things = 0.475, and PQ stuff = 0.312.

---

### Full Comparison Table — Clean vs Distorted vs Enhanced vs Fine-tuned

Each row is one experimental cell: the same model and the same 1,521 val images — only image quality changes.

**Column guide:**
- **SNR (dB)** — measured signal-to-noise ratio of distorted vs clean images (lower = more corrupted)
- **clean** — task metric on original images (constant per task; detection = mAP@[.5:.95], features = ORB match ratio, keypoints = OKS AP, segmentation = PQ)
- **distorted** — metric on corrupted images
- **enhanced** — metric after matched classical restoration (non-blind BM3D for Gaussian, median filter for salt & pepper, non-blind Wiener for motion blur)
- **finetuned** — fine-tuned YOLOv8n (detection only; trained on a per-image mixture of 25% clean + all 9 distortion cells, half of the corrupted picks classically restored); evaluated on all cells
- **finetuned+enh** — fine-tuned detector running on enhanced images: both repairs stacked
- **degradation** = distorted − clean
- **recovery** = improved − distorted (positive = improvement; negative = made it worse). Bold marks recoveries ≥ +0.04.

| task | distortion | severity | SNR (dB) | clean | distorted | enhanced | finetuned | finetuned+enh | degradation | recovery (enhance) | recovery (finetune) | recovery (combined) |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| detection | gauss_noise | low | 19.5 | 0.352 | 0.293 | **0.336** | 0.282 | 0.299 | −0.059 | **+0.044** | −0.010 | +0.006 |
| detection | gauss_noise | med | 13.9 | 0.352 | 0.201 | **0.312** | 0.240 | **0.279** | −0.151 | **+0.111** | +0.039 | **+0.078** |
| detection | gauss_noise | high | 10.0 | 0.352 | 0.097 | **0.269** | **0.183** | **0.246** | −0.255 | **+0.172** | **+0.086** | **+0.149** |
| detection | motion_blur | low | 20.8 | 0.352 | 0.294 | 0.334 | 0.282 | 0.299 | −0.058 | +0.039 | −0.012 | +0.005 |
| detection | motion_blur | med | 17.7 | 0.352 | 0.157 | **0.257** | **0.211** | **0.261** | −0.195 | **+0.100** | **+0.054** | **+0.104** |
| detection | motion_blur | high | 15.6 | 0.352 | 0.059 | **0.132** | **0.113** | **0.189** | −0.293 | **+0.073** | **+0.054** | **+0.131** |
| detection | salt_pepper | low | 18.7 | 0.352 | 0.284 | **0.339** | 0.292 | 0.299 | −0.068 | **+0.055** | +0.008 | +0.015 |
| detection | salt_pepper | med | 11.8 | 0.352 | 0.167 | **0.336** | **0.255** | **0.297** | −0.185 | **+0.169** | **+0.088** | **+0.130** |
| detection | salt_pepper | high | 8.2 | 0.352 | 0.079 | **0.321** | **0.202** | **0.288** | −0.274 | **+0.242** | **+0.124** | **+0.210** |
| features | gauss_noise | low | 19.5 | 1.000 | 0.834 | 0.847 | — | — | −0.166 | +0.013 | — | — |
| features | gauss_noise | med | 13.9 | 1.000 | 0.731 | 0.758 | — | — | −0.269 | +0.027 | — | — |
| features | gauss_noise | high | 10.0 | 1.000 | 0.616 | **0.658** | — | — | −0.384 | **+0.042** | — | — |
| features | motion_blur | low | 20.8 | 1.000 | 0.643 | **0.781** | — | — | −0.357 | **+0.138** | — | — |
| features | motion_blur | med | 17.7 | 1.000 | 0.399 | **0.626** | — | — | −0.601 | **+0.227** | — | — |
| features | motion_blur | high | 15.6 | 1.000 | 0.197 | **0.476** | — | — | −0.803 | **+0.280** | — | — |
| features | salt_pepper | low | 18.7 | 1.000 | 0.554 | **0.705** | — | — | −0.446 | **+0.151** | — | — |
| features | salt_pepper | med | 11.8 | 1.000 | 0.404 | **0.692** | — | — | −0.596 | **+0.288** | — | — |
| features | salt_pepper | high | 8.2 | 1.000 | 0.326 | **0.662** | — | — | −0.674 | **+0.336** | — | — |
| keypoints | gauss_noise | low | 19.5 | 0.657 | 0.590 | 0.619 | — | — | −0.068 | +0.029 | — | — |
| keypoints | gauss_noise | med | 13.9 | 0.657 | 0.489 | **0.555** | — | — | −0.168 | **+0.066** | — | — |
| keypoints | gauss_noise | high | 10.0 | 0.657 | 0.369 | **0.479** | — | — | −0.288 | **+0.110** | — | — |
| keypoints | motion_blur | low | 20.8 | 0.657 | 0.578 | **0.620** | — | — | −0.080 | **+0.043** | — | — |
| keypoints | motion_blur | med | 17.7 | 0.657 | 0.386 | **0.524** | — | — | −0.272 | **+0.138** | — | — |
| keypoints | motion_blur | high | 15.6 | 0.657 | 0.159 | **0.387** | — | — | −0.499 | **+0.229** | — | — |
| keypoints | salt_pepper | low | 18.7 | 0.657 | 0.590 | 0.623 | — | — | −0.068 | +0.034 | — | — |
| keypoints | salt_pepper | med | 11.8 | 0.657 | 0.438 | **0.614** | — | — | −0.219 | **+0.176** | — | — |
| keypoints | salt_pepper | high | 8.2 | 0.657 | 0.348 | **0.598** | — | — | −0.310 | **+0.250** | — | — |
| segmentation | gauss_noise | low | 19.5 | 0.410 | 0.359 | 0.386 | — | — | −0.050 | +0.027 | — | — |
| segmentation | gauss_noise | med | 13.9 | 0.410 | 0.294 | **0.350** | — | — | −0.116 | **+0.056** | — | — |
| segmentation | gauss_noise | high | 10.0 | 0.410 | 0.209 | **0.299** | — | — | −0.201 | **+0.090** | — | — |
| segmentation | motion_blur | low | 20.8 | 0.410 | 0.354 | 0.385 | — | — | −0.056 | +0.031 | — | — |
| segmentation | motion_blur | med | 17.7 | 0.410 | 0.231 | **0.311** | — | — | −0.179 | **+0.081** | — | — |
| segmentation | motion_blur | high | 15.6 | 0.410 | 0.123 | **0.204** | — | — | −0.287 | **+0.081** | — | — |
| segmentation | salt_pepper | low | 18.7 | 0.410 | 0.290 | **0.378** | — | — | −0.120 | **+0.088** | — | — |
| segmentation | salt_pepper | med | 11.8 | 0.410 | 0.150 | **0.373** | — | — | −0.260 | **+0.223** | — | — |
| segmentation | salt_pepper | high | 8.2 | 0.410 | 0.105 | **0.348** | — | — | −0.304 | **+0.243** | — | — |

---

### Plot 1 — Recovery per Distortion × Severity Cell

![recovery per cell](results/figures/recovery_bars.png)

Each panel corresponds to one task (detection, features, keypoints, segmentation). Within each panel, bar groups correspond to the nine distortion × severity cells (three distortions × three severities). Blue bars show the raw distorted metric; green bars show the metric after classical enhancement; amber bars show the metric after fine-tuning (detection only). The dashed horizontal line marks the clean baseline for that task. Each bar is labeled with the fraction of the total damage that the repair recovered (e.g., +67% means two-thirds of the degradation was undone).

---

### Per-Class Detection AP — Clean and High-Severity Corruptions

**Clean baseline**

![per-class AP clean](results/figures/per_class_ap_clean.png)

The clean-baseline figure shows AP for all 80 COCO object classes, sorted from highest to lowest. The dashed horizontal line marks the overall mAP and establishes the per-class reference for the corruption comparisons below.

**High-severity corruptions**

**Gaussian noise**

![per-class AP under Gaussian noise](results/figures/per_class_ap_gauss_noise_high.png)

**Salt-and-pepper noise**

![per-class AP under salt-and-pepper noise](results/figures/per_class_ap_salt_pepper_high.png)

**Motion blur**

![per-class AP under motion blur](results/figures/per_class_ap_motion_blur_high.png)

Each figure compares the 15 COCO classes with the highest clean AP at high corruption severity. Gray bars show the clean detection baseline, blue bars the distorted AP, green bars the AP after matched classical enhancement, and amber bars the AP of the fine-tuned YOLOv8n detector. Classes are sorted by clean AP; the `n=` label gives the number of non-crowd ground-truth instances in the 1,521-image subset.

---

### Plot 3 — Task Metric vs SNR, per Distortion

![detection vs SNR](results/figures/acc_vs_snr_detection.png)
![features vs SNR](results/figures/acc_vs_snr_features.png)
![keypoints vs SNR](results/figures/acc_vs_snr_keypoints.png)
![segmentation vs SNR](results/figures/acc_vs_snr_segmentation.png)

One figure per task; within each figure, one panel per distortion type (Gaussian noise, salt-and-pepper, motion blur), with a shared y-axis. The x-axis is the measured per-image SNR (dB) of each severity level; lower SNR corresponds to higher severity. Series: distorted (blue, solid ●), enhanced (green, dashed ■), fine-tuned (amber, dotted ▲; detection only), and the clean baseline (gray dashed line). Each severity step is a single point on the x-axis because all images within a severity share the same distribution of sampled degradation parameters.

---

### Plots 5–8 — Annotated Prediction Examples by Task

The following figures overlay predictions, ground truth, segmentation masks, or feature matches on images from the benchmark subset for all three distortion types at high severity. They provide a direct visual comparison of clean, corrupted, and restored outputs, including fine-tuned detection where applicable.

**Plot 5 — Object detection (YOLOv8n):** Gray dashed boxes = COCO ground-truth bounding boxes; solid colored boxes = YOLO predictions with class label and confidence score. Columns: clean / distorted (high severity) / enhanced / distorted with the fine-tuned model.

![annotated gauss noise](results/figures/annotated_gauss_noise.png)
![annotated salt & pepper](results/figures/annotated_salt_pepper.png)
![annotated motion blur](results/figures/annotated_motion_blur.png)

**Plot 6 — Keypoint detection (Keypoint R-CNN):** Gray dashed lines and dots = ground-truth skeleton from the COCO keypoints annotation; white skeleton + yellow joint dots = model predictions at confidence ≥ 0.35. Columns: clean / distorted (high severity) / enhanced.

![keypoints gauss noise](results/figures/keypoints_gauss_noise.png)
![keypoints salt & pepper](results/figures/keypoints_salt_pepper.png)
![keypoints motion blur](results/figures/keypoints_motion_blur.png)

**Plot 7 — Panoptic segmentation (Panoptic FPN):** Color = semantic category (consistent palette across all panels); white hairlines = segment boundaries; each column header shows the cell's PQ value. Columns: ground truth / clean / distorted (high severity) / enhanced.

![panoptic gauss noise](results/figures/panoptic_gauss_noise.png)
![panoptic salt & pepper](results/figures/panoptic_salt_pepper.png)
![panoptic motion blur](results/figures/panoptic_motion_blur.png)

**Plot 8 — ORB feature matching:** Each image pair shows the clean image (left) matched against the distorted or enhanced variant (right). Lines connect good matches (Hamming distance ≤ 64, cross-checked — the same criterion as the metric). The caption below each pair is the actual per-image match ratio (good matches / clean keypoints). Two or three example pairs are shown per distortion row.

![orb matches gauss noise](results/figures/orb_matches_gauss_noise.png)
![orb matches salt & pepper](results/figures/orb_matches_salt_pepper.png)
![orb matches motion blur](results/figures/orb_matches_motion_blur.png)

---

### Plots 9–11 — Raw Noise and Enhancement Examples

Clean / distorted (high severity) / enhanced image grids for each distortion type, showing images **different from those used in Plots 5–8**. These show directly why the metrics behave as they do at the pixel level — without model predictions, so the pixel-level effect of each corruption and its restoration is visible in isolation.

![gauss noise grid](results/figures/grid_gauss_noise.png)
![salt & pepper grid](results/figures/grid_salt_pepper.png)
![motion blur grid](results/figures/grid_motion_blur.png)

---

## Conclusions and Discussion

### Classical enhancement is the best standalone repair in all 9 detection cells

For detection, classical enhancement outperforms fine-tuning alone in every distortion × severity cell. When the stacked strategy (enhancement + fine-tuning) is also considered, enhancement produces the highest score in 7 of 9 cells, while the stack wins on medium and high motion blur. Enhancement also improves over the raw distorted baseline in all 36 cells across all four tasks. Three structural reasons help explain this:

1. **Restoration uses known corruption information; the fine-tuned model does not.** BM3D derives its noise level from the recorded Gaussian variance, and Wiener deconvolution reconstructs a matched kernel from the recorded size and angle; the median filter is fixed and uses no per-image parameter. The fine-tuned detector receives no corruption label or parameter at inference time, so one set of weights must handle every input domain. Thus, in this controlled experiment, matched restoration has an information advantage over fine-tuning alone. This advantage may not hold for mixed, spatially varying, or unknown real-world corruptions, which were not evaluated here.

2. **A plausible explanation is distribution alignment.** The pretrained models were trained on COCO images without the synthetic corruptions used in this benchmark. Matched restoration makes corrupted inputs more similar to those training images. Fine-tuning instead adapts YOLOv8n to a 9,000-image mixed-domain subset; it improves on the pretrained detector in 7 of 9 corrupted cells but reduces clean mAP from 0.3521 to 0.3104 (−0.0417).

3. **Enhancement is reusable across models; fine-tuning is model-specific.** The same restored image sets are consumed unchanged by all four task pipelines and improve on their raw distorted scores in every cell. The fine-tuned checkpoint, by contrast, applies only to YOLOv8n detection; adapting the other learned models would require separate training.

### Recovery completeness is determined by how much information the corruption destroys

Enhancement recovers very different fractions of the damage depending on corruption type, and this reflects the fundamental invertibility of each forward model:

- **Salt-and-pepper** is the most recoverable: the median filter restores 80–91% of the detection loss and 74–88% of the segmentation loss at high severity. Each corrupted pixel is an isolated outlier surrounded by uncorrupted neighbors, so the median can discard it exactly and replace it with a true local value. Almost no information is permanently lost.
- **Gaussian noise** reaches 67–74% recovery on detection with non-blind BM3D. Additive white noise spreads energy uniformly across all frequencies; a strong denoiser can suppress most of it, but the lowest-amplitude signal components (fine texture, small edges) are irreversibly masked even at the best denoising strength. Hence recovery is strong but not complete.
- **Motion blur** achieves only 25–45% recovery at high severity with matched Wiener deconvolution. A linear motion kernel strongly attenuates some frequencies and has zeros at others, so stable linear inversion cannot restore those components; Wiener deconvolution instead balances partial inversion against amplification of residual errors. This helps explain why stacking enhancement with fine-tuning gives the best detection result for heavy motion blur: Wiener recovers the components that can be restored reliably, while the nonlinear detector can use learned semantic priors and the remaining image context to infer task-relevant structure. It does not reconstruct the truly missing frequencies or violate the information limit, but it can recover detection performance without exactly recovering the original image.

The ordering — salt-and-pepper ≫ Gaussian ≫ motion blur — reflects how permanently each corruption destroys image information, not how severe it looks by eye or by SNR.

### Tasks have different sensitivity profiles across corruption types

The same corruption at the same SNR damages different tasks to very different degrees, revealing what each metric actually measures:

- **Motion blur** is by far the most damaging corruption for localization-sensitive tasks: ORB match ratio falls −0.80 and OKS AP falls −0.50 at high severity, both worse than any noise corruption despite motion blur having a *higher* SNR. Blur removes the sharp edges and fine texture that ORB descriptors and keypoint localization depend on. Detection mAP and PQ also suffer badly (−0.29, −0.29), but their losses are closer to those from the noise corruptions.
- **Salt-and-pepper** hits all tasks roughly equally in absolute degradation terms, but the median filter rescues them equally well too — making it the corruption with the smallest *net* impact after enhancement.
- **Gaussian noise** is most damaging to ORB (−0.38 at high) relative to what enhancement recovers (+0.04), because even after BM3D removes the noise the enhanced image still differs from the clean one in fine texture — and ORB penalises that difference by construction. The GT-scored tasks (detection, keypoints, segmentation) recover much better (+0.17, +0.11, +0.09 at high severity) because a smoothed image can still contain the semantic structure needed to localise objects and joints.

The practical implication is that corruption sensitivity depends on the information required by the task. Methods that rely on sharp edges, local geometry, or precise spatial localization are especially vulnerable to motion blur. Methods or metrics that depend on accurate pixel values and fine texture can instead be strongly affected by Gaussian and salt-and-pepper noise. SNR alone is therefore insufficient; the relevant question is whether the task depends primarily on geometric structure, radiometric fidelity, or semantic context.

### Fine-tuning becomes more useful as corruption severity increases

The fine-tuned detector's absolute mAP still decreases as corruption becomes stronger, but its improvement over the pretrained detector on the same corrupted inputs grows with severity. Averaged across the three corruption types, fine-tuning recovery rises from −0.005 mAP at low severity to +0.060 at medium and +0.088 at high severity. The increase is monotonic for Gaussian and salt-and-pepper noise; for motion blur, recovery plateaus at approximately +0.054 for both medium and high severity.

This pattern is plausible because mild corruption leaves the inputs relatively close to the pretrained model's training domain, so the original detector remains strong and fine-tuning has little room to help; its reduced clean accuracy can even make low-severity performance slightly worse. As corruption intensifies, the domain gap and the pretrained model's degradation grow, so exposure to corrupted and restored training examples provides a larger relative benefit. The motion-blur plateau suggests a limit to adaptation alone: once enough spatial detail has been removed, fine-tuning cannot reconstruct the missing image information.

Fine-tuning alone still trails classical enhancement in all 9 detection cells. Its clearest complementary value appears when the two methods are stacked: the combined strategy produces the highest mAP for medium and high motion blur. At high motion blur, it recovers 44.5% of the clean-to-distorted loss, compared with 24.9% for enhancement alone and 18.3% for fine-tuning alone. Mixed, spatially varying, and previously unseen corruptions were not evaluated, so the results do not establish how either method would rank in those settings.

### SNR tracks severity within a corruption type, but not damage across types

Within each corruption type, task performance decreases monotonically as mean SNR falls and severity increases, supporting the internal consistency of the severity settings. However, SNR alone is not comparable across corruption types as a complete predictor of task damage. High motion blur (15.6 dB) causes the largest losses for ORB (−0.80) and keypoints (−0.50), despite having higher SNR than high Gaussian noise (10.0 dB) and high salt-and-pepper noise (8.2 dB). In this experiment, preservation of spatial structure is therefore more informative than pixel-wise SNR for geometry-sensitive tasks.

### The results confirm and deepen expected algorithm behavior

The benchmark confirms published intuitions — median filter rescues impulse noise, BM3D is the strongest Gaussian denoiser, known-kernel Wiener is optimal for blur — while quantifying *how much* of each task's damage each repair recovers, and exposing the conditions under which the ranking flips. The result is a practical decision matrix: corruption known and parameterizable → restore classically; heavy or partially-repairable → stack; unknown → fine-tune alone. This is not an academic distinction: it maps directly to deployment choices in autonomous driving, medical imaging, and surveillance systems where the noise source may or may not be known at inference time.

### Conclusion

Image distortion inflicts substantial but largely recoverable damage on modern vision systems when the restoration method matches the degradation model. Classical enhancement is the strongest standalone repair in all 9 detection cells and improves the distorted baseline in all 36 cells across the four tasks. When enhancement and fine-tuning are stacked, the combination produces the best result on medium and high motion blur. The project also demonstrates the diagnostic power of metric decomposition (PQ → SQ/RQ, AP → small/large): by examining *where* recovery fails we located and fixed a denoiser that actively hurt the system. Together, the four phases form a repeatable robustness-evaluation framework for controlled image corruptions.

---

## Appendix: Previous Versions and Known Failures

### v1 — Single-Cell Fine-tuning

**What was tried:** YOLOv8n fine-tuned on a single corruption cell (gauss_noise/high only).

**What failed:** Textbook **negative transfer**. The model more than doubled in-domain mAP (gauss_noise/high) but lost to the pretrained model on every motion-blur cell and every low-severity cell — recovery of −0.02…−0.10 on 5 of 9 cells. Training on one cell shifts weights toward one corruption distribution, making the model actively worse on all others.

**Lesson:** Single-cell fine-tuning is not a valid robustness strategy. A mixture covering all target corruptions is necessary.

Full v1 numbers: [docs/archive/](docs/archive/)

### v2 — NLM Denoiser + 10%-Clean Mixture

**What was tried:** (1) Sigma-adaptive NLM as the Gaussian noise enhancement; (2) a 10% clean / 90% distorted training mixture with no restored images in fine-tuning.

**What failed — NLM denoiser:** NLM scored *below the raw noisy images* on every pixel-precise task (keypoints, panoptic, ORB match ratio). Three concurrent mechanisms were identified:
- NLM output retained more high-frequency energy than the clean image (blotchy residual noise rather than clean signal)
- Damage concentrated on small objects (small-AP fell while large-AP rose under NLM)
- Texture-defined "stuff" classes lost entire segments (panoptic RQ collapsed while SQ barely moved — whole regions went missing, not sloppy boundaries)

BM3D's collaborative patch filtering removes ~4–5 dB more noise while keeping texture, flipping all 12 Gaussian-noise cells positive. Full diagnosis: [docs/archive/v2_nlm_gauss_enhance.md](docs/archive/v2_nlm_gauss_enhance.md)

**What failed — 10%-clean mixture:** Clean accuracy dropped by −0.068 mAP (vs. −0.042 for v3). More critically, the model never trained on restored images — so on the enhanced val images a real deployment would feed it, the fine-tuned model scored *below the pretrained model* at every severity (domain mismatch). The stack was non-additive. Full v2 fine-tuning numbers: [docs/archive/v2_finetune_mixture.md](docs/archive/v2_finetune_mixture.md)

**Lesson:** The filter matters as much as the concept; NLM and BM3D behave very differently on texture-rich images. And the training distribution must include the deployment distribution — if the pipeline applies restoration before inference, training without restored images creates a domain gap that erases the enhancement benefit.

---

## Environments

Two virtualenvs are required because detectron2 needs an older torch than the main stack:

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
.venv/bin/python     -c "import torch,torchvision,cv2,ultralytics,pycocotools; print(torch.cuda.is_available())"
.venv-det/bin/python -c "import torch,detectron2; from panopticapi.evaluation import pq_compute; print(torch.cuda.is_available())"
```

> The detectron2 env builds from source; for GPU support, build it on a node that has the CUDA toolkit (`nvcc`).

---

## How to Run Each Phase

The orchestrator `scripts/run_pipeline.py` is **resumable** (skips work whose output already exists; add `--force` to recompute). Phases map to stages: `data, distort, enhance, infer, eval, finetune, report`.

### Everything at once
```bash
python scripts/run_pipeline.py          # phases 0–5
sbatch slurm/pipeline.sbatch            # same, on a GPU node
```

### Phase 0 — Setup & Data
```bash
python -m src.data                      # annotations + subset images (+ panoptic GT)
```

### Phase 1 — Clean Baseline
```bash
python scripts/run_pipeline.py --only infer eval
# or manually:
python -m src.inference --task detection --variant clean
python -m src.inference --task keypoints --variant clean
python -m src.metrics   --task segmentation --variant clean
python -m src.metrics   --task detection    --variant clean
python -m src.metrics   --task keypoints    --variant clean
python -m src.metrics   --task features     --variant clean
```

### Phase 2 — Distortion
```bash
python -m src.distortions               # writes data/distorted/** + snr_index.csv
python -m src.inference --task detection --variant distorted --dtype motion_blur --severity high
python -m src.metrics   --task detection --variant distorted --dtype motion_blur --severity high
```

### Phase 3 — Enhancement
```bash
python -m src.enhancement               # writes data/enhanced/**
python -m src.inference --task detection --variant enhanced --dtype gauss_noise --severity high
python -m src.metrics   --task detection --variant enhanced --dtype gauss_noise --severity high
```

### Phase 4 — Fine-tuning (YOLOv8)
```bash
python -m src.finetune_det --mode both  # train + evaluate
sbatch slurm/finetune.sbatch            # on a GPU node
```

### Phase 5 — Report
```bash
python -m src.tables                    # comparison.csv / comparison.md / summary_long.csv
python -m src.visualize                 # all figures
```

---

## Repository Structure

```
.
├── configs/
│   └── config.yaml          # Central experiment configuration
├── src/
│   ├── config.py            # Configuration and path helpers
│   ├── data.py              # COCO download and seeded subset creation
│   ├── distortions.py       # Corruption generation and SNR measurement
│   ├── enhancement.py       # BM3D, median, and Wiener restoration
│   ├── models.py            # YOLOv8 and Keypoint R-CNN loaders
│   ├── inference.py         # Detection and keypoint inference
│   ├── segmentation.py      # Panoptic FPN inference and PQ evaluation
│   ├── metrics.py           # ORB, detection, keypoint, and panoptic evaluation
│   ├── finetune_det.py      # YOLOv8 fine-tuning and evaluation
│   ├── tables.py            # Result aggregation and comparison tables
│   └── visualize.py         # Plots and annotated examples
├── scripts/
│   ├── run_pipeline.py      # Resumable pipeline orchestrator
│   └── readme_table.py      # README result-table generator
├── slurm/
│   └── *.sbatch             # Cluster jobs for preparation, inference, and training
├── data/
│   ├── splits/              # Committed train/validation image-ID manifests
│   ├── coco/                # Downloaded COCO data (gitignored)
│   ├── distorted/           # Generated corrupted images (gitignored)
│   └── enhanced/            # Generated restored images (gitignored)
├── results/
│   ├── preds/               # Model predictions
│   ├── metrics/             # Per-cell metrics and comparison tables
│   └── figures/             # Generated plots and visual examples
├── docs/
│   └── archive/             # Previous experiments and failure analyses
├── slides/
│   └── slide_script.md
├── models/                  # Generated fine-tuned checkpoints (gitignored)
├── requirements.txt
└── README.md
```

## Configuration

All knobs live in [configs/config.yaml](configs/config.yaml):
- `dataset.val_subset_size` / `train_subset_size`, `seed`
- `distortions:` — the three corruptions × 3 severities with per-severity parameters
- `enhancement:` — `gauss_method: bm3d|nlm` and the CPU worker fan-out
- `tasks:` — `[detection, keypoints, segmentation]`
- `yolo.weights` — `yolov8n.pt` (bump to `yolov8s/m.pt` for accuracy)
- `finetune:` — epochs / batch / imgsz / lr0 / val_fraction / `clean_fraction` / `restored_fraction`
- `segmentation:` — detectron2 venv python path + Panoptic FPN config

## Outputs

- `results/preds/{task}__{variant}.json` — predictions (segmentation also writes PNG masks)
- `results/metrics/{task}__{variant}.json` — per-(task, variant) metrics
- `results/metrics/snr_index.csv` — per-image SNR for every distortion/severity
- `results/metrics/summary_long.csv`, `comparison.csv`, `comparison.md` — aggregated tables
- `results/figures/*.png` — all figures (acc-vs-SNR curves, per-class AP bars, image grids)

---

## Deliverables

- This README as the project report (choices, methods, metrics, results, run instructions)
- Result tables and figures under [`results/`](results/)
- Source code, configuration, and SLURM scripts under [`src/`](src/) and [`slurm/`](slurm/)
- Fine-tuned checkpoint: [`models/yolov8_finetuned.pt`](models/yolov8_finetuned.pt)
- Final presentation: [`slides/`](slides/)
