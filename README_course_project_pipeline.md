# Image Processing / Vision Course Project

This project evaluates the robustness of image-processing and computer-vision methods under image distortions. The pipeline follows four phases:

1. **Baseline on clean images**
2. **Performance degradation on distorted images**
3. **Performance recovery after image restoration / enhancement**
4. **Fine-tuning a model on distorted images**

The example implementation uses the **ADE20K tiny** dataset and three vision tasks:

| Component | Choice |
|---|---|
| Dataset | `nateraw/ade20k-tiny` |
| Task 1 | Feature detection / matching |
| Task 2 | Semantic segmentation |
| Task 3 | Object detection |
| Feature method | ORB |
| Segmentation model | SegFormer `nvidia/segformer-b0-finetuned-ade-512-512` |
| Detection model | YOLOv8n |
| Distortions | Gaussian noise, severe JPEG compression, low light |
| Enhancements | Non-Local Means + Bilateral filter, Y-channel bilateral filtering, gamma correction + CLAHE |
| Metrics | IoU / mIoU, object-detection recall, ORB matching ratio, SNR |

> Note: the code below was transcribed and reconstructed from the screenshots in the project slides. In places where the screenshot edge cut off part of a line, the standard equivalent implementation was completed.

---

## 1. Environment setup

```bash
pip install numpy matplotlib pillow opencv-python datasets albumentations torch transformers ultralytics
```

Recommended imports:

```python
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torch.nn.functional as F
from datasets import load_dataset
import albumentations
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from ultralytics import YOLO
```

---

## 2. Repository structure

```text
project/
├── README.md
├── notebooks/
│   ├── 01_clean_baseline.ipynb
│   ├── 02_distortions.ipynb
│   ├── 03_enhancement.ipynb
│   └── 04_finetune_yolo.ipynb
├── src/
│   ├── data.py
│   ├── distortions.py
│   ├── enhancement.py
│   ├── metrics.py
│   └── models.py
├── outputs/
│   ├── clean/
│   ├── distorted/
│   ├── enhanced/
│   ├── metrics/
│   └── figures/
└── yolo_work/
    ├── images/train/
    ├── labels/train/
    └── data.yaml
```

---

# Phase 1 - Measure performance on clean images

## 1.1 Prepare dataset and sample images

The dataset is loaded from Hugging Face. The example samples four random images and their segmentation labels.

```python
random.seed(7)

ds = load_dataset("nateraw/ade20k-tiny", split="train")
N = len(ds)

idxs = random.sample(range(N), 4)
samples = [ds[i] for i in idxs]

images = [s["image"] for s in samples]
masks = [s["label"] for s in samples]
```

---

## 1.2 Visualize images with ADE20K labels

This helper overlays the segmentation mask on top of the RGB image.

```python
def overlay_mask(img_pil, mask_pil, alpha=0.45):
    img = np.array(img_pil.convert("RGB")).astype(np.float32)
    m = np.array(mask_pil).astype(np.int32)
    rng = np.random.default_rng(0)
    palette = rng.integers(0, 255, size=(256, 3), dtype=np.uint8)
    color = palette[(m % 256).astype(np.int32)]
    out = (img * (1 - alpha) + color.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)
    return Image.fromarray(out)

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, m in zip(axes, images, masks):
    ax.imshow(overlay_mask(img, m))
    ax.axis("off")
plt.tight_layout()
plt.show()
```

---

## 1.3 Run ORB feature detector

ORB is used as the low-level feature detector. The output image visualizes the detected keypoints.

```python
def orb_overlay(img_pil, nfeatures=800):
    img = np.array(img_pil.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps = orb.detect(gray, None)
    out = cv2.drawKeypoints(
        img,
        kps,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )
    return out, kps

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, img in zip(axes, images):
    out, kps = orb_overlay(img)
    ax.imshow(out)
    ax.set_title(f"ORB keypoints: {len(kps)}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

Clean-image sample results shown in the slides:

| Image | ORB keypoints |
|---|---:|
| Sample 1 | 761 |
| Sample 2 | 800 |
| Sample 3 | 800 |
| Sample 4 | 800 |

---

## 1.4 Run pretrained YOLO object detection

YOLOv8n is used as the pretrained object detector.

```python
model = YOLO("yolov8n.pt")

def yolo_overlay(img_pil, conf=0.25):
    img = np.array(img_pil.convert("RGB"))
    r = model.predict(img, conf=conf, verbose=False)[0]
    out = r.plot()
    return out, r

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, img in zip(axes, images):
    out, r = yolo_overlay(img)
    ax.imshow(out)
    ax.set_title(f"YOLO detections: {len(r.boxes)}")
    ax.axis("off")
plt.tight_layout()
plt.show()
```

Clean-image sample results shown in the slides:

| Image | YOLO detections |
|---|---:|
| Sample 1 | 1 |
| Sample 2 | 0 |
| Sample 3 | 2 |
| Sample 4 | 5 |

---

## 1.5 Run pretrained semantic segmentation

SegFormer is used for semantic segmentation on ADE20K classes.

```python
seg_processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
seg_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
seg_model.eval()


def predict(img_pil):
    inputs = seg_processor(images=img_pil, return_tensors="pt")
    with torch.no_grad():
        logits = seg_model(**inputs).logits
    up = F.interpolate(
        logits,
        size=(img_pil.height, img_pil.width),
        mode="bilinear",
        align_corners=False,
    )
    return up.argmax(1).squeeze().numpy().astype(np.int32)  # 0..149
```

---

## 1.6 Display clean image, ground truth, and prediction

```python
id2label = seg_model.config.id2label
NUM_CLS = 150
PALETTE = np.random.default_rng(42).integers(40, 230, size=(NUM_CLS, 3), dtype=np.uint8)


def colorize(mask_0idx):
    return PALETTE[np.clip(mask_0idx, 0, NUM_CLS - 1)]


def seg_overlay(img_rgb, mask_0idx, alpha=0.55):
    return (img_rgb * (1 - alpha) + colorize(mask_0idx) * alpha).clip(0, 255).astype(np.uint8)


preds = [predict(img) for img in images]
gt_arrs = [np.array(m) for m in masks]  # 0..150 ADE20K raw

fig, axes = plt.subplots(4, 3, figsize=(14, 16))
for j, title in enumerate(["Clean", "Ground Truth", "Prediction"]):
    axes[0, j].set_title(title, fontsize=12)

for i, img in enumerate(images):
    img_np = np.array(img.convert("RGB"))
    axes[i, 0].imshow(img_np)
    axes[i, 1].imshow(seg_overlay(img_np, np.clip(gt_arrs[i] - 1, 0, NUM_CLS - 1)))
    axes[i, 2].imshow(seg_overlay(img_np, preds[i]))
    for j in range(3):
        axes[i, j].axis("off")

plt.tight_layout()
plt.show()
```

---

## 1.7 Compute IoU / mIoU for segmentation

ADE20K ground-truth masks use `0` for unlabeled pixels and `1..150` for classes. The SegFormer prediction is already indexed as `0..149`, so the ground truth must be shifted by `-1`.

```python
def compute_ious(pred_0idx, gt_raw):
    """pred: 0..149 | gt: 0..150 (ADE20K: 0=unlabeled, 1..150=classes -> shift by -1)."""
    valid = gt_raw > 0
    gt_0 = gt_raw - 1
    ious = {}
    for c in np.unique(gt_0[valid]):
        p = pred_0idx == c
        g = (gt_0 == c) & valid
        union = int((p | g).sum())
        if union:
            ious[int(c)] = float((p & g).sum()) / union
    return ious


def mean_iou(preds, gt_arrs):
    all_ious = []
    for pred, gt in zip(preds, gt_arrs):
        all_ious.extend(compute_ious(pred, gt).values())
    return float(np.mean(all_ious)) if all_ious else 0.0

clean_miou = mean_iou(preds, gt_arrs)
print(f"Clean mIoU: {clean_miou:.3f}")
```

Clean baseline result shown in the slides:

| Metric | Value |
|---|---:|
| Segmentation mIoU | 0.472 |

---

# Phase 2 - Measure performance on distorted images

## 2.1 Define image distortions

Three distortions are applied to the same clean images:

1. Gaussian noise
2. Severe JPEG compression
3. Low light

```python
distortions = {
    "GaussNoise": albumentations.GaussNoise(var_limit=(500.0, 1500.0), p=1.0),
    "SevereJPEG": albumentations.ImageCompression(quality_lower=1, quality_upper=5, p=1.0),
    "LowLight": albumentations.RandomBrightnessContrast(
        brightness_limit=(-0.8, -0.6),
        contrast_limit=(0.0, 0.0),
        p=1.0,
    ),
}


def apply_aug(img_pil, aug):
    img = np.array(img_pil.convert("RGB"))
    out = aug(image=img)["image"]
    return out


fig, axes = plt.subplots(4, len(distortions), figsize=(14, 14))
for row, img in enumerate(images):
    for col, (name, aug) in enumerate(distortions.items()):
        axes[row, col].imshow(apply_aug(img, aug))
        axes[row, col].axis("off")
        if row == 0:
            axes[row, col].set_title(name)
plt.tight_layout()
plt.show()
```

---

## 2.2 Evaluate segmentation under distortion

```python
def evaluate_segmentation_on_distortion(name, aug):
    distorted = [Image.fromarray(apply_aug(img, aug)) for img in images]
    dist_preds = [predict(img) for img in distorted]
    return mean_iou(dist_preds, gt_arrs)

seg_dist_results = {
    name: evaluate_segmentation_on_distortion(name, aug)
    for name, aug in distortions.items()
}

print(seg_dist_results)
```

Distorted-image segmentation results shown in the slides:

| Distortion | mIoU |
|---|---:|
| Gaussian noise | 0.181 |
| Severe JPEG | 0.534 |
| Low light | 0.353 |
| Clean baseline | 0.472 |

---

## 2.3 Evaluate ORB matching accuracy

The matching ratio compares features extracted from the clean image with features extracted from the distorted version.

```python
def orb_keypoints_and_descriptors(img_rgb, nfeatures=800):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps, desc = orb.detectAndCompute(gray, None)
    return kps, desc


def orb_match_ratio(clean_img_pil, distorted_rgb, nfeatures=800, distance_threshold=64):
    clean_rgb = np.array(clean_img_pil.convert("RGB"))
    kps1, desc1 = orb_keypoints_and_descriptors(clean_rgb, nfeatures=nfeatures)
    kps2, desc2 = orb_keypoints_and_descriptors(distorted_rgb, nfeatures=nfeatures)

    if desc1 is None or desc2 is None or len(kps1) == 0:
        return 0.0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    good = [m for m in matches if m.distance <= distance_threshold]
    return len(good) / len(kps1)


orb_dist_results = {}
for name, aug in distortions.items():
    ratios = []
    for img in images:
        distorted_rgb = apply_aug(img, aug)
        ratios.append(orb_match_ratio(img, distorted_rgb))
    orb_dist_results[name] = float(np.mean(ratios))

print(orb_dist_results)
```

ORB matching results shown in the slides:

| Distortion | Good matches / clean keypoints |
|---|---:|
| Gaussian noise | 0.284 |
| Severe JPEG | 0.959 |
| Low light | 0.106 |
| Clean baseline | 1.000 |

---

## 2.4 Evaluate YOLO detection degradation

The slides compare YOLO detections on clean images against detections on distorted images. The following utility treats clean YOLO boxes as the reference and computes detection recall using IoU.

```python
def boxes_xyxy_from_result(result):
    if result.boxes is None or len(result.boxes) == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return result.boxes.xyxy.cpu().numpy().astype(np.float32)


def box_iou_xyxy(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = np.maximum(0, a[:, 2] - a[:, 0]) * np.maximum(0, a[:, 3] - a[:, 1])
    area_b = np.maximum(0, b[:, 2] - b[:, 0]) * np.maximum(0, b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def detection_recall(ref_boxes, pred_boxes, iou_thr=0.5):
    if len(ref_boxes) == 0:
        return 1.0
    if len(pred_boxes) == 0:
        return 0.0
    ious = box_iou_xyxy(ref_boxes, pred_boxes)
    return float((ious.max(axis=1) >= iou_thr).mean())


clean_ref_boxes = []
for img in images:
    clean_rgb = np.array(img.convert("RGB"))
    r = model.predict(clean_rgb, conf=0.25, verbose=False)[0]
    clean_ref_boxes.append(boxes_xyxy_from_result(r))


yolo_dist_results = {}
for name, aug in distortions.items():
    recalls = []
    for img, ref in zip(images, clean_ref_boxes):
        distorted_rgb = apply_aug(img, aug)
        r = model.predict(distorted_rgb, conf=0.25, verbose=False)[0]
        pred_boxes = boxes_xyxy_from_result(r)
        recalls.append(detection_recall(ref, pred_boxes))
    yolo_dist_results[name] = float(np.mean(recalls))

print(yolo_dist_results)
```

---

## 2.5 Convert degradation level to SNR

SNR is used to quantify distortion intensity, especially in the low-light sweep.

```python
def compute_snr(clean_rgb, dark_rgb):
    """SNR (dB) = 10 * log10(signal_power / noise_power), noise = clean - dark."""
    clean = clean_rgb.astype(np.float64)
    noise = clean - dark_rgb.astype(np.float64)
    signal_power = np.mean(clean ** 2)
    noise_power = np.mean(noise ** 2)
    return 10.0 * np.log10(signal_power / noise_power) if noise_power > 0 else np.inf
```

---

## 2.6 Performance per SNR for low-light degradation

```python
def darken(img_pil, brightness):
    aug = albumentations.RandomBrightnessContrast(
        brightness_limit=(brightness, brightness),
        contrast_limit=(0.0, 0.0),
        p=1.0,
    )
    return apply_aug(img_pil, aug)


brightness_levels = [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.7, -0.8, -0.9]
snr_vals = []
recall_vals = []

for brightness in brightness_levels:
    lvl_snr, lvl_rec = [], []
    for i, img in enumerate(images):
        clean_rgb = np.array(img.convert("RGB"))
        dark_rgb = darken(img, brightness)

        lvl_snr.append(compute_snr(clean_rgb, dark_rgb))

        ref = clean_ref_boxes[i]
        r = model.predict(dark_rgb, conf=0.25, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else np.zeros((0, 4))
        lvl_rec.append(detection_recall(ref, boxes))

    snr_vals.append(float(np.mean(lvl_snr)))
    recall_vals.append(float(np.mean(lvl_rec)))

plt.figure(figsize=(10, 5))
plt.plot(snr_vals, recall_vals, marker="o")
plt.axhline(1.0, linestyle="--", label="Clean baseline 1.00")
plt.gca().invert_xaxis()
plt.xlabel("SNR (dB) <- darker")
plt.ylabel("Mean detection recall vs clean")
plt.title("YOLO detection recall vs SNR - low-light degradation sweep")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

# Phase 3 - Measure performance on restored / enhanced images

## 3.1 Enhancement functions

Each distortion receives a matching restoration method.

### Gaussian noise restoration

```python
def restore_noise(img_rgb):
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    # Strong NLM denoising: h=25 handles heavy noise; larger search window.
    den = cv2.fastNlMeansDenoisingColored(bgr, None, 25, 25, 7, 35)
    # Bilateral pass smooths residual grain while preserving edges.
    den = cv2.bilateralFilter(den, d=9, sigmaColor=80, sigmaSpace=80)
    out = cv2.cvtColor(den, cv2.COLOR_BGR2RGB)
    return out
```

### Severe JPEG restoration

```python
def restore_jpeg(img_rgb):
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y = cv2.bilateralFilter(y, d=7, sigmaColor=40, sigmaSpace=40)
    out = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)
    return out
```

### Low-light restoration

```python
def restore_lowlight(img_rgb):
    # Gamma correction lifts very dark pixels before local equalization.
    gamma = 0.35
    lut = (np.arange(256) / 255.0) ** gamma * 255
    lut = np.clip(lut, 0, 255).astype(np.uint8)
    img_gamma = cv2.LUT(img_rgb, lut)

    # Strong CLAHE on L channel for local contrast boost.
    lab = cv2.cvtColor(img_gamma, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    return out
```

```python
restorers = {
    "GaussNoise": restore_noise,
    "SevereJPEG": restore_jpeg,
    "LowLight": restore_lowlight,
}
```

---

## 3.2 Run ORB and YOLO on enhanced images

```python
def run_enhancement_eval():
    orb_enh_results = {}
    yolo_enh_results = {}

    for name, aug in distortions.items():
        restorer = restorers[name]
        orb_scores = []
        yolo_scores = []

        for i, img in enumerate(images):
            distorted_rgb = apply_aug(img, aug)
            enhanced_rgb = restorer(distorted_rgb)

            orb_scores.append(orb_match_ratio(img, enhanced_rgb))

            r = model.predict(enhanced_rgb, conf=0.25, verbose=False)[0]
            pred_boxes = boxes_xyxy_from_result(r)
            yolo_scores.append(detection_recall(clean_ref_boxes[i], pred_boxes))

        orb_enh_results[name] = float(np.mean(orb_scores))
        yolo_enh_results[name] = float(np.mean(yolo_scores))

    return orb_enh_results, yolo_enh_results


orb_enh_results, yolo_enh_results = run_enhancement_eval()
print("ORB enhanced:", orb_enh_results)
print("YOLO enhanced:", yolo_enh_results)
```

Enhanced-image comparison results shown in the slides:

### ORB feature matching

| Distortion | Distorted | Enhanced |
|---|---:|---:|
| Gaussian noise | 0.26 | 0.18 |
| Severe JPEG | 0.96 | 0.77 |
| Low light | 0.12 | 0.12 |
| Clean baseline | 1.00 | 1.00 |

### YOLO object detection

| Distortion | Distorted | Enhanced |
|---|---:|---:|
| Gaussian noise | 0.25 | 0.25 |
| Severe JPEG | 1.00 | 0.95 |
| Low light | 0.85 | 0.90 |
| Clean baseline | 1.00 | 1.00 |

---

# Phase 4 - Fine-tune YOLO on distorted images

The fine-tuning flow shown in the slides creates YOLO-format pseudo-labels from clean YOLO predictions, then fine-tunes YOLO on distorted/enhanced examples.

> Important: pseudo-labels are not true ground truth. They are useful for a small demonstration, but a full project should use real labels when available.

---

## 4.1 Create YOLO labels from clean images

YOLO label format:

```text
<class_id> <x_center> <y_center> <width> <height>
```

All coordinates are normalized to `[0, 1]` relative to image width and height.

```python
def save_yolo_label(txt_path, boxes_xyxy, cls_ids, w, h):
    lines = []
    for (x1, y1, x2, y2), c in zip(boxes_xyxy, cls_ids):
        cx = ((x1 + x2) / 2) / w
        cy = ((y1 + y2) / 2) / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        lines.append(f"{int(c)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    txt_path.write_text("\n".join(lines))
```

Example label lines shown in the slides:

```text
0 0.512344 0.601250 0.318750 0.427500
2 0.245000 0.540833 0.120000 0.186667
```

---

## 4.2 Build a YOLO training folder

```python
work = Path("yolo_work")
(work / "images/train").mkdir(parents=True, exist_ok=True)
(work / "labels/train").mkdir(parents=True, exist_ok=True)

pseudo_names = []
for i, img_pil in enumerate(images):
    img = np.array(img_pil.convert("RGB"))
    h, w = img.shape[:2]

    r = model.predict(img, conf=0.35, iou=0.5, verbose=False)[0]
    boxes = r.boxes
    if boxes is None or len(boxes) == 0:
        continue

    xyxy = boxes.xyxy.cpu().numpy()
    cls = boxes.cls.cpu().numpy()

    img_path = work / "images/train" / f"im_{i}.jpg"
    lbl_path = work / "labels/train" / f"im_{i}.txt"

    cv2.imwrite(str(img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    save_yolo_label(lbl_path, xyxy, cls, w, h)
    pseudo_names.append(f"im_{i}.jpg")
```

---

## 4.3 Create YOLO `data.yaml`

```python
# Use COCO-style class names from the pretrained YOLO model.
names = model.names
names_yaml = "\n".join([f"  {int(k)}: {v}" for k, v in names.items()])

(work / "data.yaml").write_text(
    f"""
path: {work.resolve()}
train: images/train
val: images/train
names:
{names_yaml}
""".strip()
)
```

---

## 4.4 Fine-tune YOLO

```python
ft_model = YOLO("yolov8n.pt")
_ = ft_model.train(
    data=str(work / "data.yaml"),
    imgsz=640,
    epochs=3,
    batch=2,
    device="cpu",
    verbose=False,
)

best = Path(ft_model.trainer.best) if hasattr(ft_model, "trainer") else (work / "runs/detect/train/weights/best.pt")
ft = YOLO(str(best))
```

For a GPU machine, replace:

```python
device="cpu"
```

with:

```python
device=0
```

---

## 4.5 Compare pretrained and fine-tuned models

```python
def yolo_count(model_obj, img_rgb, conf=0.25):
    r = model_obj.predict(img_rgb, conf=conf, verbose=False)[0]
    return r.plot(), r


example_img = images[2]
example_clean = np.array(example_img.convert("RGB"))
example_dist = apply_aug(example_img, distortions["GaussNoise"])
example_enh = restorers["GaussNoise"](example_dist)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, title, model_obj, img_rgb in [
    (axes[0], "Clean (pre)", model, example_clean),
    (axes[1], "Dist (pre)", model, example_dist),
    (axes[2], "Enh (pre)", model, example_enh),
    (axes[3], "Dist (ft)", ft, example_dist),
]:
    out, r = yolo_count(model_obj, img_rgb)
    ax.imshow(out)
    ax.set_title(f"{title} {len(r.boxes)}")
    ax.axis("off")

plt.tight_layout()
plt.show()
```

---

# Final reporting checklist

The final README should include:

- Dataset choice and links
- Chosen tasks and methods
- Distortions and enhancement methods
- Clean baseline visualizations and metrics
- Distorted image visualizations and metrics
- Enhanced image visualizations and metrics
- Fine-tuning setup and results
- Per-class segmentation IoU
- Performance per SNR curves
- Input/output image grids: clean, distorted, restored, predicted annotations
- Tables comparing clean vs distorted vs enhanced vs fine-tuned performance

---

# Project outcomes

| Stage | Expected artifact |
|---|---|
| Clean baseline | Outputs/labels on clean images and baseline metrics |
| Distortion evaluation | Distorted image folders, degradation tables, per-SNR plots |
| Enhancement evaluation | Restored images, side-by-side visual comparison, metrics after enhancement |
| Fine-tuning | Training code, YOLO labels, model checkpoint, post-finetuning metrics |
| Final submission | Detailed README, code, data/visuals, final presentation |

---

# Suggested weekly plan

| Week | Task | Artifact |
|---:|---|---|
| 1 | Form team, open Git, register | GitHub repo and course registration |
| 2 | Select dataset, distortions, tasks | Decision table in README |
| 3 | Select methods and enhancements | Method table in README |
| 4 | Download data and visualize annotations | EDA code and sample grids |
| 5 | Run models on clean data | Clean outputs and labels |
| 6 | Measure clean performance | Results tables and per-class plots |
| 7 | Apply distortions | Distorted data and before/after visualization |
| 8 | Measure degradation | Model outputs, tables, comparison plots |
| 9 | Apply enhancements | Restored images and performance comparison |
| 10 | Fine-tune models | Training code and model weights |
| 11 | Evaluate fine-tuned models | Tables and visualizations |
| 12 | Improve README | Rich final report with visuals and tables |
| 13 | Prepare final slides | Full repo, PPT, and PDF |
