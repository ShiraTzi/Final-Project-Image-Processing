# Final-Project-Image-Processing

## Project Description
The objective of this course project is to evaluate the robustness of image processing and vision algorithms/models on a public dataset. We use the COCO (Common Objects in Context) dataset and compare performance on clean images versus distorted images.

The main focus is on measuring robustness under noise and corruption, especially the drop in metrics such as mAP (mean Average Precision), mAP@0.50, and mAP@0.75. This also allows us to compare how different object categories and object sizes respond to distortion.

## Part 1: Dataset
We use the COCO validation set (Val2017) as the benchmark set and keep the training data unchanged. The clean validation set serves as the baseline, and the same annotations are used for all distorted versions of the images.

## Part 2: Tasks and Models
The project includes tasks at both the low level and the high level, as required by the course (the task set must cover low-level and high-level vision):

Low-level task:
1. Feature / Corner Detection (ORB or SIFT keypoints), evaluated by feature-matching accuracy between clean and distorted images. This task needs no manual ground truth: the clean-image features serve as the reference.

High-level tasks:
2. Object Detection
3. Keypoint Detection
4. Panoptic Segmentation

For each task, we will select a representative model or algorithm:
1. Feature / Corner Detection: ORB (classical) or SIFT, with a brute-force/ratio-test matcher
2. Object Detection: Faster R-CNN or a YOLO-style detector
3. Keypoint Detection: HRNet or OpenPose
4. Panoptic Segmentation: Mask R-CNN or Panoptic FPN

This satisfies the requirement of at least three tasks spanning low-level and high-level vision, while keeping at least one deep-learning model.

## Part 3: Distortions
We will generate distorted versions of the validation set using three corruption types that reflect real-world conditions:
1. Gaussian noise, because it models sensor noise and random intensity fluctuations that often appear in low-quality captures.
2. Salt-and-pepper noise, because it simulates sparse pixel corruption caused by transmission errors, dead pixels, or impulsive interference.
3. Motion blur, because it reflects camera shake and object movement, which are common in real-world image capture.

The clean and noisy images will be stored separately so that both sets share the same annotation files and ground truth objects.

## Part 4: Benchmarking Workflow
For each task and dataset combination, we will follow this workflow:
1. Baseline: measure performance on clean images.
2. Distortion: apply the selected distortions at multiple intensity levels and measure the degradation of each method.
3. Compare the results using pycocotools (for the high-level tasks) and feature-matching accuracy (for the low-level task), and report the performance drop caused by each corruption.

Per-intensity measurement (required): each distortion is applied at a range of severity levels, and we report performance as a function of distortion intensity, characterized by the signal-to-noise ratio (SNR). This produces accuracy-vs-SNR curves per task, in addition to per-class results.

## Part 5: Improvement Strategies
We will measure two approaches for improving robustness:
1. Enhance distorted images during pre-processing, for example by denoising or de-blurring before inference.
2. Fine-tune the models on distorted data for the deep-learning-based methods to improve robustness to corruption.

## Part 6: Evaluation and Robustness Analysis
We report results both per class and per distortion intensity (per SNR), as required:
- High-level tasks: standard COCO metrics (mAP, mAP@0.50, mAP@0.75) via pycocotools, plus absolute performance degradation, class-specific sensitivity, and the effect of corruption on small versus large objects.
- Low-level task: feature-matching accuracy between clean and distorted images as a function of SNR.

All metrics are plotted as accuracy-vs-SNR curves and summarized in per-class tables. We can also extend the project to a COCO-C style robustness benchmark using multiple corruption types and severity levels to compute a mean corruption error score.

## Deliverables and Documentation
The repository itself is the submission, so we will keep it complete and up to date:
- This README serves as the detailed report: documented choices (dataset, tasks, methods, distortions, enhancements, metrics) with links.
- Tables of results/metrics (per class and per SNR) and visualizations: input/output processing steps (image with annotation, before/after) and measurement plots (bar plots, accuracy-vs-SNR curves, comparisons).
- Code, data folders (clean and distorted), and saved model outputs/checkpoints.
- A final presentation (PPT and exported PDF) as an easy-to-read version of this README.
- Team registration on the course project page: team member names and emails, and the accessible GitHub repository URL.
