# Final-Project-Image-Processing

## Project Description
The objective of this course project is to evaluate the robustness of image processing and vision algorithms/models on a public dataset. We use the COCO (Common Objects in Context) dataset and compare performance on clean images versus distorted images.

The main focus is on measuring robustness under noise and corruption, especially the drop in metrics such as mAP (mean Average Precision), mAP@0.50, and mAP@0.75. This also allows us to compare how different object categories and object sizes respond to distortion.

## Part 1: Dataset
We use the COCO validation set (Val2017) as the benchmark set and keep the training data unchanged. The clean validation set serves as the baseline, and the same annotations are used for all distorted versions of the images.

## Part 2: Tasks and Models
The project includes three tasks:
1. Object Detection
2. Keypoint Detection
3. Panoptic Segmentation

For each task, we will select a representative model or algorithm:
1. Object Detection: Faster R-CNN or a YOLO-style detector
2. Keypoint Detection: HRNet or OpenPose
3. Panoptic Segmentation: Mask R-CNN or Panoptic FPN

## Part 3: Distortions
We will generate distorted versions of the validation set using three corruption types that reflect real-world conditions:
1. Gaussian noise, because it models sensor noise and random intensity fluctuations that often appear in low-quality captures.
2. Salt-and-pepper noise, because it simulates sparse pixel corruption caused by transmission errors, dead pixels, or impulsive interference.
3. Motion blur, because it reflects camera shake and object movement, which are common in real-world image capture.

The clean and noisy images will be stored separately so that both sets share the same annotation files and ground truth objects.

## Part 4: Benchmarking Workflow
For each task and dataset combination, we will follow this workflow:
1. Baseline: measure performance on clean images.
2. Distortion: apply the selected distortions and measure the degradation of each method.
3. Compare the results using pycocotools and report the performance drop caused by each corruption.

## Part 5: Improvement Strategies
We will measure two approaches for improving robustness:
1. Enhance distorted images during pre-processing, for example by denoising or de-blurring before inference.
2. Fine-tune the models on distorted data for the deep-learning-based methods to improve robustness to corruption.

## Part 6: Evaluation and Robustness Analysis
In addition to the standard COCO metrics, we will report the absolute performance degradation, class-specific sensitivity, and the effect of corruption on small versus large objects. If needed, we can also extend the project to a COCO-C style robustness benchmark using multiple corruption types and severity levels to compute a mean corruption error score.
