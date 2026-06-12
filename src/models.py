"""Model loaders + ORB helpers.

Object detection uses YOLOv8 (ultralytics), COCO-pretrained.  YOLO predicts
contiguous 0-79 class indices, so COCO80_TO_91 maps them to real COCO category
ids for pycocotools; COCO91_TO_80 is the reverse (used to write YOLO-format
training labels from COCO GT).

Keypoint detection uses torchvision's Keypoint R-CNN (person, 17 keypoints).

ORB feature matching is taken from the reference pipeline (§2.3): the metric is
``good_matches / clean_keypoints`` between the clean image and a variant.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import cv2
import numpy as np


# COCO "80-class" (YOLO/contiguous) index -> COCO "91-class" category id.
# Standard mapping used by all COCO tooling.
COCO80_TO_91 = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
COCO91_TO_80 = {cid: i for i, cid in enumerate(COCO80_TO_91)}


def resolve_device(requested: str) -> str:
    import torch

    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[models] CUDA unavailable -> falling back to CPU")
        return "cpu"
    return requested


def yolo_device(requested: str):
    """ultralytics wants 0 / [0,1] / 'cpu' rather than 'cuda'."""
    dev = resolve_device(requested)
    return 0 if dev.startswith("cuda") else "cpu"


# --------------------------------------------------------------------------- #
# YOLOv8 object detector
# --------------------------------------------------------------------------- #
def get_yolo_model(weights: str = "yolov8n.pt"):
    """Load a (pretrained or fine-tuned) YOLOv8 model."""
    from ultralytics import YOLO

    return YOLO(weights)


# --------------------------------------------------------------------------- #
# torchvision keypoint detector
# --------------------------------------------------------------------------- #
def get_keypoint_model(device: str):
    """Keypoint R-CNN ResNet50-FPN, COCO weights (person, 17 keypoints)."""
    from torchvision.models.detection import (
        keypointrcnn_resnet50_fpn, KeypointRCNN_ResNet50_FPN_Weights,
    )

    weights = KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
    model = keypointrcnn_resnet50_fpn(weights=weights).eval().to(device)
    return model, weights.transforms(), weights.meta["categories"]


# --------------------------------------------------------------------------- #
# ORB (low-level feature / corner detection)
# --------------------------------------------------------------------------- #
def orb_keypoints_and_descriptors(img_rgb: np.ndarray, nfeatures: int = 800):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=nfeatures)
    kps, desc = orb.detectAndCompute(gray, None)
    return kps, desc


def orb_match_ratio(clean_rgb: np.ndarray, variant_rgb: np.ndarray,
                    nfeatures: int = 800, distance_threshold: int = 64) -> float:
    """good_matches / clean_keypoints between clean and a distorted/enhanced image.
    Reference §2.3.  Clean-vs-clean ≈ 1.0 by construction."""
    kps1, desc1 = orb_keypoints_and_descriptors(clean_rgb, nfeatures)
    kps2, desc2 = orb_keypoints_and_descriptors(variant_rgb, nfeatures)
    if desc1 is None or desc2 is None or len(kps1) == 0:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(desc1, desc2)
    good = [m for m in matches if m.distance <= distance_threshold]
    return len(good) / len(kps1)
