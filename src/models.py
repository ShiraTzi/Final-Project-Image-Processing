"""Model loaders.

Object detection uses YOLOv8 (ultralytics), COCO-pretrained.  YOLO predicts
contiguous 0-79 class indices, so COCO80_TO_91 maps them to real COCO category
ids for pycocotools; COCO91_TO_80 is the reverse (used to write YOLO-format
training labels from COCO GT).

Keypoint detection uses torchvision's Keypoint R-CNN (person, 17 keypoints).

Panoptic segmentation (Panoptic FPN, detectron2) lives in src/segmentation.py
and runs in a dedicated venv.
"""
from __future__ import annotations

from typing import Dict, List, Tuple


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
