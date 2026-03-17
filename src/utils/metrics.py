#src/utils/metrics.py
"""
Evaluation metrics computed on full-resolution padded masks.
"""

from __future__ import annotations
import torch
import numpy as np
from typing import Dict


def _binary(x: np.ndarray) -> np.ndarray:
    return (x > 0.5).astype(np.uint8)


def dice_coef(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred, gt = _binary(pred), _binary(gt)
    inter = (pred & gt).sum()
    union = pred.sum() + gt.sum()
    return (2.0 * inter + eps) / (union + eps)


def iou(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred, gt = _binary(pred), _binary(gt)
    inter = (pred & gt).sum()
    union = (pred | gt).sum()
    return (inter + eps) / (union + eps)


def sensitivity(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred, gt = _binary(pred), _binary(gt)
    tp = (pred & gt).sum()
    fn = ((~pred.astype(bool)) & gt.astype(bool)).sum()
    return (tp + eps) / (tp + fn + eps)


def specificity(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    pred, gt = _binary(pred), _binary(gt)
    tn = ((~pred.astype(bool)) & (~gt.astype(bool))).sum()
    fp = (pred.astype(bool) & (~gt.astype(bool))).sum()
    return (tn + eps) / (tn + fp + eps)


def compute_all(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    return {
        "dice": dice_coef(pred, gt),
        "iou": iou(pred, gt),
        "sensitivity": sensitivity(pred, gt),
        "specificity": specificity(pred, gt),
    }
