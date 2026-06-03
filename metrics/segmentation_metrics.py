import torch
import numpy as np
from typing import Tuple, Dict

def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute the Dice coefficient.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute the Intersection over Union (IoU).
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def precision_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute Precision.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp + smooth) / (tp + fp + smooth)

def recall_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
    """
    Compute Recall.
    """
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp + smooth) / (tp + fn + smooth)

def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Compute all segmentation metrics.
    """
    return {
        "dice": dice_score(pred, target).item(),
        "iou": iou_score(pred, target).item(),
        "precision": precision_score(pred, target).item(),
        "recall": recall_score(pred, target).item(),
    }
