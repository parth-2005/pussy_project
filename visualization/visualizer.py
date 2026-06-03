import matplotlib.pyplot as plt
import torch
import numpy as np
from typing import Tuple

def visualize_predictions(image: torch.Tensor, target: torch.Tensor, pred: torch.Tensor):
    """
    Plot MRI slice, target mask, and prediction side by side.

    Args:
        image (torch.Tensor): Image tensor [C, H, W].
        target (torch.Tensor): Target mask [1, H, W].
        pred (torch.Tensor): Predicted mask [1, H, W].
    """
    # Convert to numpy and remove channel dim for grayscale
    # For image, if multiple channels, take the first one or mean
    img_np = image.cpu().numpy()
    if img_np.ndim == 3:
        img_np = np.mean(img_np, axis=0)

    target_np = target.cpu().numpy().squeeze()
    # Sigmoid and threshold for prediction
    pred_np = torch.sigmoid(pred).cpu().numpy().squeeze()
    pred_np = (pred_np > 0.5).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title("Original MRI")
    axes[0].axis('off')

    axes[1].imshow(target_np, cmap='gray')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(pred_np, cmap='gray')
    axes[2].set_title("Prediction")
    axes[2].axis('off')

    plt.tight_layout()
    plt.show()
