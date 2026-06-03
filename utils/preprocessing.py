import numpy as np
import cv2
from typing import Tuple

def normalize_intensity(image: np.ndarray) -> np.ndarray:
    """
    Perform Min-Max normalization on the image intensity.

    Args:
        image (np.ndarray): Input image array.
    Returns:
        np.ndarray: Normalized image array in range [0, 1].
    """
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max - img_min == 0:
        return image
    return (image - img_min) / (img_max - img_min)

def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to the target size.

    Args:
        image (np.ndarray): Input image array.
        size (Tuple[int, int]): Target (height, width).
    Returns:
        np.ndarray: Resized image array.
    """
    return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)

def resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize mask to target size using nearest neighbor interpolation to preserve labels.

    Args:
        mask (np.ndarray): Input mask array.
        size (Tuple[int, int]): Target (height, width).
    Returns:
        np.ndarray: Resized mask array.
    """
    return cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)

def volume_to_slices(volume: np.ndarray) -> np.ndarray:
    """
    Ensure volume is in a format that can be easily sliced.

    Args:
        volume (np.ndarray): NIfTI volume.
    Returns:
        np.ndarray: Volume array.
    """
    # Volume is typically (H, W, D) or (D, H, W) depending on nibabel/NIfTI
    # We want to return as is, and the dataset class will handle the slicing axis.
    return volume
