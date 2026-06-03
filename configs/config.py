import os
from dataclasses import dataclass, field
from typing import List, Tuple

@dataclass
class Config:
    # --- Dataset Paths ---
    data_root: str = "data/UPENN-GBM"
    train_split: float = 0.8

    # --- Image Parameters ---
    image_size: Tuple[int, int] = (256, 256)
    # Possible modalities: "FLAIR", "T1", "T2", "T1CE"
    # User can configure which modalities to use
    modalities: List[str] = field(default_factory=lambda: ["FLAIR", "T1", "T2", "T1CE"])
    remove_empty_slices: bool = True

    # --- Model Parameters ---
    # Options: "unet", "attention_unet"
    model_name: str = "unet"
    in_channels: int = 4 # Matches len(modalities)
    out_channels: int = 1 # Binary segmentation

    # --- Training Parameters ---
    batch_size: int = 16
    learning_rate: float = 1e-4
    epochs: int = 50
    optimizer: str = "adam"
    weight_decay: float = 1e-5

    # --- Training Pipeline ---
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    device: str = "cuda" # CUDA-only training; main.py fails fast if CUDA is unavailable

    # --- Evaluation ---
    val_interval: int = 1 # Validate every N epochs

def get_config() -> Config:
    return Config()
