import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from configs.config import get_config
from datasets.brain_dataset import BrainMRIDataset
from models.factory import ModelFactory
from losses.dice_bce_loss import DiceBCELoss
from trainers.trainer import Trainer
import os


def _require_cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is required for this project, but torch.cuda.is_available() is False. "
            "Run this on a machine with a working NVIDIA GPU and CUDA-enabled PyTorch."
        )

    device = torch.device("cuda")
    print("CUDA test passed")
    print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"  torch.cuda.device_count(): {torch.cuda.device_count()}")
    print(f"  selected device: {device}")
    print(f"  current device index: {torch.cuda.current_device()}")
    print(f"  device name: {torch.cuda.get_device_name(device)}")

    smoke_tensor = torch.empty(1, device=device)
    print(f"  smoke test tensor device: {smoke_tensor.device}")

    return device


def main():
    # Load Configuration
    config = get_config()
    device = _require_cuda_device()
    print(f"Using device: {device}")
    pin_memory = device.type == "cuda"
    persistent_workers = config.num_workers > 0
    train_dir = os.path.join(config.processed_data_root, "train")
    val_dir = os.path.join(config.processed_data_root, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        raise RuntimeError(
            "Processed train/val folders were not found. Run preprocess.py first so the dataset is split at the source. "
            f"Expected: {train_dir} and {val_dir}"
        )

    print(f"Loading Train dataset from: {train_dir}")
    print(f"Loading Val dataset from: {val_dir}")

    # 1. Dataset Preparation
    train_dataset = BrainMRIDataset(
        root_dir=train_dir,
        modalities=config.modalities,
        image_size=config.image_size,
        remove_empty_slices=config.remove_empty_slices
    )

    val_dataset = BrainMRIDataset(
        root_dir=val_dir,
        modalities=config.modalities,
        image_size=config.image_size,
        remove_empty_slices=config.remove_empty_slices
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    # 2. Model Creation
    model = ModelFactory.create(
        model_name=config.model_name,
        in_channels=config.in_channels,
        out_channels=config.out_channels
    )
    model = model.to(device)
    print(f"Model parameters are on: {next(model.parameters()).device}")

    # 3. Loss and Optimizer
    criterion = DiceBCELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 4. Training
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=device
    )

    history = trainer.fit()

    print("Training Complete.")

if __name__ == "__main__":
    main()
