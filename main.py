import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from configs.config import get_config
from datasets.brain_dataset import BrainMRIDataset
from models.factory import ModelFactory
from losses.dice_bce_loss import DiceBCELoss
from trainers.trainer import Trainer
import os

def main():
    # Load Configuration
    config = get_config()
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Dataset Preparation
    # Note: In a real scenario, the data_root should contain the UPENN-GBM folders
    dataset = BrainMRIDataset(
        root_dir=config.data_root,
        modalities=config.modalities,
        image_size=config.image_size,
        remove_empty_slices=config.remove_empty_slices
    )

    # Split dataset
    train_size = int(config.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    # 2. Model Creation
    model = ModelFactory.create(
        model_name=config.model_name,
        in_channels=config.in_channels,
        out_channels=config.out_channels
    )

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
