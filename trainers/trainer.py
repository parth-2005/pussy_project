import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from typing import Dict, List, Any
from metrics.segmentation_metrics import compute_metrics
import os
from tqdm import tqdm

class Trainer:
    """
    Generic trainer for segmentation models.
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: Any,
        device: str = "cuda"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device

        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        all_metrics = []

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch in pbar:
            images, masks = batch
            images, masks = images.to(self.device), masks.to(self.device)

            self.optimizer.zero_grad()
            preds = self.model(images)
            loss = self.criterion(preds, masks)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

            with torch.no_grad():
                metrics = compute_metrics(preds, masks)
                all_metrics.append(metrics)

            pbar.set_postfix({"loss": loss.item()})

        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = self._aggregate_metrics(all_metrics)
        return {"loss": avg_loss, **avg_metrics}

    def validate(self) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        all_metrics = []

        with torch.no_grad():
            for batch in self.val_loader:
                images, masks = batch
                images, masks = images.to(self.device), masks.to(self.device)

                preds = self.model(images)
                loss = self.criterion(preds, masks)
                total_loss += loss.item()

                metrics = compute_metrics(preds, masks)
                all_metrics.append(metrics)

        avg_loss = total_loss / len(self.val_loader)
        avg_metrics = self._aggregate_metrics(all_metrics)
        return {"loss": avg_loss, **avg_metrics}

    def fit(self) -> Dict[str, List[float]]:
        history = {"train_loss": [], "val_loss": [], "train_dice": [], "val_dice": []}

        for epoch in range(1, self.config.epochs + 1):
            train_res = self.train_epoch(epoch)
            val_res = self.validate()

            history["train_loss"].append(train_res["loss"])
            history["val_loss"].append(val_res["loss"])
            history["train_dice"].append(train_res["dice"])
            history["val_dice"].append(val_res["dice"])

            print(f"Epoch {epoch}: Train Loss: {train_res['loss']:.4f}, Val Loss: {val_res['loss']:.4f}, Val Dice: {val_res['dice']:.4f}")

            if epoch % self.config.val_interval == 0:
                self.save_checkpoint(epoch, val_res["dice"])

        return history

    def save_checkpoint(self, epoch: int, metric: float):
        path = os.path.join(self.config.checkpoint_dir, f"model_epoch_{epoch}_dice_{metric:.4f}.pth")
        torch.save(self.model.state_dict(), path)

    def _aggregate_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        return {k: np.mean([m[k] for m in metrics_list]) for k in keys}
