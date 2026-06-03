import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class BaseSegmentationModel(nn.Module, ABC):
    """
    Abstract base class for all segmentation models.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super(BaseSegmentationModel, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
