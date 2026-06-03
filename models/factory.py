from typing import Dict, Type
from models.base_model import BaseSegmentationModel
from models.unet import UNet
from models.attention_unet import AttentionUNet

class ModelFactory:
    """
    Factory for creating segmentation models.
    """
    _models: Dict[str, Type[BaseSegmentationModel]] = {
        "unet": UNet,
        "attention_unet": AttentionUNet
    }

    @classmethod
    def create(cls, model_name: str, in_channels: int, out_channels: int) -> BaseSegmentationModel:
        """
        Create a model based on the provided name.
        """
        model_class = cls._models.get(model_name.lower())
        if model_class is None:
            raise ValueError(f"Model {model_name} not supported. Available: {list(cls._models.keys())}")

        return model_class(in_channels, out_channels)
