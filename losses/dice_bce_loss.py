import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceBCELoss(nn.Module):
    """
    Combined Dice Loss and Binary Cross Entropy Loss.
    """
    def __init__(self, smooth: float = 1e-6):
        super(DiceBCELoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Prediction tensor.
            target (torch.Tensor): Ground truth tensor.
        Returns:
            torch.Tensor: Combined loss.
        """
        # BCE Loss
        bce = F.binary_cross_entropy_with_logits(pred, target)

        # Dice Loss
        pred_sigmoid = torch.sigmoid(pred)
        intersection = (pred_sigmoid * target).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (pred_sigmoid.sum() + target.sum() + self.smooth)

        return bce + dice_loss
