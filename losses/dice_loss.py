import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, target_class=1):
        """
        target_class: Specify the class index for which to compute Dice Loss.
        smooth: Smoothing factor to avoid division by zero.
        """
        super().__init__()
        self.smooth = smooth
        self.target_class = target_class

    def forward(self, logits, targets):
        # Convert logits to probabilities using softmax
        probs = torch.softmax(logits, dim=1)
        
        # Extract the probabilities for the target class
        probs = probs[:, self.target_class, :, :]  # Shape: [batch_size, height, width]
        
        # Ensure targets are floats for computation
        targets = targets.float()
        
        # Flatten tensors
        probs = probs.reshape(-1)
        targets = targets.reshape(-1)
        
        # Compute Dice coefficient
        intersection = (probs * targets).sum()
        dice = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice
