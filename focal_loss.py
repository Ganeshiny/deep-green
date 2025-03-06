import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, logits=True, reduction='mean'):
        """
        Args:
            alpha (torch.Tensor): Class-specific weights (computed using class frequencies)
            gamma (float): Focusing parameter for modulating factor (1 - p_t)
            logits (bool): If True, applies BCE with logits; else, expects probabilities
            reduction (str): 'mean' | 'sum' | 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Model predictions (logits or probabilities)
            targets (torch.Tensor): Ground truth labels (same shape as inputs)
        Returns:
            torch.Tensor: Computed focal loss
        """
        if self.logits:
            # Compute binary cross-entropy loss with logits
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            probs = torch.sigmoid(inputs)  # Convert logits to probabilities
        else:
            # Compute binary cross-entropy loss with probabilities
            probs = inputs.clamp(min=1e-6, max=1-1e-6)
            BCE_loss = -targets * torch.log(probs) - (1 - targets) * torch.log(1 - probs)

        # Compute the focal loss term
        p_t = torch.exp(-BCE_loss)  # Probability of the correct class
        focal_loss = (1 - p_t) ** self.gamma * BCE_loss  # Apply focusing factor

        # Apply alpha (class-specific weighting)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)  # Applies alpha to positives only
        focal_loss = alpha_t * focal_loss

        # Apply reduction method
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
