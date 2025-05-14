import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=4, logits=True, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.logits:
            # Binary cross-entropy loss with logits
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            # Binary cross-entropy loss with probabilities
            BCE_loss = -targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs)

        p_t = torch.exp(-BCE_loss)  # Calculate probabilities from loss
        focal_loss = (1 - p_t) ** self.gamma * BCE_loss

        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


'''        # Debugging prints (Remove in production)
        print(f"BCE Loss min: {BCE_loss.min()}, max: {BCE_loss.max()}")
        print(f"p_t min: {p_t.min()}, max: {p_t.max()}")
        print(f"alpha_t min: {alpha_t.min()}, max: {alpha_t.max()}")
        print(f"Focal Loss min: {focal_loss.min()}, max: {focal_loss.max()}")'''