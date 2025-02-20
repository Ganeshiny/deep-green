import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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

