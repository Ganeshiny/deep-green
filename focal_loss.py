import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha = None, gamma=4, logits=True, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha # Controls strenth of loss - Balances positive/negative classes
        self.gamma = gamma # Higher values penalize easy examples more
        self.logits = logits # Logit - True means its a raw score
        self.reduction = reduction # How to aggregate losses across batches

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')

        # calculate the focal loss components
        p_t = torch.exp(-BCE_loss) # Calculates the predicted probabilities based of BCE loss
        #High p_t - model is mostly corrent, low p_t model is wrong
        focal_loss = self.alpha * (1 - p_t)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

