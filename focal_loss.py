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


class HierarchicalFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=4, logits=True, reduction='mean', 
                 hierarchy_matrix=None, reg_weight=0.3):
        """
        Args:
            hierarchy_matrix (torch.Tensor): Sparse tensor of shape [num_classes, num_classes]
                where hierarchy_matrix[i,j] = 1 if term i is a child of term j
            reg_weight (float): Weight for hierarchical consistency regularization
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduction = reduction
        self.reg_weight = reg_weight
        
        if hierarchy_matrix is not None:
            # Convert to sparse tensor and register buffer
            self.register_buffer('hierarchy_matrix', hierarchy_matrix.coalesce())
        else:
            self.hierarchy_matrix = None

    def forward(self, inputs, targets):
        # Original focal loss calculation
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            probs = torch.sigmoid(inputs)
        else:
            probs = inputs.clamp(min=1e-6, max=1-1e-6)
            bce_loss = -targets*torch.log(probs) - (1-targets)*torch.log(1-probs)
        
        p_t = torch.exp(-bce_loss)
        focal_loss = (1 - p_t)**self.gamma * bce_loss
        
        if self.alpha is not None:
            focal_loss = self.alpha * focal_loss

        # Hierarchical consistency regularization
        hierarchy_loss = 0
        if self.hierarchy_matrix is not None and self.training:
            # Get parent probabilities using matrix multiplication
            parent_probs = torch.sparse.mm(self.hierarchy_matrix, probs.t()).t()
            
            # Calculate violation where child probabilities > parent probabilities
            violations = F.relu(probs - parent_probs)
            
            # Exclude null parent relationships (where hierarchy_matrix was 0)
            valid_mask = (parent_probs > 0).float()
            hierarchy_loss = (violations**2 * valid_mask).mean()

        # Combine losses
        total_loss = focal_loss + self.reg_weight * hierarchy_loss

        if self.reduction == 'mean':
            return total_loss.mean()
        elif self.reduction == 'sum':
            return total_loss.sum()
        return total_loss