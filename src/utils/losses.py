"""
Loss functions for CAFA-6 protein function prediction
Includes focal loss for handling class imbalance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification
    Focuses training on hard examples and down-weights easy ones

    Reference: Lin et al. "Focal Loss for Dense Object Detection"
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Args:
            alpha: Weighting factor in [0,1] to balance positive/negative examples
            gamma: Focusing parameter for modulating loss (gamma >= 0)
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size, num_classes) - binary labels

        Returns:
            loss: scalar or (batch_size, num_classes) depending on reduction
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)

        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Calculate focal weight: (1 - p_t)^gamma
        # p_t is the probability of the true class
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Calculate alpha weight
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Combine focal and alpha weights
        focal_loss = alpha_weight * focal_weight * bce_loss

        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class WeightedBCELoss(nn.Module):
    """
    Weighted Binary Cross-Entropy Loss for multi-label classification
    Uses class weights to handle imbalance
    """

    def __init__(self, class_weights: torch.Tensor = None, reduction: str = 'mean'):
        """
        Args:
            class_weights: (num_classes,) - weight for each class
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size, num_classes) - binary labels

        Returns:
            loss: scalar
        """
        # Calculate BCE loss per element
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Apply class weights if provided
        if self.class_weights is not None:
            # Expand weights to match batch
            weights = self.class_weights.unsqueeze(0).to(inputs.device)
            bce_loss = bce_loss * weights

        # Apply reduction
        if self.reduction == 'mean':
            return bce_loss.mean()
        elif self.reduction == 'sum':
            return bce_loss.sum()
        else:
            return bce_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification
    Uses different focusing parameters for positive and negative samples

    Reference: Ridnik et al. "Asymmetric Loss For Multi-Label Classification"
    """

    def __init__(self,
                 gamma_neg: float = 4.0,
                 gamma_pos: float = 1.0,
                 clip: float = 0.05,
                 reduction: str = 'mean'):
        """
        Args:
            gamma_neg: Focusing parameter for negative samples
            gamma_pos: Focusing parameter for positive samples
            clip: Probability clipping value
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: (batch_size, num_classes) - logits
            targets: (batch_size, num_classes) - binary labels

        Returns:
            loss: scalar
        """
        # Apply sigmoid
        probs = torch.sigmoid(inputs)

        # Probability clipping
        if self.clip is not None and self.clip > 0:
            probs = probs.clamp(min=self.clip, max=1.0 - self.clip)

        # Calculate positive and negative losses separately
        # Positive loss (actual labels = 1)
        pos_loss = targets * torch.log(probs)
        pos_loss = pos_loss * ((1 - probs) ** self.gamma_pos)

        # Negative loss (actual labels = 0)
        neg_loss = (1 - targets) * torch.log(1 - probs)
        neg_loss = neg_loss * (probs ** self.gamma_neg)

        # Combine losses
        loss = -(pos_loss + neg_loss)

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class MultiTaskLoss(nn.Module):
    """
    Loss function for multi-task learning with separate aspect heads
    Combines losses from C, F, P predictions with learnable weights
    """

    def __init__(self,
                 num_classes_C: int,
                 num_classes_F: int,
                 num_classes_P: int,
                 loss_type: str = 'focal',
                 class_weights: dict = None,
                 learnable_weights: bool = True):
        """
        Args:
            num_classes_C/F/P: Number of GO terms per aspect
            loss_type: 'focal', 'bce', or 'asymmetric'
            class_weights: Dict with keys 'C', 'F', 'P' containing class weights
            learnable_weights: If True, learn task weights during training
        """
        super().__init__()

        # Create loss functions for each aspect
        if loss_type == 'focal':
            self.loss_C = FocalLoss()
            self.loss_F = FocalLoss()
            self.loss_P = FocalLoss()
        elif loss_type == 'bce':
            weights_C = class_weights['C'] if class_weights else None
            weights_F = class_weights['F'] if class_weights else None
            weights_P = class_weights['P'] if class_weights else None
            self.loss_C = WeightedBCELoss(weights_C)
            self.loss_F = WeightedBCELoss(weights_F)
            self.loss_P = WeightedBCELoss(weights_P)
        elif loss_type == 'asymmetric':
            self.loss_C = AsymmetricLoss()
            self.loss_F = AsymmetricLoss()
            self.loss_P = AsymmetricLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Task weights (uncertainty weighting from "Multi-Task Learning Using Uncertainty")
        if learnable_weights:
            # Initialize with log(sigma^2) for numerical stability
            self.log_var_C = nn.Parameter(torch.zeros(1))
            self.log_var_F = nn.Parameter(torch.zeros(1))
            self.log_var_P = nn.Parameter(torch.zeros(1))
            self.learnable = True
        else:
            # Fixed weights
            self.weight_C = 1.0
            self.weight_F = 1.0
            self.weight_P = 1.0
            self.learnable = False

    def forward(self,
                outputs: dict,
                targets: dict) -> tuple:
        """
        Args:
            outputs: Dict with keys 'C', 'F', 'P' containing logits
            targets: Dict with keys 'C', 'F', 'P' containing labels

        Returns:
            total_loss: Combined loss
            loss_dict: Dict with individual losses for logging
        """
        # Calculate individual losses
        loss_C = self.loss_C(outputs['C'], targets['C'])
        loss_F = self.loss_F(outputs['F'], targets['F'])
        loss_P = self.loss_P(outputs['P'], targets['P'])

        # Combine losses with weights
        if self.learnable:
            # Uncertainty weighting: L = 1/(2*sigma^2) * loss + log(sigma)
            precision_C = torch.exp(-self.log_var_C)
            precision_F = torch.exp(-self.log_var_F)
            precision_P = torch.exp(-self.log_var_P)

            total_loss = (
                precision_C * loss_C + self.log_var_C +
                precision_F * loss_F + self.log_var_F +
                precision_P * loss_P + self.log_var_P
            )
        else:
            total_loss = (
                self.weight_C * loss_C +
                self.weight_F * loss_F +
                self.weight_P * loss_P
            )

        # Create loss dict for logging
        loss_dict = {
            'loss_C': loss_C.item(),
            'loss_F': loss_F.item(),
            'loss_P': loss_P.item(),
            'total_loss': total_loss.item()
        }

        if self.learnable:
            loss_dict['weight_C'] = precision_C.item()
            loss_dict['weight_F'] = precision_F.item()
            loss_dict['weight_P'] = precision_P.item()

        return total_loss, loss_dict


def get_loss_function(loss_type: str, **kwargs):
    """
    Factory function to get loss function by name

    Args:
        loss_type: One of 'focal', 'bce', 'asymmetric', 'multitask'
        **kwargs: Loss-specific arguments

    Returns:
        Loss function
    """
    losses = {
        'focal': FocalLoss,
        'bce': WeightedBCELoss,
        'asymmetric': AsymmetricLoss,
        'multitask': MultiTaskLoss
    }

    if loss_type not in losses:
        raise ValueError(f"Unknown loss type: {loss_type}. Choose from {list(losses.keys())}")

    return losses[loss_type](**kwargs)


if __name__ == '__main__':
    # Test loss functions
    batch_size = 4
    num_classes = 100

    inputs = torch.randn(batch_size, num_classes)
    targets = torch.randint(0, 2, (batch_size, num_classes)).float()

    print("Testing FocalLoss...")
    focal_loss = FocalLoss()
    loss = focal_loss(inputs, targets)
    print(f"Loss: {loss.item():.4f}")

    print("\nTesting WeightedBCELoss...")
    class_weights = torch.rand(num_classes)
    weighted_bce = WeightedBCELoss(class_weights)
    loss = weighted_bce(inputs, targets)
    print(f"Loss: {loss.item():.4f}")

    print("\nTesting AsymmetricLoss...")
    asym_loss = AsymmetricLoss()
    loss = asym_loss(inputs, targets)
    print(f"Loss: {loss.item():.4f}")

    print("\nTesting MultiTaskLoss...")
    outputs = {
        'C': torch.randn(batch_size, 50),
        'F': torch.randn(batch_size, 50),
        'P': torch.randn(batch_size, 50)
    }
    targets = {
        'C': torch.randint(0, 2, (batch_size, 50)).float(),
        'F': torch.randint(0, 2, (batch_size, 50)).float(),
        'P': torch.randint(0, 2, (batch_size, 50)).float()
    }
    mt_loss = MultiTaskLoss(50, 50, 50, learnable_weights=True)
    total_loss, loss_dict = mt_loss(outputs, targets)
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss dict: {loss_dict}")
