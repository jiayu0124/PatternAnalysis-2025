"""
utils.py
--------

This module provides common utility functions used by the training and
evaluation scripts.  In particular it includes:

* A Dice loss implementation combining soft Dice and cross entropy.
* A per‑class Dice coefficient calculator for reporting metrics.

These functions operate on raw model outputs (logits) and integer
labels.  They do not perform activation functions; the training
pipeline is expected to apply softmax as needed.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss for multi‑class segmentation.

    This implementation computes the Dice loss on probabilities
    (obtained via softmax) against a one‑hot encoding of the labels.
    Optionally it can be combined with a cross‑entropy term for
    improved stability.

    Parameters
    ----------
    weight : float, optional
        Relative weight of the Dice loss compared to cross entropy.
        When set to 1.0 (default) only Dice loss is used.
    smooth : float, optional
        Smoothing constant to avoid division by zero.  Defaults to 1e-5.
    """

    def __init__(self, weight: float = 1.0, smooth: float = 1e-5) -> None:
        super().__init__()
        self.weight = weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = preds.shape[1]
        # Cross entropy component
        ce_loss = self.ce(preds, targets)
        # Convert targets to one‑hot format
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 4, 1, 2, 3).float()
        # Apply softmax to get class probabilities
        probs = F.softmax(preds, dim=1)
        # Compute Dice per class
        intersection = (probs * targets_one_hot).sum(dim=(2, 3, 4))
        union = probs.sum(dim=(2, 3, 4)) + targets_one_hot.sum(dim=(2, 3, 4))
        dice_per_class = (2 * intersection + self.smooth) / (union + self.smooth)
        dice_loss = 1 - dice_per_class.mean()
        return self.weight * dice_loss + (1 - self.weight) * ce_loss


def dice_coefficients(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> List[float]:
    """Compute per‑class Dice coefficients for evaluation.

    Parameters
    ----------
    preds : torch.Tensor
        Raw model logits of shape (N, C, D, H, W).
    targets : torch.Tensor
        Ground truth labels of shape (N, D, H, W) with integer class indices.
    num_classes : int
        Total number of classes.

    Returns
    -------
    List[float]
        Dice coefficient for each class, averaged over the batch.
    """
    with torch.no_grad():
        # Convert logits to predicted class indices
        pred_labels = torch.argmax(preds, dim=1)
        dice_scores: List[float] = []
        for c in range(num_classes):
            pred_c = (pred_labels == c).float()
            target_c = (targets == c).float()
            intersection = (pred_c * target_c).sum(dim=(1, 2, 3))
            union = pred_c.sum(dim=(1, 2, 3)) + target_c.sum(dim=(1, 2, 3))
            dice = (2 * intersection + 1e-5) / (union + 1e-5)
            # Average over the batch; if there are no voxels for class c in both pred and target the dice is defined as 1
            dice_scores.append(dice.mean().item())
        return dice_scores