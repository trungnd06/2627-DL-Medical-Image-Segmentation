"""Loss M1: BCEWithLogits + soft Dice, bỏ pixel padding bằng valid_mask."""

import math

import torch
from torch import nn
from torch.nn import functional as F


class BCEDiceLoss(nn.Module):
    """BCE trung bình pixel hợp lệ; Dice trung bình từng ảnh, trọng số mặc định 1:1."""

    def __init__(self, bce_weight=1.0, dice_weight=1.0, smooth=1e-6):
        super().__init__()
        if not all(math.isfinite(v) and v >= 0 for v in (bce_weight, dice_weight)):
            raise ValueError("Loss weights must be finite and nonnegative")
        if bce_weight + dice_weight == 0 or not math.isfinite(smooth) or smooth <= 0:
            raise ValueError("At least one loss weight and smooth must be positive")
        self.bce_weight, self.dice_weight, self.smooth = bce_weight, dice_weight, smooth

    def components(self, logits, target, valid_mask):
        if logits.ndim != 4 or logits.shape[1] != 1 or logits.shape != target.shape:
            raise ValueError("Expected matching B1HW logits and targets")
        if valid_mask is None or valid_mask.shape != logits.shape:
            raise ValueError("A matching valid_mask is required to exclude padding")
        logits, target = logits.float(), target.to(device=logits.device, dtype=torch.float32)
        valid_mask = valid_mask.to(logits.device)
        if not torch.isfinite(logits).all() or not ((target == 0) | (target == 1)).all():
            raise ValueError("Nonfinite logits or nonbinary target")
        if not ((valid_mask == 0) | (valid_mask == 1)).all():
            raise ValueError("valid_mask must contain 0/1")
        valid = valid_mask.float()
        if not valid.flatten(1).any(1).all():
            raise ValueError("Every image needs valid pixels")
        # Compute in float32 even with mixed precision, for stable reductions.
        bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none') * valid).sum() / valid.sum()
        probability = logits.sigmoid() * valid
        truth = target * valid
        dims = (1, 2, 3)
        dice = 1 - ((2 * (probability * truth).sum(dims) + self.smooth)
                    / (probability.sum(dims) + truth.sum(dims) + self.smooth)).mean()
        return {'loss': self.bce_weight * bce + self.dice_weight * dice, 'bce': bce, 'soft_dice_loss': dice}

    def forward(self, logits, target, valid_mask):
        return self.components(logits, target, valid_mask)['loss']
