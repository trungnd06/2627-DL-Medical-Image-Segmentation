"""Binary foreground metrics, returned per image (not pooled across a batch)."""

import torch
import torch.nn.functional as F


def _as_bchw(tensor):
    if tensor.ndim == 2:
        tensor = tensor[None, None]
    elif tensor.ndim == 3:
        tensor = tensor[:, None]
    if tensor.ndim != 4 or tensor.shape[1] != 1:
        raise ValueError("Expected HW, BHW or B1HW binary tensors")
    if not tensor.numel():
        raise ValueError("Empty tensor")
    return tensor


@torch.no_grad()
def binary_metrics(prediction, target, threshold=0.5, from_logits=True):
    """Dice/IoU=1 if both empty, 0 if only one empty; undefined P/R=0.

    Use from_logits=False for probabilities or binary predictions. Each returned
    tensor has shape [B]; take its mean to compute the image-wise dataset metric.
    """
    prediction, target = _as_bchw(prediction), _as_bchw(target)
    if prediction.shape != target.shape:
        raise ValueError("Prediction and target shapes differ")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0, 1]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("Nonfinite prediction or target")
    if not ((target == 0) | (target == 1)).all():
        raise ValueError("Target must contain binary 0/1 labels")
    if from_logits:
        probability = prediction.float().sigmoid()
    else:
        probability = prediction.float()
        if ((probability < 0) | (probability > 1)).any():
            raise ValueError("Probabilities must be in [0, 1]")
    predicted = (probability >= threshold).flatten(1)
    truth = target.to(predicted.device).bool().flatten(1)
    intersection = (predicted & truth).sum(1).float()
    pred_count, true_count = predicted.sum(1).float(), truth.sum(1).float()
    total = pred_count + true_count
    union = total - intersection
    return {
        "dice": torch.where(total == 0, torch.ones_like(total), 2 * intersection / total.clamp_min(1)),
        "iou": torch.where(union == 0, torch.ones_like(union), intersection / union.clamp_min(1)),
        "precision": intersection / pred_count.clamp_min(1),
        "recall": intersection / true_count.clamp_min(1),
    }


@torch.no_grad()
def metrics_at_original_size(logits, original_masks, threshold=0.5):
    """Resize logits BEFORE sigmoid/threshold; compare with untouched source masks.

    Bilinear interpolation uses align_corners=False as defined in pipeline.md.
    Variable image dimensions are handled separately for every sample.
    """
    logits = _as_bchw(logits)
    if len(original_masks) != len(logits):
        raise ValueError("One original mask is required per prediction")
    values = {name: [] for name in ("dice", "iou", "precision", "recall")}
    for index, mask in enumerate(original_masks):
        mask = _as_bchw(mask)
        if mask.shape[0] != 1:
            raise ValueError("Each original mask must represent one image")
        restored = F.interpolate(logits[index:index + 1].float(), size=mask.shape[-2:],
                                 mode="bilinear", align_corners=False)
        scores = binary_metrics(restored, mask, threshold=threshold, from_logits=True)
        for name, score in scores.items():
            values[name].append(score)
    return {name: torch.cat(scores) for name, scores in values.items()}
