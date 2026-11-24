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
def binary_metrics(prediction, target, threshold=0.5, from_logits=True, valid_mask=None):
    """Dice/IoU=1 if both empty, 0 if only one empty; undefined P/R=0.

    Use from_logits=False for probabilities or binary predictions. Each returned
    tensor has shape [B]; take its mean to compute the image-wise dataset metric.
    At padded input resolution, pass valid_mask to ignore artificial border pixels.
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
    if valid_mask is not None:
        valid_mask = _as_bchw(valid_mask).to(predicted.device)
        if valid_mask.shape != prediction.shape:
            raise ValueError("valid_mask shape must match prediction")
        if not ((valid_mask == 0) | (valid_mask == 1)).all():
            raise ValueError("valid_mask must contain binary 0/1 values")
        valid = valid_mask.bool().flatten(1)
        if not valid.any(1).all():
            raise ValueError("Each image needs at least one valid pixel")
        predicted, truth = predicted & valid, truth & valid
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


def restore_original_logits(logits, geometries):
    """Remove centered padding, then resize each logit map to original HW.

    Metadata is required: a padded input must never silently use stretch resize.
    Decoder logits at lower resolution are first upsampled to padded input size.
    Training augmentation cannot be inverted with letterbox metadata alone.
    """
    logits = _as_bchw(logits)
    if len(geometries) != len(logits):
        raise ValueError("One geometry record is required per prediction")
    restored = []
    for index, geometry in enumerate(geometries):
        if geometry["augmented"]:
            raise ValueError("Cannot restore randomly augmented samples to original orientation")
        original_h, original_w = geometry["original_size"]
        input_h, input_w = geometry["input_size"]
        height, width = geometry["resized_size"]
        left, top, right, bottom = geometry["padding"]
        dimensions = (original_h, original_w, input_h, input_w, height, width)
        if any(not isinstance(v, int) or v <= 0 for v in dimensions):
            raise ValueError("Geometry dimensions must be positive integers")
        if any(not isinstance(v, int) or v < 0 for v in (left, top, right, bottom)):
            raise ValueError("Padding must contain nonnegative integers")
        if left + width + right != input_w or top + height + bottom != input_h:
            raise ValueError("Padding and resized dimensions do not match padded input")
        if tuple(geometry["source_size"]) != (original_h, original_w):
            raise ValueError("Evaluation source dimensions differ from original image")
        padded_logits = logits[index:index + 1].float()
        if padded_logits.shape[-2:] != (input_h, input_w):
            padded_logits = F.interpolate(padded_logits, size=(input_h, input_w),
                                          mode="bilinear", align_corners=False)
        content = padded_logits[:, :, top:top + height, left:left + width]
        restored.append(F.interpolate(content, size=(original_h, original_w),
                                      mode="bilinear", align_corners=False))
    return restored


@torch.no_grad()
def metrics_at_original_size(logits, original_masks, geometries, threshold=0.5):
    """Unpad -> resize logits -> sigmoid/threshold -> compare with original masks."""
    restored_logits = restore_original_logits(logits, geometries)
    if len(original_masks) != len(restored_logits):
        raise ValueError("One original mask is required per prediction")
    values = {name: [] for name in ("dice", "iou", "precision", "recall")}
    for restored, mask in zip(restored_logits, original_masks):
        mask = _as_bchw(mask)
        if mask.shape[0] != 1:
            raise ValueError("Each original mask must represent one image")
        scores = binary_metrics(restored, mask, threshold=threshold, from_logits=True)
        for name, score in scores.items():
            values[name].append(score)
    return {name: torch.cat(scores) for name, scores in values.items()}
