"""Train/validation dùng chung; không tải dữ liệu hoặc chạy train khi import."""

from contextlib import nullcontext

import torch
from tqdm.auto import tqdm

from .metrics import binary_metrics, metrics_at_original_size


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              amp=False, threshold=0.5, original_resolution=True, progress=True):
    """Có optimizer => train; nếu không, eval/no_grad. Trả mean và metric từng ảnh.

    BCE epoch được gộp theo số pixel hợp lệ; soft Dice và metric theo số ảnh.
    Validation dùng mask gốc, bỏ padding trước khi resize logits về ảnh gốc.
    """
    training = optimizer is not None
    model.train(training)
    count = 0
    pixels = 0
    bce_sum = dice_loss_sum = 0.0
    totals = dict.fromkeys(('dice', 'iou', 'precision', 'recall'), 0.0)
    rows = []
    iterator = tqdm(loader, desc='train' if training else 'val', leave=False, disable=not progress)
    with (nullcontext() if training else torch.no_grad()):
        for batch in iterator:
            image = batch['image'].to(device, non_blocking=True)
            target = batch['mask'].to(device, non_blocking=True)
            valid = batch['valid_mask'].to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                logits = model(image)
                parts = criterion.components(logits, target, valid)
            if not torch.isfinite(parts['loss']):
                raise FloatingPointError('Nonfinite loss; stop instead of saving an invalid checkpoint')
            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(parts['loss']).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    parts['loss'].backward()
                    optimizer.step()
            if not training and original_resolution:
                scores = metrics_at_original_size(logits.detach(), batch['original_mask'],
                                                  batch['geometry'], threshold)
            else:
                scores = binary_metrics(logits.detach(), target, threshold=threshold, valid_mask=valid)
            scores = {key: value.cpu().tolist() for key, value in scores.items()}
            n, valid_count = len(image), int(valid.sum().item())
            count += n
            pixels += valid_count
            bce_sum += parts['bce'].item() * valid_count
            dice_loss_sum += parts['soft_dice_loss'].item() * n
            for key, values in scores.items():
                totals[key] += sum(values)
            if not training:
                rows.extend({'image_id': image_id, **{key: values[i] for key, values in scores.items()}}
                            for i, image_id in enumerate(batch['image_id']))
            if progress:
                iterator.set_postfix(loss=f"{parts['loss'].item():.4f}")
    if count == 0:
        raise ValueError('Cannot run an empty DataLoader')
    bce, dice_loss = bce_sum / pixels, dice_loss_sum / count
    summary = {'loss': criterion.bce_weight * bce + criterion.dice_weight * dice_loss,
               'bce': bce, 'soft_dice_loss': dice_loss,
               **{key: value / count for key, value in totals.items()}, 'samples': count}
    return summary, rows
