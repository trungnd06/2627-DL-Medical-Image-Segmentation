"""Điều phối M1: smoke test, overfit vài ảnh và huấn luyện đầy đủ."""

import copy
import csv
import json
import platform
import random
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from PIL import __version__ as pillow_version
from torch.utils.data import DataLoader, Subset

from .datasets.common import file_sha256, find_project_root, save_csv
from .datasets.dataset import PolypDataset, collate_samples, create_dataloaders
from .datasets.duplicates import HASH_PREPROCESSING
from .engine import run_epoch
from .losses import BCEDiceLoss
from .models.unet import UNet
from .utils import load_config, seed_everything
from .visualization import save_architecture, save_learning_curves, save_prediction_preview


def select_device(requested='auto'):
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(requested)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('Supported devices: auto, cpu, cuda or cuda:N')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; install a compatible PyTorch wheel or choose cpu')
    return device


def _review_status(root):
    """Older screening remains usable for debugging; label it provisional."""
    manifest_path = root / 'splits/split_manifest.json'
    current = False
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        current = manifest.get('screening', {}).get('hash_preprocessing') == HASH_PREPROCESSING
    path = root / 'splits/duplicate_review.csv'
    pending = 0
    if path.exists():
        with path.open(encoding='utf-8-sig', newline='') as stream:
            pending = sum(row['scope'] != 'CVC-ClinicDB' and row['decision'] == 'pending'
                          for row in csv.DictReader(stream))
    return {'current_hash_method': current, 'pending_internal_or_cross': pending,
            'ready_for_final_reporting': current and path.exists() and pending == 0}


def _verify_training_sources(root, loaders):
    """Check actual train/val image and mask bytes; never read test predictions."""
    manifest = json.loads((root / 'data/dataset_manifest.json').read_text(encoding='utf-8'))
    count = 0
    for loader in loaders.values():
        dataset = loader.dataset
        for image_id in dataset.ids:
            for kind in ('images', 'masks'):
                relative = f'{kind}/{image_id}{dataset.extension}'
                path = root / 'data' / dataset.dataset_name / relative
                expected = manifest['datasets'][dataset.dataset_name]['files_sha256'][relative]
                if file_sha256(path) != expected:
                    raise ValueError(f'Source checksum mismatch: {path}')
                count += 1
    return count


def _small_loaders(loaders, config, mode, samples):
    n = min(samples, len(loaders['train'].dataset))
    if n < 1:
        raise ValueError('samples must be positive')
    indices = torch.randperm(len(loaders['train'].dataset),
                             generator=torch.Generator().manual_seed(config['seed']))[:n].tolist()
    dataset = loaders['train'].dataset
    # The eval transform also disables augmentation on the overfit training set.
    dataset.transform = loaders['val'].dataset.transform
    train_subset = Subset(dataset, indices)
    if mode == 'overfit':
        eval_dataset = PolypDataset(dataset.root, 'train', dataset.transform, return_original_mask=True)
        eval_subset = Subset(eval_dataset, indices)
    else:
        eval_subset = Subset(loaders['val'].dataset, range(min(samples, len(loaders['val'].dataset))))
    options = dict(batch_size=min(n, config['loader']['batch_size']), num_workers=0,
                   collate_fn=collate_samples, pin_memory=config['loader']['pin_memory'])
    return {'train': DataLoader(train_subset, shuffle=True,
                                generator=torch.Generator().manual_seed(config['seed']), **options),
            'val': DataLoader(eval_subset, shuffle=False, **options)}, indices


def run_training(project_root=None, config_path='configs/unet.yaml', *, mode='overfit',
                 epochs=None, samples=8, device=None, batch_size=None, progress=True):
    """Explicitly launch a fresh run. Overfit scores are diagnostic, never a baseline.

    Each call writes to a unique results/M1/<run_id> directory; splits stay fixed.
    Full mode reads train/val only. Checkpoints contain model + optimizer + scheduler
    + scaler + RNG states; no model/network activity happens at import time.
    """
    if mode not in ('smoke', 'overfit', 'full'):
        raise ValueError('mode must be smoke, overfit or full')
    root = find_project_root(project_root)
    path = Path(config_path)
    config = copy.deepcopy(load_config(path if path.is_absolute() else root / path))
    settings = config['training']
    if epochs is not None:
        settings['max_epochs'] = epochs
    elif mode == 'smoke':
        settings['max_epochs'] = 1
    if settings['max_epochs'] < 1 or settings['early_stopping_patience'] < 1:
        raise ValueError('epochs and early stopping patience must be positive')
    if batch_size is not None:
        config['loader']['batch_size'] = batch_size
    if config['model']['name'] != 'unet' or config['model']['output_channels'] != 1:
        raise ValueError('Step 4 supports binary UNet M1 only')
    if settings['loss'] != 'bce_dice' or not settings['ignore_padding']:
        raise ValueError('M1 requires bce_dice with ignore_padding=true')
    if settings['optimizer'] != 'Adam' or settings['checkpoint_metric'] != 'val_dice':
        raise ValueError('M1 requires Adam and checkpoint_metric=val_dice')
    if not config['evaluation']['original_resolution']:
        raise ValueError('M1 validation must use original-resolution masks')
    scheduler_config = settings['scheduler']
    if (scheduler_config['name'], scheduler_config['monitor'], scheduler_config['mode']) != ('ReduceLROnPlateau', 'val_dice', 'max'):
        raise ValueError('Expected ReduceLROnPlateau on val_dice in max mode')
    device = select_device(device or settings.get('device', 'auto'))
    amp = bool(settings.get('amp', True) and device.type == 'cuda')
    config['training'].update(device=str(device), amp=amp)
    config['run'] = {'mode': mode, 'samples': samples if mode != 'full' else None,
                     'augmentation_enabled': mode == 'full', 'scheduler_enabled': mode != 'overfit',
                     'early_stopping_enabled': mode != 'overfit'}
    seed_everything(config['seed'])
    loaders = create_dataloaders(root, config)
    source_files_verified = _verify_training_sources(root, loaders)
    indices = None
    if mode != 'full':
        loaders, indices = _small_loaders(loaders, config, mode, samples)
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model = UNet(channels=config['model']['channels'], output_channels=1).to(device)
    criterion = BCEDiceLoss(settings['bce_weight'], settings['dice_weight'])
    optimizer = torch.optim.Adam(model.parameters(), lr=settings['learning_rate'], weight_decay=settings['weight_decay'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=scheduler_config['factor'],
        patience=scheduler_config['patience'], min_lr=scheduler_config['min_lr'])
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + f'_{mode}_seed{config["seed"]}_{uuid.uuid4().hex[:6]}'
    output = root / 'results/M1' / run_id
    output.mkdir(parents=True, exist_ok=False)
    (output / 'config.yaml').write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=root, text=True,
                                             stderr=subprocess.DEVNULL).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    review = _review_status(root)
    metadata = {'mode': mode, 'run_id': run_id, 'seed': config['seed'], 'commit': commit, 'tracked_changes': dirty,
                'review_status': review, 'source_files_verified': source_files_verified,
                'device': str(device), 'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor(),
                'precision': 'cuda_fp16_amp' if amp else 'float32', 'python': platform.python_version(),
                'torch': str(torch.__version__), 'cuda': torch.version.cuda, 'pillow': pillow_version,
                'parameters': sum(p.numel() for p in model.parameters()), 'pretrained_weights': None,
                'determinism': 'seeded Python/torch, cuDNN deterministic; cross-platform bitwise equality not guaranteed',
                'counts': {name: len(loader.dataset) for name, loader in loaders.items()},
                'selected_train_indices': indices, 'validation_role': 'same_training_subset' if mode == 'overfit' else 'held_out_validation',
                'train_metric_resolution': 'padded_input_valid_pixels', 'val_metric_resolution': 'original_image',
                'split_sha256': {p.name: file_sha256(p) for p in sorted((root / 'splits').glob('*.txt'))},
                'review_sha256': file_sha256(root / 'splits/duplicate_review.csv') if (root / 'splits/duplicate_review.csv').exists() else None,
                'source_manifest_sha256': file_sha256(root / 'data/dataset_manifest.json'),
                'source_code_sha256': {p.relative_to(root).as_posix(): file_sha256(p) for parent in ('src', 'scripts', 'configs')
                                       for p in sorted((root / parent).rglob('*')) if p.suffix in ('.py', '.yaml')},
                'status': 'running'}
    def write_metadata():
        (output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    write_metadata()
    print(f'Run: {output}\nMode={mode}, device={device}, AMP={amp}, counts={metadata["counts"]}', flush=True)
    if not review['ready_for_final_reporting']:
        print('NOTE: duplicate screening/review is not current. This run is provisional; finish notebook 02 before final reporting.', flush=True)
    history = []
    best_dice, best_epoch, stale = -1.0, 0, 0
    started = time.perf_counter()
    try:
        save_architecture(config['model']['channels'], output / 'architecture.png')
        save_prediction_preview(model, loaders['val'].dataset, device, output / 'predictions_before.png',
                                config['evaluation']['threshold'])
        for epoch in range(1, settings['max_epochs'] + 1):
            epoch_start = time.perf_counter()
            lr = optimizer.param_groups[0]['lr']
            train, _ = run_epoch(model, loaders['train'], criterion, device, optimizer, scaler, amp,
                                 config['evaluation']['threshold'], progress=progress)
            val, rows = run_epoch(model, loaders['val'], criterion, device, amp=amp,
                                  threshold=config['evaluation']['threshold'], progress=progress)
            improved = val['dice'] > best_dice
            if improved:
                best_dice, best_epoch, stale = val['dice'], epoch, 0
            else:
                stale += 1
            # For overfit, use fixed LR and all requested epochs to diagnose learning.
            if mode != 'overfit':
                scheduler.step(val['dice'])
            row = {'epoch': epoch, 'learning_rate': lr, 'seconds': time.perf_counter() - epoch_start,
                   **{f'train_{key}': value for key, value in train.items()},
                   **{f'val_{key}': value for key, value in val.items()}}
            history.append(row)
            save_csv(output / 'history.csv', list(row), history)
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                          'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                          'scaler_state_dict': scaler.state_dict(), 'best_dice': best_dice, 'best_epoch': best_epoch,
                          'config': config, 'mode': mode, 'metadata': metadata,
                          'rng_state': {'python': random.getstate(), 'torch': torch.get_rng_state(),
                                        'cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
                                        'train_loader': loaders['train'].generator.get_state()}}
            # Replace only within this new run; a partial write cannot corrupt the previous checkpoint.
            temporary = output / 'checkpoint.tmp'
            torch.save(checkpoint, temporary)
            temporary.replace(output / 'last.pt')
            if improved:
                torch.save(checkpoint, temporary)
                temporary.replace(output / 'best.pt')
                save_csv(output / 'best_val_metrics.csv', list(rows[0]), rows)
            print(f"Epoch {epoch}/{settings['max_epochs']} | train_loss={train['loss']:.4f} | val_loss={val['loss']:.4f} | val_dice={val['dice']:.4f} | val_iou={val['iou']:.4f} | lr={lr:.2g}", flush=True)
            if mode != 'overfit' and stale >= settings['early_stopping_patience']:
                print(f'Early stopping; best epoch={best_epoch}', flush=True)
                break
        checkpoint = torch.load(output / 'best.pt', map_location=device, weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        del checkpoint
        save_prediction_preview(model, loaders['val'].dataset, device, output / 'predictions_best.png',
                                config['evaluation']['threshold'])
        save_learning_curves(history, output / 'learning_curves.png', mode=mode)
        metadata.update(status='complete', epochs_completed=len(history), best_epoch=best_epoch, best_dice=best_dice,
                        elapsed_seconds=time.perf_counter() - started,
                        overfit_passed=(best_dice >= 0.95) if mode == 'overfit' else None,
                        peak_cuda_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None)
        if mode == 'overfit' and not metadata['overfit_passed']:
            print('Overfit Dice < 0.95: inspect curves/overlays before launching full training.', flush=True)
    except BaseException as error:
        metadata.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error),
                        elapsed_seconds=time.perf_counter() - started)
        if history:
            save_learning_curves(history, output / 'learning_curves.png', mode=mode)
        raise
    finally:
        write_metadata()
    print(f'Saved: {output}', flush=True)
    return {'run_dir': output, 'history': history, 'metadata': metadata}
