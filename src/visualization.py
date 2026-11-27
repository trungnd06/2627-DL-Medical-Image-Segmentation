"""Hình kiến trúc, learning curve và dự đoán; chỉ ghi file khi được gọi."""

from pathlib import Path
import csv
import json
import math

import torch
from PIL import Image, ImageDraw

from .datasets.preview import overlay
from .metrics import restore_original_logits


def save_architecture(channels, path):
    canvas = Image.new('RGB', (1080, 390), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 12), 'M1 U-Net | DoubleConv = 2 x (Conv3x3 + ReLU) | no pretrained weights', fill='black')
    for index, width in enumerate(channels[:-1]):
        x, y = 20 + index * 210, 60 + index * 40
        draw.rectangle((x, y, x + 165, y + 35), fill='#dceaff', outline='navy')
        draw.text((x + 8, y + 10), f'Encoder {width} | /{2 ** index}', fill='black')
        draw.rectangle((x, y + 160, x + 165, y + 195), fill='#dcffdf', outline='green')
        draw.text((x + 8, y + 170), f'Decoder {width} | /{2 ** index}', fill='black')
        draw.line((x + 80, y + 35, x + 80, y + 160), fill='gray', width=2)
        draw.text((x + 85, y + 100), 'skip', fill='gray')
        if index < 3:
            draw.line((x + 165, y + 17, x + 210, y + 57), fill='navy', width=2)
            draw.line((x + 210, y + 217, x + 165, y + 177), fill='green', width=2)
    draw.rectangle((860, 190, 1060, 245), fill='#ffe5c8', outline='brown')
    draw.text((870, 207), f'Bottleneck {channels[-1]} | /16', fill='black')
    draw.line((815, 197, 860, 207), fill='navy', width=2)
    draw.line((860, 230, 815, 357), fill='green', width=2)
    draw.text((20, 355), 'RGB -> Encoder -> MaxPool x4 -> Bottleneck -> ConvTranspose2d + skip x4 -> Conv1x1 -> logits', fill='black')
    canvas.save(path)
    return Path(path)


def save_learning_curves(history, path, *, mode=None):
    """Save loss/Dice/IoU/LR panels without a GUI or changing training output."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    if not history:
        raise ValueError('History is empty; no completed epoch to plot')
    keys = ('epoch', 'train_loss', 'val_loss', 'train_dice', 'val_dice',
            'train_iou', 'val_iou', 'learning_rate')
    values = {key: [float(row[key]) for row in history] for key in keys}
    if not all(math.isfinite(v) for series in values.values() for v in series):
        raise ValueError('History must contain finite values')
    epochs = values['epoch']
    if any(e < 1 or not e.is_integer() for e in epochs) or any(a >= b for a, b in zip(epochs, epochs[1:])):
        raise ValueError('Epoch numbers must be increasing positive integers')
    best_index = max(range(len(history)), key=lambda i: values['val_dice'][i])
    best_epoch = epochs[best_index]
    figure = Figure(figsize=(12, 8), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    title = 'Segmentation learning curves'
    if mode == 'overfit':
        title += ' — overfit diagnostic (same training subset)'
    elif mode:
        title += f' — {mode}'
    figure.suptitle(title, fontsize=14)
    for axis, metric, label in zip(axes.flat, ('loss', 'dice', 'iou'),
                                   ('Loss (BCE + soft Dice)', 'Dice', 'IoU')):
        for split, color in (('train', '#1768ac'), ('val', '#e67700')):
            axis.plot(epochs, values[f'{split}_{metric}'], color=color, label=split, linewidth=1.5)
        axis.set_title(label)
        axis.set_ylabel(label.split(' (')[0])
        axis.axvline(best_epoch, color='gray', linestyle='--', linewidth=1,
                     label=f'Best Dice: epoch {int(best_epoch)}')
        axis.set_ylim(0, 1.02 if metric != 'loss' else max(1e-6, max(values['train_loss'] + values['val_loss']) * 1.08))
        axis.legend(fontsize=8)
    axis = axes[1, 1]
    axis.step(epochs, values['learning_rate'], where='post', color='#397d43', label='Learning rate')
    axis.set_title('Learning rate')
    axis.set_ylabel('LR')
    axis.set_ylim(bottom=0)
    axis.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    axis.legend(fontsize=8)
    for axis in axes.flat:
        axis.set_xlabel('Epoch')
        axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        axis.set_xlim(epochs[0] - 0.5, epochs[-1] + 0.5)
        axis.grid(alpha=0.25)
    figure.supxlabel('Train metrics: valid pixels at input size | Val metrics: original image size', fontsize=9)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=160)
    return path


def plot_run_history(run_dir):
    """Regenerate an image from an existing CSV, with no model/checkpoint loading.

    Works after a kernel restart, including a partially completed run. If training
    is currently writing the CSV, wait for the epoch log and retry on read errors.
    """
    run_dir = Path(run_dir)
    with (run_dir / 'history.csv').open(encoding='utf-8-sig', newline='') as source:
        history = list(csv.DictReader(source))
    metadata_path = run_dir / 'metadata.json'
    mode = json.loads(metadata_path.read_text(encoding='utf-8')).get('mode') if metadata_path.exists() else None
    return save_learning_curves(history, run_dir / 'learning_curves.png', mode=mode)


def plot_ablation_curves(a1_dir, m2_dir):
    """Overlay recorded validation curves; each line ends at its actual last epoch."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator
    from .comparison import load_run_summary
    figure = Figure(figsize=(12, 8), layout='constrained')
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    figure.suptitle('A1 vs M2 — validation and learning-rate schedules')
    metrics = (('val_loss', 'Validation loss'), ('val_dice', 'Validation Dice'),
               ('val_iou', 'Validation IoU'), ('learning_rate', 'Learning rate'))
    for directory, label, color in ((a1_dir, 'A1: random', '#1768ac'), (m2_dir, 'M2: ImageNet', '#e67700')):
        summary = load_run_summary(directory)
        epochs = [int(row['epoch']) for row in summary['history']]
        for axis, (key, title) in zip(axes.flat, metrics):
            values = [float(row[key]) for row in summary['history']]
            axis.plot(epochs, values, color=color, label=label,
                      drawstyle='steps-post' if key == 'learning_rate' else 'default')
            if key != 'learning_rate':
                epoch = summary['metadata']['best_epoch']
                axis.scatter([epoch], [float(summary['best'][key])], color=color, marker='*', s=100, zorder=4)
            axis.set_title(title)
            axis.set_xlabel('Epoch')
            axis.set_ylim(bottom=0)
            if key in ('val_dice', 'val_iou'):
                axis.set_ylim(0, 1.02)
            if key == 'learning_rate':
                axis.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
            axis.xaxis.set_major_locator(MaxNLocator(integer=True))
            axis.grid(alpha=.25)
    for axis in axes.flat:
        axis.legend()
    figure.supxlabel('Stars: each model checkpoint selected by validation Dice. Curves stop at recorded epochs.', fontsize=9)
    path = Path(a1_dir) / 'A1_M2_learning_curves.png'
    figure.savefig(path, dpi=160)
    return path


@torch.no_grad()
def save_prediction_preview(model, dataset, device, path, threshold=0.5, count=4):
    """RGB, GT và dự đoán ở kích thước gốc. Dataset phải không augmentation."""
    model.eval()
    source_dataset = dataset
    while hasattr(source_dataset, 'dataset'):
        source_dataset = source_dataset.dataset
    count = min(count, len(dataset))
    canvas = Image.new('RGB', (960, 35 + count * 240), 'white')
    draw = ImageDraw.Draw(canvas)
    for col, title in enumerate(('Original RGB', 'Ground truth overlay', 'Prediction overlay')):
        draw.text((col * 320 + 8, 8), title, fill='black')
    for index in range(count):
        sample = dataset[index]
        logits = model(sample['image'][None].to(device))
        native = restore_original_logits(logits, [sample['geometry']])[0]
        prediction = native.sigmoid().ge(threshold).cpu().to(torch.uint8).mul(255)[0, 0]
        predicted_mask = Image.frombytes('L', (prediction.shape[1], prediction.shape[0]),
                                         bytes(prediction.flatten().tolist()))
        mask = sample['original_mask'][0].to(torch.uint8).mul(255)
        gt = Image.frombytes('L', (mask.shape[1], mask.shape[0]), bytes(mask.flatten().tolist()))
        name, filename = sample['image_id'].split('/', 1)
        with Image.open(source_dataset.root / 'data' / name / 'images' / filename) as source:
            image = source.convert('RGB')
        draw.text((6, 35 + index * 240), filename, fill='black')
        for col, panel in enumerate((image, overlay(image, gt), overlay(image, predicted_mask))):
            panel = panel.copy()
            panel.thumbnail((308, 210), Image.Resampling.BILINEAR)
            canvas.paste(panel, (col * 320 + (320 - panel.width) // 2, 55 + index * 240))
    canvas.save(path)
    return Path(path)
