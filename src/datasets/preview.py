"""Visualize the real paired transforms without a plotting dependency."""

from pathlib import Path

import torch
from PIL import Image, ImageDraw

from .transforms import binarize_mask


def tensor_image(tensor):
    pixels = tensor.detach().cpu().clamp(0, 1).mul(255).round().to(torch.uint8)
    return Image.frombytes("RGB", (pixels.shape[2], pixels.shape[1]),
                           bytes(pixels.permute(1, 2, 0).contiguous().flatten().tolist()))


def overlay(image, mask):
    return Image.composite(Image.blend(image, Image.new("RGB", image.size, "red"), 0.4), image, mask)


def save_transform_preview(train_dataset, output_path, count=3):
    """Show the same samples raw, labeled, resized, and randomly augmented."""
    from .transforms import SegmentationTransform
    transform = train_dataset.transform
    evaluation = SegmentationTransform(size=transform.size, mask_threshold=transform.mask_threshold,
                                       mean=transform.mean.flatten().tolist(), std=transform.std.flatten().tolist())
    count = min(count, len(train_dataset))
    if count < 1:
        raise ValueError("At least one preview sample is required")
    panel_width, panel_height, row_height = 280, 235, 265
    canvas = Image.new("RGB", (4 * panel_width, 30 + count * row_height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, title in enumerate(("Original RGB", "Original GT overlay", "Resize (eval) + GT", "Augment (train) + GT")):
        draw.text((column * panel_width + 6, 7), title, fill="black")
    for index in range(count):
        filename = train_dataset.ids[index] + train_dataset.extension
        folder = train_dataset.root / "data" / train_dataset.dataset_name
        with Image.open(folder / "images" / filename) as source:
            original = source.convert("RGB")
        with Image.open(folder / "masks" / filename) as source:
            mask = source.convert("L")
        eval_image, eval_mask = evaluation(original, mask)
        sample = train_dataset[index]
        train_image, train_mask = sample["image"], sample["mask"]
        def transformed_panel(image, mask):
            rgb = tensor_image(transform.denormalize(image))
            binary = Image.frombytes("L", (mask.shape[2], mask.shape[1]),
                                     bytes(mask.mul(255).byte().flatten().tolist()))
            return overlay(rgb, binary)
        panels = (original, overlay(original, binarize_mask(mask, transform.mask_threshold)),
                  transformed_panel(eval_image, eval_mask), transformed_panel(train_image, train_mask))
        y = 30 + index * row_height
        draw.text((6, y + 2), filename, fill="black")
        for column, panel in enumerate(panels):
            panel = panel.copy()
            panel.thumbnail((panel_width - 12, panel_height), Image.Resampling.BILINEAR)
            canvas.paste(panel, (column * panel_width + (panel_width - panel.width) // 2, y + 22))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path
