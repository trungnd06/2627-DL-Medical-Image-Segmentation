"""Paired image/mask transforms for CNN and SegFormer inputs (not SAM)."""

import random

import torch
from PIL import Image

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def pil_to_tensor(image):
    """Copy 8-bit PIL pixels to a float32 CHW tensor in [0, 1]."""
    channels = len(image.getbands())
    pixels = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    return pixels.reshape(image.height, image.width, channels).permute(2, 0, 1).float().div(255)


def binarize_mask(mask, threshold=127):
    if not 0 <= threshold <= 255:
        raise ValueError("Mask threshold must be between 0 and 255")
    return mask.convert("L").point(lambda value: 255 if value > threshold else 0)


def letterbox_geometry(source_size, target_size):
    """Compute centered padding. Sizes are HW; padding is left/top/right/bottom.

    Integer rounding may leave one extra pixel on the right or bottom.
    """
    source_h, source_w = source_size
    target_h, target_w = target_size
    if min(source_h, source_w, target_h, target_w) <= 0:
        raise ValueError("Source and target sizes must be positive")
    scale = min(target_h / source_h, target_w / source_w)
    height = max(1, min(target_h, round(source_h * scale)))
    width = max(1, min(target_w, round(source_w * scale)))
    left, top = (target_w - width) // 2, (target_h - height) // 2
    return {"source_size": (source_h, source_w), "input_size": (target_h, target_w),
            "resized_size": (height, width),
            "padding": (left, top, target_w - width - left, target_h - height - top)}


class SegmentationTransform:
    """Sample geometry once for both image and mask; normalize only the image.

    Geometry is applied before aspect-ratio-preserving resize and centered padding.
    No random augmentation is performed in evaluation mode. apply() also returns
    valid_mask (exclude padding from future loss) and geometry for evaluation.
    """

    def __init__(self, size=(256, 256), training=False, mask_threshold=127,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD,
                 horizontal_flip=0.5, vertical_flip=0.5, rotate90=True,
                 resize_mode="letterbox", image_pad_value=0):
        if isinstance(size, int):
            size = (size, size)
        if len(size) != 2 or any(not isinstance(v, int) or v <= 0 for v in size):
            raise ValueError("size must be positive (height, width)")
        if len(mean) != 3 or len(std) != 3 or any(v <= 0 for v in std):
            raise ValueError("Normalization requires three means and positive standard deviations")
        if any(not 0 <= p <= 1 for p in (horizontal_flip, vertical_flip)):
            raise ValueError("Flip probabilities must be in [0, 1]")
        if resize_mode != "letterbox":
            raise ValueError("Only centered letterbox resizing is supported")
        if not isinstance(image_pad_value, int) or not 0 <= image_pad_value <= 255:
            raise ValueError("Image padding value must be an integer in [0, 255]")
        self.size = tuple(size)
        self.training = training
        self.mask_threshold = mask_threshold
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate90 = rotate90
        self.image_pad_value = image_pad_value

    def __call__(self, image, mask):
        """Convenience image/mask pair; use apply() when metadata is needed."""
        result = self.apply(image, mask)
        return result["image"], result["mask"]

    def apply(self, image, mask):
        if image.size != mask.size:
            raise ValueError("Image and mask sizes differ")
        original_size = (image.height, image.width)
        image = image.convert("RGB")
        mask = binarize_mask(mask, self.mask_threshold)
        operations = []
        if self.training:
            if random.random() < self.horizontal_flip:
                operations.append(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < self.vertical_flip:
                operations.append(Image.Transpose.FLIP_TOP_BOTTOM)
            if self.rotate90:
                rotation = random.randrange(4)
                if rotation:
                    operations.append((Image.Transpose.ROTATE_90, Image.Transpose.ROTATE_180,
                                       Image.Transpose.ROTATE_270)[rotation - 1])
            for operation in operations:
                image, mask = image.transpose(operation), mask.transpose(operation)
        geometry = letterbox_geometry((image.height, image.width), self.size)
        geometry.update(original_size=original_size, augmented=bool(operations))
        height, width = geometry["resized_size"]
        left, top, _, _ = geometry["padding"]
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
        target_h, target_w = self.size
        padded_image = Image.new("RGB", (target_w, target_h), (self.image_pad_value,) * 3)
        padded_mask = Image.new("L", (target_w, target_h), 0)
        padded_image.paste(image, (left, top))
        padded_mask.paste(mask, (left, top))
        valid_mask = torch.zeros((1, target_h, target_w), dtype=torch.bool)
        valid_mask[:, top:top + height, left:left + width] = True
        return {"image": (pil_to_tensor(padded_image) - self.mean) / self.std,
                "mask": pil_to_tensor(padded_mask), "valid_mask": valid_mask,
                "geometry": geometry}

    def denormalize(self, tensor):
        """Return display RGB values; supports CHW or BCHW tensors."""
        return (tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)).clamp(0, 1)
