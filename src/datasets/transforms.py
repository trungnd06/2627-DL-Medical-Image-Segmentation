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


class SegmentationTransform:
    """Sample geometry once for both image and mask; normalize only the image.

    Geometry is applied before resize so 90-degree rotations also work with
    rectangular source images without cropping or changing the output shape.
    No random augmentation is performed in evaluation mode.
    """

    def __init__(self, size=(256, 256), training=False, mask_threshold=127,
                 mean=IMAGENET_MEAN, std=IMAGENET_STD,
                 horizontal_flip=0.5, vertical_flip=0.5, rotate90=True):
        if isinstance(size, int):
            size = (size, size)
        if len(size) != 2 or any(not isinstance(v, int) or v <= 0 for v in size):
            raise ValueError("size must be positive (height, width)")
        if len(mean) != 3 or len(std) != 3 or any(v <= 0 for v in std):
            raise ValueError("Normalization requires three means and positive standard deviations")
        if any(not 0 <= p <= 1 for p in (horizontal_flip, vertical_flip)):
            raise ValueError("Flip probabilities must be in [0, 1]")
        self.size = tuple(size)
        self.training = training
        self.mask_threshold = mask_threshold
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotate90 = rotate90

    def __call__(self, image, mask):
        if image.size != mask.size:
            raise ValueError("Image and mask sizes differ")
        image = image.convert("RGB")
        mask = binarize_mask(mask, self.mask_threshold)
        if self.training:
            operations = []
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
        height, width = self.size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)
        return (pil_to_tensor(image) - self.mean) / self.std, pil_to_tensor(mask)

    def denormalize(self, tensor):
        """Return display RGB values; supports CHW or BCHW tensors."""
        return (tensor * self.std.to(tensor.device) + self.mean.to(tensor.device)).clamp(0, 1)
