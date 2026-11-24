"""Load fixed polyp splits and build batches without altering original files."""

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from .common import find_project_root
from .transforms import SegmentationTransform, binarize_mask, pil_to_tensor
from ..utils import seed_everything, seed_worker

SPLITS = {
    "train": ("Kvasir-SEG", ".jpg", "kvasir_train.txt"),
    "val": ("Kvasir-SEG", ".jpg", "kvasir_val.txt"),
    "test": ("Kvasir-SEG", ".jpg", "kvasir_test.txt"),
    "external": ("CVC-ClinicDB", ".png", "clinicdb_external.txt"),
}


class PolypDataset(Dataset):
    """Read IDs only from a committed split; optionally retain original masks.

    original_size is (height, width). original_mask stays at source resolution
    for evaluation, while mask is resized to model-input resolution.
    """

    def __init__(self, project_root, split="train", transform=None,
                 return_original_mask=False):
        self.root = find_project_root(project_root)
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.split = split
        dataset, extension, filename = SPLITS[split]
        self.dataset_name = dataset
        self.extension = extension
        self.transform = transform or SegmentationTransform(training=(split == "train"))
        if split != "train" and self.transform.training:
            raise ValueError("Evaluation splits must not use training augmentation")
        if return_original_mask and self.transform.training:
            raise ValueError("Original masks are for unaugmented evaluation samples")
        self.return_original_mask = return_original_mask
        path = self.root / "splits" / filename
        entries = [line.partition("#")[0].strip()
                   for line in path.read_text(encoding="utf-8-sig").splitlines()]
        self.ids = []
        for entry in filter(None, entries):
            if Path(entry).name != entry or "/" in entry or "\\" in entry or ":" in entry:
                raise ValueError(f"Split entries must be filenames or IDs: {entry}")
            # The committed lists use stems; allow the correct source suffix too.
            if "." in entry and Path(entry).suffix.lower() != extension:
                raise ValueError(f"Unexpected image extension: {entry}")
            self.ids.append(Path(entry).stem)
        if not self.ids or len(self.ids) != len(set(self.ids)):
            raise ValueError(f"{path}: empty list or duplicate IDs")
        for image_id in self.ids:
            for kind in ("images", "masks"):
                file = self.root / "data" / dataset / kind / (image_id + extension)
                if not file.is_file():
                    raise FileNotFoundError(file)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        folder = self.root / "data" / self.dataset_name
        filename = image_id + self.extension
        with Image.open(folder / "images" / filename) as source:
            image = source.convert("RGB")
        with Image.open(folder / "masks" / filename) as source:
            mask = source.convert("L")
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch: {self.dataset_name}/{filename}")
        original_size = (image.height, image.width)
        original_mask = None
        if self.return_original_mask:
            original_mask = pil_to_tensor(binarize_mask(mask, self.transform.mask_threshold))
        image_tensor, mask_tensor = self.transform(image, mask)
        sample = {"image": image_tensor, "mask": mask_tensor,
                  "image_id": f"{self.dataset_name}/{filename}", "original_size": original_size}
        if original_mask is not None:
            sample["original_mask"] = original_mask
        return sample


def collate_samples(samples):
    """Stack resized tensors; keep variable-resolution original masks as a list."""
    if not samples:
        raise ValueError("Cannot collate an empty batch")
    batch = {"image": torch.stack([s["image"] for s in samples]),
             "mask": torch.stack([s["mask"] for s in samples]),
             "image_id": [s["image_id"] for s in samples],
             "original_size": [s["original_size"] for s in samples]}
    has_original = ["original_mask" in s for s in samples]
    if any(has_original) and not all(has_original):
        raise ValueError("Samples disagree about original-mask availability")
    if all(has_original):
        batch["original_mask"] = [s["original_mask"] for s in samples]
    return batch


def create_dataloaders(project_root, config, include_test=False):
    """Build train/val by default. Test/external are explicit final-evaluation opt-ins."""
    seed = config["seed"]
    seed_everything(seed)
    data = config["data"]
    settings = config["loader"]
    augmentation = data["augmentation"]
    transforms = {
        name: SegmentationTransform(
            size=tuple(data["image_size"]), training=name == "train",
            mask_threshold=data["mask_threshold"], mean=data["mean"], std=data["std"],
            horizontal_flip=augmentation["horizontal_flip"],
            vertical_flip=augmentation["vertical_flip"], rotate90=augmentation["rotate90"])
        for name in ("train", "val", "test", "external")
    }
    names = ("train", "val", "test", "external") if include_test else ("train", "val")
    datasets = {name: PolypDataset(project_root, name, transforms[name], return_original_mask=name != "train")
                for name in names}
    # Dataset-qualified IDs avoid confusing ClinicDB numbers with Kvasir IDs.
    seen = set()
    for name, dataset in datasets.items():
        ids = {f"{dataset.dataset_name}/{key}" for key in dataset.ids}
        if ids & seen:
            raise ValueError(f"Split overlap detected at {name}")
        seen.update(ids)
    workers = settings["num_workers"]
    if workers < 0 or settings["batch_size"] <= 0:
        raise ValueError("Invalid batch_size or num_workers")
    return {name: DataLoader(
        dataset, batch_size=settings["batch_size"], shuffle=name == "train",
        num_workers=workers, pin_memory=settings["pin_memory"], drop_last=False,
        collate_fn=collate_samples, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed), persistent_workers=workers > 0)
        for name, dataset in datasets.items()}
