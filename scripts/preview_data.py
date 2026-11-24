"""Check step-3 batches/metrics and save a real image/mask transform preview."""

import argparse
import json
import platform
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from src.datasets.common import find_project_root
from src.datasets.dataset import create_dataloaders
from src.datasets.preview import save_transform_preview
from src.metrics import binary_metrics, metrics_at_original_size
from src.utils import load_config, seed_everything


def check_pipeline(project_root=None, config_path="configs/unet.yaml"):
    root = find_project_root(project_root)
    config_path = Path(config_path)
    config = load_config(config_path if config_path.is_absolute() else root / config_path)
    loaders = create_dataloaders(root, config)
    batch = next(iter(loaders["train"]))
    val_batch = next(iter(loaders["val"]))
    for name, current in (("train", batch), ("val", val_batch)):
        if current["image"].shape[1:] != (3, *config["data"]["image_size"]):
            raise ValueError(f"{name}: unexpected image shape")
        if current["mask"].shape[1:] != (1, *config["data"]["image_size"]):
            raise ValueError(f"{name}: unexpected mask shape")
        if not torch.isfinite(current["image"]).all():
            raise ValueError(f"{name}: nonfinite image")
        if not ((current["mask"] == 0) | (current["mask"] == 1)).all():
            raise ValueError(f"{name}: nonbinary mask")
        if current["valid_mask"].shape != current["mask"].shape or not current["valid_mask"].flatten(1).any(1).all():
            raise ValueError(f"{name}: invalid content mask")
        if current["mask"][~current["valid_mask"]].any():
            raise ValueError(f"{name}: padding must have background labels")

    # Analytical examples, not model predictions or reported experimental scores.
    truth = torch.tensor([[[1., 0.], [0., 0.]], [[0., 0.], [0., 0.]]])
    perfect = binary_metrics(truth, truth, from_logits=False)
    if not torch.equal(perfect["dice"], torch.ones(2)) or not torch.equal(perfect["iou"], torch.ones(2)):
        raise RuntimeError("Perfect/empty-mask metric checks failed")
    disjoint = binary_metrics(1 - truth[:1], truth[:1], from_logits=False)
    if disjoint["dice"].item() != 0 or disjoint["iou"].item() != 0:
        raise RuntimeError("Disjoint-mask metric check failed")
    # Exercise variable-resolution evaluation with dummy background logits only.
    dummy = torch.full_like(val_batch["mask"], -20)
    original_scores = metrics_at_original_size(dummy, val_batch["original_mask"], val_batch["geometry"])
    if len(original_scores["dice"]) != len(val_batch["image_id"]):
        raise RuntimeError("Original-resolution batch length differs")
    seed_everything(config["seed"])
    figure = save_transform_preview(loaders["train"].dataset, root / "results/figures/step3_transforms.png")
    summary = {
        "step": 3, "status": "data_pipeline_checked_no_model_trained",
        "sample_counts": {name: len(loader.dataset) for name, loader in loaders.items()},
        "train_image_shape": list(batch["image"].shape), "train_mask_shape": list(batch["mask"].shape),
        "mask_values": sorted(batch["mask"].unique().tolist()),
        "resize_mode": config["data"]["resize_mode"],
        "validation_geometry": val_batch["geometry"],
        "train_valid_pixel_counts": batch["valid_mask"].flatten(1).sum(1).tolist(),
        "validation_original_sizes": [list(size) for size in val_batch["original_size"]],
        "checks": {"batch_shapes_valid": True, "binary_masks": True, "padding_masks_valid": True,
                   "analytical_metric_cases": True, "original_resolution_path": True},
        "python": platform.python_version(), "torch": torch.__version__,
        "config": config, "figure": figure.relative_to(root).as_posix(),
    }
    output = root / "results/tables/step3_check.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path)
    parser.add_argument("--config", default="configs/unet.yaml")
    args = parser.parse_args()
    check_pipeline(args.project_root, args.config)


if __name__ == "__main__":
    main()
