"""Shared configuration and reproducibility helpers; no work on import."""

import random
from pathlib import Path

import torch


def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    # Reproducibility still depends on platform, library versions and operators.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    """Top-level function so DataLoader workers can import it on Windows."""
    random.seed(torch.initial_seed() % (2 ** 32))


def load_config(path):
    import yaml
    with Path(path).open(encoding="utf-8") as source:
        config = yaml.safe_load(source)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a YAML mapping")
    return config
