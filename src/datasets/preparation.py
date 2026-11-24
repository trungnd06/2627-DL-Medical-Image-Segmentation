"""Coordinate step 2 without requiring a running Jupyter kernel or CUDA."""

from .common import find_project_root
from .statistics import scan_datasets, resolution_statistics
from .duplicates import (
    fingerprint_datasets, screen_duplicates, load_review, save_confirmed_groups,
)
from .splitting import build_split_plan, write_split_artifacts
from .reporting import write_reports


def prepare_data(project_root=None, seed=42):
    """Validate existing data and reproduce splits; never change source images."""
    root = find_project_root(project_root)
    rows = scan_datasets(root)
    resolutions = resolution_statistics(rows)
    fingerprints = fingerprint_datasets(root, rows)
    exact, candidates = screen_duplicates(root, fingerprints)
    reviews = load_review(root, candidates)
    save_confirmed_groups(root, fingerprints, exact, reviews)
    plan = build_split_plan(fingerprints, exact, reviews, seed=seed)
    manifest = write_split_artifacts(root, rows, fingerprints, reviews, plan)
    return write_reports(root, rows, resolutions, reviews, plan, manifest)
