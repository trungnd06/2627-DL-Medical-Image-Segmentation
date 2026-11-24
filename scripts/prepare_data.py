"""Reproduce step 2 from existing data; opt in to downloading with --download."""

import argparse
import sys
from pathlib import Path

# Support both `python -m scripts.prepare_data` and `python scripts/prepare_data.py`.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.datasets.common import find_project_root
from src.datasets.preparation import prepare_data


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true",
                        help="Verify/download pinned sources before step 2 (requires Internet).")
    args = parser.parse_args()
    root = find_project_root(args.project_root)
    if args.download:
        from src.datasets.download import DatasetDownloader
        DatasetDownloader(root).run()
    prepare_data(root, seed=args.seed)


if __name__ == "__main__":
    main()
