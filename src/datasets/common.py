"""Paths, fixed dataset specifications and UTF-8 artifact writing."""

import csv
import io
from hashlib import sha256
from pathlib import Path

DATASET_SPECS = {"Kvasir-SEG": (".jpg", 1000), "CVC-ClinicDB": (".png", 612)}
MASK_THRESHOLD = 127


def find_project_root(start=None):
    start = Path(start or Path.cwd()).expanduser().resolve()
    for folder in (start, *start.parents):
        if (folder / "src/datasets").is_dir() and (folder / "README.md").is_file():
            return folder
    raise FileNotFoundError(f"Không tìm thấy gốc dự án từ {start}")


def save_csv(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8-sig", newline="")
    return path


def file_sha256(path):
    digest = sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
