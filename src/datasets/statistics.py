"""Pair validation, original-resolution statistics and label illustrations."""
from collections import Counter
from pathlib import Path
from statistics import mean, median, quantiles
from PIL import Image, ImageDraw
from .common import DATASET_SPECS, MASK_THRESHOLD, save_csv

def files_by_name(folder, extension):
    return {path.name: path for path in folder.iterdir() if path.is_file() and path.suffix.lower() == extension}

def scan_dataset(data_dir, dataset, extension, expected_pairs):
    folder = data_dir / dataset
    images = files_by_name(folder / 'images', extension)
    masks = files_by_name(folder / 'masks', extension)
    print(f'{dataset}: {len(images)} ảnh, {len(masks)} mask')
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        raise ValueError(f'{dataset}: {len(missing_masks)} ảnh thiếu mask (ví dụ: {missing_masks[:5]}); {len(missing_images)} mask thiếu ảnh (ví dụ: {missing_images[:5]}).')
    if len(images) != expected_pairs:
        raise ValueError(f'{dataset}: cần {expected_pairs} cặp, thực tế có {len(images)}.')
    rows = []
    for filename, image_path in sorted(images.items()):
        try:
            with Image.open(image_path) as image:
                image.load()
                width, height = image.size
            with Image.open(masks[filename]) as mask:
                mask_gray = mask.convert('L')
        except OSError as exc:
            raise OSError(f'{dataset}/{filename}: không đọc được ảnh hoặc mask.') from exc
        if mask_gray.size != (width, height):
            raise ValueError(f'{dataset}/{filename}: ảnh {width}×{height}, mask {mask_gray.size}.')
        histogram = mask_gray.histogram()
        polyp_pixels = sum(histogram[MASK_THRESHOLD + 1:])
        image_pixels = width * height
        rows.append({'dataset': dataset, 'filename': filename, 'width': width, 'height': height, 'resolution': f'{width}x{height}', 'polyp_pixels': polyp_pixels, 'area_ratio': polyp_pixels / image_pixels, 'empty_mask': polyp_pixels == 0, 'all_foreground_mask': polyp_pixels == image_pixels, 'has_intermediate_gray': any(histogram[1:255])})
    return rows

def scan_datasets(project_root):
    data_dir = Path(project_root) / "data"
    return {name: scan_dataset(data_dir, name, ext, count)
            for name, (ext, count) in DATASET_SPECS.items()}


def resolution_statistics(rows_by_dataset):
    resolution_rows = []
    for dataset, rows in rows_by_dataset.items():
        counts = Counter(row["resolution"] for row in rows)
        ranked = counts.most_common()
        print(f"\n{dataset}: {len(rows)} cặp, {len(counts)} độ phân giải")
        for resolution, count in ranked[:10]:
            print(f"  {resolution}: {count} ảnh")
        if len(ranked) > 10:
            print("  Xem các độ phân giải còn lại trong resolution_counts.csv.")

        for resolution, count in ranked:
            resolution_rows.append({
                "dataset": dataset, "resolution": resolution, "image_count": count,
            })

        ratios = [row["area_ratio"] for row in rows]
        print(
            "Tỷ lệ diện tích polyp (min / trung vị / trung bình / max): "
            f"{min(ratios):.2%} / {median(ratios):.2%} / {mean(ratios):.2%} / {max(ratios):.2%}"
        )
        empty_masks = [row["filename"] for row in rows if row["empty_mask"]]
        full_masks = [row["filename"] for row in rows if row["all_foreground_mask"]]
        print(f"Mask rỗng: {len(empty_masks)} | Mask toàn polyp: {len(full_masks)}")
        print(f"Mask có mức xám trung gian: {sum(row['has_intermediate_gray'] for row in rows)}")
        if empty_masks or full_masks:
            print(f"Ví dụ cần xem lại — mask rỗng: {empty_masks[:5]}; toàn polyp: {full_masks[:5]}")
    return resolution_rows


def assign_size_groups(project_root, rows_by_dataset):
    PROJECT_ROOT = Path(project_root)


    train_file = PROJECT_ROOT / "splits" / "kvasir_train.txt"
    train_ids = []
    if train_file.is_file():
        for line in train_file.read_text(encoding="utf-8-sig").splitlines():
            entry = line.partition("#")[0].strip()
            if entry:
                train_ids.append(Path(entry).stem)

    if train_ids:
        kvasir_by_id = {Path(row["filename"]).stem: row for row in rows_by_dataset["Kvasir-SEG"]}
        unknown_ids = sorted(set(train_ids) - kvasir_by_id.keys())
        if unknown_ids:
            raise ValueError(f"Không tìm thấy ID Kvasir train trong dữ liệu: {unknown_ids[:5]}")
        if len(train_ids) != len(set(train_ids)):
            raise ValueError("splits/kvasir_train.txt có ID bị lặp.")

        train_ratios = [
            kvasir_by_id[image_id]["area_ratio"]
            for image_id in train_ids if not kvasir_by_id[image_id]["empty_mask"]
        ]
        if len(train_ratios) < 3:
            raise ValueError("Cần ít nhất 3 mask không rỗng trong Kvasir train để tính phân vị.")
        q1, q2 = quantiles(train_ratios, n=3, method="inclusive")
        size_rule = "kvasir_train_tertiles"
        print(f"Mốc từ {len(train_ratios)} ảnh Kvasir train: {q1:.2%} và {q2:.2%}")
    else:
        q1, q2 = 0.05, 0.20
        size_rule = "fixed_preview"
        print(f"Chưa có split train. Mốc THAM KHẢO TẠM: {q1:.0%} và {q2:.0%}")


    def polyp_size_group(area_ratio):
        if area_ratio == 0:
            return "empty"
        if area_ratio <= q1:
            return "small"
        if area_ratio <= q2:
            return "medium"
        return "large"


    size_rows = []
    for dataset, rows in rows_by_dataset.items():
        for row in rows:
            row["size_group"] = polyp_size_group(row["area_ratio"])
        counts = Counter(row["size_group"] for row in rows)
        print(
            f"{dataset}: nhỏ {counts['small']} | vừa {counts['medium']} | "
            f"lớn {counts['large']} | rỗng {counts['empty']}"
        )
        size_rows.append({
            "dataset": dataset, "rule": size_rule,
            "small_upper_ratio": q1, "medium_upper_ratio": q2,
            **{group: counts[group] for group in ("small", "medium", "large", "empty")},
        })
    return size_rows, (q1, q2), size_rule


def load_panels(data_dir, dataset, filename):
    folder = data_dir / dataset
    with Image.open(folder / 'images' / filename) as source:
        image = source.convert('RGB')
    with Image.open(folder / 'masks' / filename) as source:
        mask = source.convert('L').point(lambda value: 255 if value > MASK_THRESHOLD else 0)
    red = Image.new('RGB', image.size, 'red')
    overlay = Image.composite(Image.blend(image, red, 0.4), image, mask)
    return (image, mask, overlay)

def save_sample_figures(project_root, rows_by_dataset, output_dir):
    data_dir = Path(project_root) / "data"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    column_width, panel_height = 300, 210
    header_height, row_height = 24, 240
    for dataset, rows in rows_by_dataset.items():
        samples = []
        for group in ("small", "medium", "large"):
            candidates = sorted(
                (row for row in rows if row["size_group"] == group),
                key=lambda row: row["area_ratio"],
            )
            if candidates:
                samples.append(candidates[len(candidates) // 2])

        if not samples:
            print(f"{dataset}: không có mask polyp để minh họa.")
            continue

        canvas = Image.new(
            "RGB", (3 * column_width, header_height + len(samples) * row_height), "white"
        )
        draw = ImageDraw.Draw(canvas)
        for column, title in enumerate(("Anh goc", "Mask that (>127)", "Overlay")):
            draw.text((column * column_width + 8, 6), title, fill="black")

        for index, row in enumerate(samples):
            y = header_height + index * row_height
            draw.text(
                (8, y + 4), f"{row['filename']} | {row['size_group']} | {row['area_ratio']:.2%}",
                fill="black",
            )
            for column, panel in enumerate(load_panels(data_dir, dataset, row["filename"])):
                resample = Image.Resampling.NEAREST if column == 1 else Image.Resampling.BILINEAR
                panel.thumbnail((column_width - 16, panel_height), resample=resample)
                x = column * column_width + (column_width - panel.width) // 2
                panel_y = y + 22 + (panel_height - panel.height) // 2
                canvas.paste(panel, (x, panel_y))

        figure_path = output_dir / f"{dataset}_samples.png"
        canvas.save(figure_path)
        print(f"{dataset}: {figure_path}")
        paths.append(figure_path)
    return paths


