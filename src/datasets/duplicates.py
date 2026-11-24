"""Hash screening and review decisions; no claim of patient independence."""
import csv
import json
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import combinations
from math import cos, pi, sqrt
from pathlib import Path
from statistics import median
from PIL import Image, ImageOps, ImageDraw
from .common import save_csv, letterbox_geometry

DHASH_MAX = 6
PHASH_MAX = 6
HASH_PREPROCESSING = "centered_letterbox_lanczos_v2"
DUPLICATE_SCOPES = ("Kvasir-SEG", "cross_dataset", "CVC-ClinicDB")
CANDIDATE_COLUMNS = ["pair_id", "scope", "image_a", "image_b", "dhash_distance", "phash_distance", "screening_reason"]
DCT_8_OF_32 = [
    [(1 / sqrt(2) if k == 0 else 1) * cos(pi * (x + 0.5) * k / 32)
     for x in range(32)]
    for k in range(8)
]


def hash_thumbnail(image, size):
    """Resize with preserved aspect ratio and centered black padding.

    size is PIL (width, height). The source image is never modified.
    Padding becomes part of the descriptor, so similarity still needs review.
    """
    if image.mode not in {"RGB", "L"}:
        raise ValueError("Hash thumbnails require RGB or grayscale images")
    geometry = letterbox_geometry((image.height, image.width), (size[1], size[0]))
    height, width = geometry["resized_size"]
    left, top, _, _ = geometry["padding"]
    content = image.resize((width, height), Image.Resampling.LANCZOS)
    canvas = Image.new(image.mode, size, 0)
    canvas.paste(content, (left, top))
    return canvas


def dhash_image(gray):
    pixels = list(hash_thumbnail(gray, (9, 8)).tobytes())
    return sum(((pixels[y * 9 + x + 1] > pixels[y * 9 + x]) << y * 8 + x for y in range(8) for x in range(8)))

def phash_image(gray):
    pixels = hash_thumbnail(gray, (32, 32)).tobytes()
    horizontal = [[sum((pixel * weight for pixel, weight in zip(pixels[i:i + 32], basis))) for basis in DCT_8_OF_32] for i in range(0, len(pixels), 32)]
    columns = list(zip(*horizontal))
    low_frequencies = [sum((value * weight for value, weight in zip(column, basis))) for basis in DCT_8_OF_32 for column in columns]
    cutoff = median(low_frequencies[1:])
    return sum(((value > cutoff) << bit for bit, value in enumerate(low_frequencies) if bit != 0))

def fingerprint_image(path):
    file_hash = sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image_file:
        rgb = ImageOps.exif_transpose(image_file).convert('RGB')
    normalized = hash_thumbnail(rgb, (64, 64))
    gray = rgb.convert('L')
    return {'file_sha256': file_hash, 'rgb64_sha256': sha256(normalized.tobytes()).hexdigest(),
            'dhash': dhash_image(gray), 'phash': phash_image(gray),
            'hash_preprocessing': HASH_PREPROCESSING}


def saved_review_pairs(project_root, fingerprints):
    """Retain reviewed pairs that fall outside a changed hash threshold/version.

    Decisions are still read by load_review, with the local review file taking
    precedence. Only pairs with a nonpending decision need preservation.
    """
    root = Path(project_root)
    pairs = {}
    for path in (root / "splits/duplicate_review.csv", root / "logs/duplicate_check/near_candidates.csv"):
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if not {"pair_id", "image_a", "image_b"} <= row.keys():
                    raise ValueError(f"Review history lacks pair identities: {path}")
                pairs[row["pair_id"]] = (row["image_a"], row["image_b"])
    decisions = {}
    for path in (root / "splits/duplicate_review.csv", root / "logs/duplicate_check/review_decisions.csv"):
        if path.exists():
            with path.open(encoding="utf-8-sig", newline="") as source:
                for row in csv.DictReader(source):
                    decision = (row.get("decision") or "pending").strip().lower()
                    if decision != "pending" or row["pair_id"] not in decisions:
                        decisions[row["pair_id"]] = decision
    kept = []
    for pair_id, decision in sorted(decisions.items()):
        if decision == "pending":
            continue
        if pair_id not in pairs:
            raise ValueError(f"Cannot recover image IDs for reviewed pair: {pair_id}")
        left, right = pairs[pair_id]
        expected = "p-" + sha256("\n".join(sorted((left, right))).encode()).hexdigest()[:12]
        if pair_id != expected or left not in fingerprints or right not in fingerprints:
            raise ValueError(f"Review history does not match current images: {pair_id}")
        kept.append((left, right))
    return kept

def duplicate_scope(members):
    sources = {image_id.split('/', 1)[0] for image_id in members}
    return next(iter(sources)) if len(sources) == 1 else 'cross_dataset'

def connected_components(ids, edges):
    parent = {key: key for key in sorted(ids)}

    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    for left, right in sorted(edges):
        a, b = (find(left), find(right))
        if a != b:
            parent[max(a, b)] = min(a, b)
    buckets = defaultdict(list)
    for key in sorted(parent):
        buckets[find(key)].append(key)
    return sorted(buckets.values(), key=lambda group: group[0])

def fingerprint_datasets(project_root, rows_by_dataset):
    data_dir = Path(project_root) / "data"
    ids = sorted(f"{name}/{row['filename']}" for name, rows in rows_by_dataset.items() for row in rows)
    return {key: fingerprint_image(data_dir / key.split("/", 1)[0] / "images" / key.split("/", 1)[1]) for key in ids}


def screen_duplicates(project_root, fingerprints):
    # Read history BEFORE updating near_candidates.csv.
    reviewed_pairs = saved_review_pairs(project_root, fingerprints)
    DUPLICATE_DIR = Path(project_root) / "logs/duplicate_check"
    DUPLICATE_DIR.mkdir(parents=True, exist_ok=True)
    def write_duplicate_csv(filename, columns, rows):
        return save_csv(DUPLICATE_DIR / filename, columns, rows)
    exact_groups = []
    for level in ("file_sha256", "rgb64_sha256"):
        buckets = defaultdict(list)
        for image_id, hashes in fingerprints.items():
            buckets[hashes[level]].append(image_id)
        exact_groups.extend((level, group) for group in buckets.values() if len(group) > 1)

    write_duplicate_csv(
        "exact_groups.csv", ["level", "scope", "image_count", "members"],
        ({"level": level, "scope": duplicate_scope(group), "image_count": len(group),
          "members": " | ".join(group)} for level, group in exact_groups),
    )

    near_candidates = []
    for (left, a), (right, b) in combinations(fingerprints.items(), 2):
        if a["file_sha256"] == b["file_sha256"] or a["rgb64_sha256"] == b["rgb64_sha256"]:
            continue
        d = (a["dhash"] ^ b["dhash"]).bit_count()
        p = (a["phash"] ^ b["phash"]).bit_count()
        if d <= DHASH_MAX or p <= PHASH_MAX:
            pair_key = "\n".join(sorted((left, right)))
            near_candidates.append({
                "pair_id": "p-" + sha256(pair_key.encode("utf-8")).hexdigest()[:12],
                "scope": duplicate_scope((left, right)),
                "image_a": left, "image_b": right, "dhash_distance": d, "phash_distance": p,
            })


    # Historical candidates remain explicit even if fresh hashes exceed thresholds.
    historical_pairs = reviewed_pairs + [
        ("Kvasir-SEG/cju2i6acqvo6l0799u20fift8.jpg", "Kvasir-SEG/cju33w4sdcivk0855x879zht7.jpg"),
        ("Kvasir-SEG/cju41r6v2lcww0871ps8k8pf5.jpg", "Kvasir-SEG/cju6vqarjv7yo0987q4b1btk1.jpg"),
    ]
    for row in near_candidates:
        row["screening_reason"] = "hash_threshold"
    existing_pairs = {row["pair_id"] for row in near_candidates}
    for left, right in historical_pairs:
        if left not in fingerprints or right not in fingerprints:
            continue
        pair_id = "p-" + sha256("\n".join(sorted((left, right))).encode()).hexdigest()[:12]
        if pair_id not in existing_pairs:
            a, b = fingerprints[left], fingerprints[right]
            near_candidates.append({"pair_id": pair_id, "scope": duplicate_scope((left, right)),
                "image_a": left, "image_b": right,
                "dhash_distance": (a["dhash"] ^ b["dhash"]).bit_count(),
                "phash_distance": (a["phash"] ^ b["phash"]).bit_count(),
                "screening_reason": "historical_manual_review"})
            existing_pairs.add(pair_id)

    near_candidates.sort(key=lambda row: (
        DUPLICATE_SCOPES.index(row["scope"]),
        min(row["dhash_distance"], row["phash_distance"]),
        max(row["dhash_distance"], row["phash_distance"]),
        row["image_a"], row["image_b"],
    ))
    candidate_by_id = {row["pair_id"]: row for row in near_candidates}
    if len(candidate_by_id) != len(near_candidates):
        raise ValueError("Duplicate shortened pair ID")
    candidate_columns = ["pair_id", "scope", "image_a", "image_b", "dhash_distance", "phash_distance", "screening_reason"]
    write_duplicate_csv("near_candidates.csv", candidate_columns, near_candidates)
    from PIL import __version__ as pillow_version
    screening_manifest = {"hash_preprocessing": HASH_PREPROCESSING,
                          "pillow_version": pillow_version,
                          "thumbnail_sizes_wh": {"rgb": [64, 64], "dhash": [9, 8], "phash": [32, 32]},
                          "padding": "black, centered; odd extra right/bottom",
                          "dhash_max": DHASH_MAX, "phash_max": PHASH_MAX}
    (DUPLICATE_DIR / "screening_manifest.json").write_text(
        json.dumps(screening_manifest, indent=2) + "\n", encoding="utf-8")

    exact_counts = Counter((level, duplicate_scope(group)) for level, group in exact_groups)
    candidate_counts = Counter(row["scope"] for row in near_candidates)
    print(f"Số cặp được so sánh: {len(fingerprints) * (len(fingerprints) - 1) // 2:,}")
    for scope in DUPLICATE_SCOPES:
        print(
            f"{scope}: {exact_counts['file_sha256', scope]} nhóm trùng file | "
            f"{exact_counts['rgb64_sha256', scope]} nhóm trùng RGB 64x64 | "
            f"{candidate_counts[scope]} cặp nghi gần trùng"
        )
    print("Thư mục kết quả:", DUPLICATE_DIR)
    return exact_groups, near_candidates


def load_review(project_root, near_candidates):
    PROJECT_ROOT = Path(project_root)
    REVIEW_FILE = PROJECT_ROOT / "logs/duplicate_check/review_decisions.csv"
    candidate_by_id = {row["pair_id"]: row for row in near_candidates}
    candidate_columns = CANDIDATE_COLUMNS
    def write_duplicate_csv(filename, columns, rows):
        return save_csv(PROJECT_ROOT / "logs/duplicate_check" / filename, columns, rows)
    decisions = {}
    review_snapshot = PROJECT_ROOT / 'splits' / 'duplicate_review.csv'
    for input_path in (review_snapshot, REVIEW_FILE):
        if not input_path.exists():
            continue
        with input_path.open(encoding='utf-8-sig', newline='') as source:
            reader = csv.DictReader(source)
            if not {'pair_id', 'decision', 'note'}.issubset(reader.fieldnames or []):
                raise ValueError(f'Invalid duplicate-review input: {input_path}')
            seen_ids = set()
            for row in reader:
                pair_id = (row['pair_id'] or '').strip().lower()
                decision = (row['decision'] or '').strip().lower() or 'pending'
                if not pair_id or pair_id in seen_ids or decision not in {'pending', 'confirmed', 'different', 'unsure'}:
                    raise ValueError(f'Invalid duplicate-review input: {input_path} {reader.line_num}')
                seen_ids.add(pair_id)
                if decision != 'pending' or pair_id not in decisions:
                    decisions[pair_id] = (decision, row['note'] or '')
    visual_review_updates = {'p-461f15ca4800': ('unsure', 'Codex visual review: similar polyp shape; insufficient matching tissue detail. Keep together conservatively at split time.'), 'p-52e6ab71dc03': ('unsure', 'Codex visual review: similar polyp shape; insufficient matching tissue detail. Keep together conservatively at split time.')}
    for pair_id, decision in visual_review_updates.items():
        if pair_id in candidate_by_id and decisions.get(pair_id, ('pending', ''))[0] == 'pending':
            decisions[pair_id] = decision
    # Pending rows from a previous screening are not decisions and can disappear.
    unused_ids = sorted(key for key in set(decisions) - set(candidate_by_id)
                        if decisions[key][0] != "pending")
    if unused_ids:
        raise ValueError(f'Invalid duplicate-review input: {unused_ids[:5]}')
    decisions = {key: value for key, value in decisions.items() if key in candidate_by_id}
    write_duplicate_csv('review_decisions.csv', ['pair_id', 'decision', 'note'], ({'pair_id': key, 'decision': value[0], 'note': value[1]} for key, value in sorted(decisions.items())))
    review_rows = [{**row, 'decision': decisions.get(row['pair_id'], ('pending', ''))[0], 'note': decisions.get(row['pair_id'], ('pending', ''))[1]} for row in near_candidates]
    write_duplicate_csv('near_review_status.csv', candidate_columns + ['decision', 'note'], review_rows)
    for scope in DUPLICATE_SCOPES:
        print(scope, dict(Counter((row['decision'] for row in review_rows if row['scope'] == scope))))
    return review_rows


def save_confirmed_groups(project_root, fingerprints, exact_groups, review_rows):
    exact_edges = [(group[0], member) for _, group in exact_groups for member in group[1:]]
    edges = exact_edges + [(r["image_a"], r["image_b"]) for r in review_rows if r["decision"] == "confirmed"]
    groups = [g for g in connected_components(fingerprints, edges) if len(g) > 1]
    rows = [{"group_id": f"G{i:03d}", "scope": duplicate_scope(g), "image_count": len(g), "members": " | ".join(g)} for i,g in enumerate(groups,1)]
    save_csv(Path(project_root)/"logs/duplicate_check/confirmed_groups.csv", ["group_id","scope","image_count","members"], rows)
    return groups


def draw_candidate_pairs(data_dir, pairs, panel_width=300, panel_height=165):
    row_height = panel_height + 36
    canvas = Image.new('RGB', (panel_width * 2, row_height * len(pairs)), 'white')
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(pairs):
        y = index * row_height
        d, p = (row['dhash_distance'], row['phash_distance'])
        draw.text((5, y + 2), f"{row['pair_id']} | dHash {d} | pHash {p}", fill='black')
        for column, image_id in enumerate((row['image_a'], row['image_b'])):
            x = column * panel_width
            draw.text((x + 5, y + 17), image_id, fill='black')
            with Image.open((Path(data_dir) / image_id.split("/", 1)[0] / "images" / image_id.split("/", 1)[1])) as source:
                image = ImageOps.exif_transpose(source).convert('RGB')
            image.thumbnail((panel_width - 10, panel_height), Image.Resampling.BILINEAR)
            canvas.paste(image, (x + (panel_width - image.width) // 2, y + 33))
    return canvas

