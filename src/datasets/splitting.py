"""Deterministic component-level splitting and conflict-safe artifact writing."""
import json
import random
from collections import Counter
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from .common import DATASET_SPECS, MASK_THRESHOLD, save_csv
from .duplicates import connected_components, CANDIDATE_COLUMNS, HASH_PREPROCESSING

def allocate_groups(groups, seed):
    total = sum(map(len, groups))
    targets = {'train': round(total * 0.7), 'val': round(total * 0.15)}
    targets['test'] = total - sum(targets.values())
    rng = random.Random(seed)
    order = [list(group) for group in sorted(groups, key=lambda g: g[0])]
    rng.shuffle(order)
    order.sort(key=lambda group: -len(group))
    result = {name: [] for name in targets}
    for group in order:
        remaining = {name: targets[name] - len(result[name]) for name in targets}
        fits = [name for name in targets if remaining[name] >= len(group)]
        choices = fits or list(targets)
        chosen = max(choices, key=lambda name: remaining[name])
        result[chosen].extend(group)
    return ({name: sorted(ids) for name, ids in result.items()}, targets)

def validate_split_plan(fingerprints, eligible_ids, clinic_ids, excluded_kvasir, all_components, plan):
    """Reject duplicate, missing, overlapping IDs and split protected components."""
    required = {'train', 'val', 'test', 'external'}
    if set(plan) != required:
        raise ValueError('Expected train, val, test and external split names')
    for label, ids in plan.items():
        if not ids or len(ids) != len(set(ids)):
            raise ValueError(f'{label}: empty split or repeated ID')
        if not set(ids) <= set(fingerprints):
            raise ValueError(f'{label}: unknown image ID')
    for a, b in combinations(required, 2):
        if set(plan[a]) & set(plan[b]):
            raise ValueError(f'Overlap between {a} and {b}')
    actual = set().union(*(set(plan[name]) for name in ('train', 'val', 'test')))
    if actual != eligible_ids or set(plan['external']) != clinic_ids:
        raise ValueError('Incomplete or incorrect internal/external coverage')
    labels = {key: label for label, ids in plan.items() for key in ids}
    if actual & excluded_kvasir:
        raise ValueError('Excluded cross-dataset IDs appear in internal splits')
    for group in all_components:
        used = {labels[key] for key in group if key in labels}
        if len(used) > 1:
            raise ValueError(f'Protected component appears in multiple splits: {group}')
    return True

def build_split_plan(fingerprints, exact_groups, review_rows, seed=42):
    """Group confirmed/unsure links and exclude cross-dataset Kvasir members."""
    duplicate_ids = sorted(fingerprints)
    exact_edges = [(group[0], member) for _, group in exact_groups for member in group[1:]]
    blocking = [r['pair_id'] for r in review_rows if r['scope'] in {'Kvasir-SEG', 'cross_dataset'} and r['decision'] == 'pending']
    if blocking:
        raise ValueError(f'Pending internal/cross-dataset review decisions: {blocking}')
    protective_edges = exact_edges + [(r['image_a'], r['image_b']) for r in review_rows if r['decision'] in {'confirmed', 'unsure'}]
    all_components = connected_components(duplicate_ids, protective_edges)
    kvasir_ids = {key for key in duplicate_ids if key.startswith('Kvasir-SEG/')}
    clinic_ids = {key for key in duplicate_ids if key.startswith('CVC-ClinicDB/')}
    excluded_kvasir = set()
    internal_groups = []
    for group in all_components:
        internal = sorted(set(group) & kvasir_ids)
        if internal and set(group) & clinic_ids:
            excluded_kvasir.update(internal)
        elif internal:
            internal_groups.append(internal)
    eligible_ids = kvasir_ids - excluded_kvasir
    if len(eligible_ids) < 3:
        raise ValueError('Not enough eligible Kvasir images for three splits')
    internal_splits, targets = allocate_groups(internal_groups, seed)
    split_ids = {**internal_splits, "external": sorted(clinic_ids)}
    context = {"fingerprints": fingerprints, "eligible_ids": eligible_ids, "clinic_ids": clinic_ids,
               "excluded_kvasir": excluded_kvasir, "all_components": all_components}
    validate_split_plan(plan=split_ids, **context)
    if allocate_groups(internal_groups, seed)[0] != internal_splits:
        raise RuntimeError("Split is not reproducible for the same seed")
    return {"split_ids": split_ids, "split_targets": targets, "context": context, "seed": seed}


def write_split_artifacts(project_root, rows_by_dataset, fingerprints, review_rows, plan):
    """Verify sources and reject conflicts before writing any split TXT."""
    validate_split_plan(plan=plan['split_ids'], **plan['context'])
    PROJECT_ROOT = Path(project_root)
    DATA_DIR = PROJECT_ROOT / "data"
    duplicate_ids = sorted(fingerprints)
    split_ids, split_targets = plan["split_ids"], plan["split_targets"]
    all_components, excluded_kvasir = plan["context"]["all_components"], plan["context"]["excluded_kvasir"]
    SPLIT_SEED = plan["seed"]
    DHASH_MAX = PHASH_MAX = 6
    candidate_columns = CANDIDATE_COLUMNS
    source_manifest_path = DATA_DIR / 'dataset_manifest.json'
    source_manifest = json.loads(source_manifest_path.read_text(encoding='utf-8'))
    source_metadata = {}
    for dataset, details in source_manifest['datasets'].items():
        source_metadata[dataset] = {key: details.get(key) for key in ('download_url', 'source_page', 'archive_sha256', 'repo_id', 'revision')}
        expected = details['files_sha256']
        actual_paths = {f"{kind}/{row['filename']}" for kind in ('images', 'masks') for row in rows_by_dataset[dataset]}
        if set(expected) != actual_paths:
            raise ValueError(f'Source manifest file coverage differs: {dataset}')
        for relative, digest in expected.items():
            if sha256((DATA_DIR / dataset / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f'Source checksum mismatch: {dataset}/{relative}')
    print('Source manifest verified: 3224 image/mask files')
    SPLIT_DIR = PROJECT_ROOT / 'splits'
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    split_filenames = {'train': 'kvasir_train.txt', 'val': 'kvasir_val.txt', 'test': 'kvasir_test.txt', 'external': 'clinicdb_external.txt'}
    payloads = {}
    for label, filename in split_filenames.items():
        payloads[filename] = ''.join((Path(key).stem + '\n' for key in split_ids[label]))
    for filename, content in payloads.items():
        path = SPLIT_DIR / filename
        if path.exists():
            previous = path.read_text(encoding='utf-8-sig')
            if previous.strip() and previous != content:
                raise ValueError(f'Existing split differs; refusing to overwrite: {path}')
    for filename, content in payloads.items():
        path = SPLIT_DIR / filename
        if not path.exists() or path.read_text(encoding='utf-8-sig') != content:
            path.write_text(content, encoding='utf-8', newline='\n')
    disk_plan = {}
    for label, filename in split_filenames.items():
        dataset = 'CVC-ClinicDB' if label == 'external' else 'Kvasir-SEG'
        extension = DATASET_SPECS[dataset][0]
        entries = (SPLIT_DIR / filename).read_text(encoding='utf-8').splitlines()
        disk_plan[label] = [f'{dataset}/{entry}{extension}' for entry in entries]
    validate_split_plan(plan=disk_plan, **plan["context"])
    labels = {key: label for label, ids in split_ids.items() for key in ids}
    assignments = []
    for group in all_components:
        group_id = 'g-' + sha256('\n'.join(group).encode()).hexdigest()[:16]
        for key in group:
            assignments.append({'image_id': key, 'group_id': group_id, 'group_size': len(group), 'split': labels.get(key, 'excluded_cross_dataset')})
    save_csv(SPLIT_DIR / 'group_assignments.csv', ['image_id', 'group_id', 'group_size', 'split'], assignments)
    save_csv(SPLIT_DIR / 'duplicate_review.csv', candidate_columns + ['decision', 'note'], review_rows)
    save_csv(SPLIT_DIR / 'excluded_kvasir.csv', ['image_id', 'reason'], [{'image_id': key, 'reason': 'confirmed_or_unsure_cross_dataset_component'} for key in sorted(excluded_kvasir)])
    source_fingerprints = []
    for key in duplicate_ids:
        dataset, filename = key.split('/', 1)
        for kind in ('images', 'masks'):
            path = DATA_DIR / dataset / kind / filename
            digest = fingerprints[key]['file_sha256'] if kind == 'images' else sha256(path.read_bytes()).hexdigest()
            source_fingerprints.append({'path': path.relative_to(PROJECT_ROOT).as_posix(), 'sha256': digest})
    save_csv(SPLIT_DIR / 'source_fingerprints.csv', ['path', 'sha256'], source_fingerprints)
    manifest = {
        'schema_version': 1,
        'dataset_sources': source_metadata,
        'source_manifest_sha256': sha256(source_manifest_path.read_bytes()).hexdigest(),
        'split_seed': SPLIT_SEED,
        'ratios': [0.7, 0.15, 0.15],
        'algorithm': 'sorted_components_seeded_shuffle_size_desc_largest_remaining_capacity_v1',
        'mask_threshold': MASK_THRESHOLD,
        'screening': {'hash_preprocessing': HASH_PREPROCESSING, 'dhash_max': DHASH_MAX, 'phash_max': PHASH_MAX, 'rule': 'dhash OR phash plus preserved reviewed pairs'},
        'grouping': 'exact_hash + confirmed + unsure; transitive components',
        'cross_dataset_policy': 'keep ClinicDB external; exclude related Kvasir IDs',
        'counts': {key: len(ids) for key, ids in split_ids.items()},
        'targets': split_targets,
        'excluded_kvasir_count': len(excluded_kvasir),
        'python_version': __import__('platform').python_version(),
        'pillow_version': __import__('PIL').__version__,
        'checksums': {name: sha256((SPLIT_DIR / name).read_bytes()).hexdigest() for name in [*payloads, 'duplicate_review.csv', 'group_assignments.csv', 'excluded_kvasir.csv', 'source_fingerprints.csv']},
        'limitations': ['Image-based screening, no verified patient-level independence', 'ClinicDB-only pending review retained in external; no model selection on external'],
    }
    (SPLIT_DIR / 'split_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print('Verified split artifacts:', SPLIT_DIR)
    return manifest


