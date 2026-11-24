"""Read-only audit of frozen splits against current screening; no repartition."""
import json
from collections import Counter
from pathlib import Path

from .common import file_sha256


def audit_fixed_splits(root, fingerprints, exact_groups, reviews):
    root = Path(root)
    definitions = {'train': ('Kvasir-SEG', '.jpg', 'kvasir_train.txt'),
                   'val': ('Kvasir-SEG', '.jpg', 'kvasir_val.txt'),
                   'test': ('Kvasir-SEG', '.jpg', 'kvasir_test.txt'),
                   'external': ('CVC-ClinicDB', '.png', 'clinicdb_external.txt')}
    labels = {}
    counts = {}
    for split, (dataset, suffix, filename) in definitions.items():
        entries = [s.partition('#')[0].strip() for s in (root / 'splits' / filename).read_text(encoding='utf-8-sig').splitlines()]
        ids = [f'{dataset}/{Path(s).stem}{suffix}' for s in entries if s]
        if not ids or len(ids) != len(set(ids)) or set(ids) & labels.keys():
            raise ValueError(f'Empty, repeated or overlapping split: {split}')
        if not set(ids) <= fingerprints.keys():
            raise ValueError(f'Unknown image in split: {split}')
        labels.update(dict.fromkeys(ids, split))
        counts[split] = len(ids)
    manifest = json.loads((root / 'data/dataset_manifest.json').read_text(encoding='utf-8'))
    checked = 0
    for dataset, details in manifest['datasets'].items():
        for relative, digest in details['files_sha256'].items():
            if file_sha256(root / 'data' / dataset / relative) != digest:
                raise ValueError(f'Source checksum mismatch: {dataset}/{relative}')
            checked += 1
    edges = [(g[0], member) for _, g in exact_groups for member in g[1:]]
    edges += [(r['image_a'], r['image_b']) for r in reviews if r['decision'] in ('confirmed', 'unsure')]
    conflicts = [{'image_a': a, 'image_b': b} for a, b in edges
                 if a in labels and b in labels and labels[a] != labels[b]]
    pending = [r for r in reviews if r['scope'] != 'CVC-ClinicDB' and r['decision'] == 'pending']
    result = {'status': 'provisional' if pending or conflicts else 'screening_checked',
              'operation': 'audit_existing_splits_no_repartition', 'split_counts': counts,
              'source_files_verified': checked, 'known_protected_pair_conflicts': conflicts,
              'unresolved_internal_or_cross_pairs': len(pending),
              'review_counts': {scope: dict(Counter(r['decision'] for r in reviews if r['scope'] == scope))
                                for scope in sorted({r['scope'] for r in reviews})},
              'limitations': 'Pending candidates are not confirmed duplicates; patient independence is not verified.'}
    output = root / 'results/tables/step2_fixed_split_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
