"""Behavioral checks for the data-preparation module boundaries (no network)."""

import copy
import hashlib
import json
import shutil
import unittest
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path

from PIL import Image

from src.datasets.common import DATASET_SPECS
from src.datasets.duplicates import connected_components, load_review
from src.datasets.splitting import build_split_plan, validate_split_plan, write_split_artifacts
from src.datasets.statistics import scan_dataset


@contextmanager
def temporary_workspace():
    """Use ordinary inherited Windows ACLs inside the writable project workspace."""
    parent = (Path(__file__).resolve().parents[1] / 'logs/test-fixtures').resolve()
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / uuid.uuid4().hex
    folder.mkdir()
    try:
        yield folder
    finally:
        if folder.resolve().parent != parent:
            raise RuntimeError('Refusing cleanup outside test fixture directory')
        shutil.rmtree(folder)


def split_fixture():
    ids = [f"Kvasir-SEG/k{i:02d}.jpg" for i in range(20)]
    ids += ["CVC-ClinicDB/1.png", "CVC-ClinicDB/2.png"]
    fingerprints = {key: {} for key in ids}
    reviews = [
        {"pair_id": "internal", "image_a": ids[0], "image_b": ids[1],
         "scope": "Kvasir-SEG", "decision": "unsure"},
        {"pair_id": "cross", "image_a": ids[2], "image_b": ids[-1],
         "scope": "cross_dataset", "decision": "confirmed"},
    ]
    return fingerprints, reviews


class PreparationTests(unittest.TestCase):
    def test_uncertain_and_cross_dataset_links(self):
        fingerprints, reviews = split_fixture()
        plan = build_split_plan(fingerprints, [], reviews)
        splits = plan['split_ids']
        labels = {key: name for name, ids in splits.items() for key in ids}
        self.assertEqual(labels['Kvasir-SEG/k00.jpg'], labels['Kvasir-SEG/k01.jpg'])
        self.assertNotIn('Kvasir-SEG/k02.jpg', labels)
        self.assertEqual(set(splits['external']), {'CVC-ClinicDB/1.png', 'CVC-ClinicDB/2.png'})
        reordered = build_split_plan(dict(reversed(list(fingerprints.items()))), [], list(reversed(reviews)))
        self.assertEqual(splits, reordered['split_ids'])

    def test_pending_internal_review_blocks_split(self):
        fingerprints, reviews = split_fixture()
        reviews[0]['decision'] = 'pending'
        with self.assertRaisesRegex(ValueError, 'Pending'):
            build_split_plan(fingerprints, [], reviews)

    def test_transitive_group_and_invalid_plans(self):
        self.assertEqual(connected_components('abcd', [('a', 'b'), ('b', 'c')]), [['a', 'b', 'c'], ['d']])
        fingerprints, reviews = split_fixture()
        prepared = build_split_plan(fingerprints, [], reviews)
        original = prepared['split_ids']
        cases = []
        bad = copy.deepcopy(original); bad['train'].append(bad['train'][0]); cases.append(bad)
        bad = copy.deepcopy(original); bad['val'].append(bad['train'][0]); cases.append(bad)
        bad = copy.deepcopy(original); bad['train'].pop(); cases.append(bad)
        bad = copy.deepcopy(original); bad['external'].pop(); cases.append(bad)
        bad = copy.deepcopy(original); bad['train'][0] = 'unknown'; cases.append(bad)
        bad = copy.deepcopy(original)
        label = next(name for name in ('train', 'val', 'test') if 'Kvasir-SEG/k00.jpg' in bad[name])
        other = next(name for name in ('train', 'val', 'test') if name != label)
        bad[label].remove('Kvasir-SEG/k00.jpg'); bad[other].append('Kvasir-SEG/k00.jpg'); cases.append(bad)
        for bad in cases:
            with self.subTest(plan=bad), self.assertRaises(ValueError):
                validate_split_plan(plan=bad, **prepared['context'])

    def test_scan_pairs_by_name_and_mask_threshold(self):
        with temporary_workspace() as folder:
            root = Path(folder)
            for kind in ('images', 'masks'):
                (root/'sample'/kind).mkdir(parents=True)
            Image.new('RGB', (2, 2)).save(root/'sample/images/a.png')
            mask = Image.new('L', (2, 2)); mask.putdata([0, 127, 128, 255])
            mask.save(root/'sample/masks/a.png')
            rows = scan_dataset(root, 'sample', '.png', 1)
            self.assertEqual(rows[0]['area_ratio'], 0.5)
            Image.new('L', (3, 2)).save(root/'sample/masks/a.png')
            with self.assertRaises(ValueError):
                scan_dataset(root, 'sample', '.png', 1)
            (root/'sample/masks/a.png').rename(root/'sample/masks/b.png')
            with self.assertRaises(ValueError):
                scan_dataset(root, 'sample', '.png', 1)

    def test_export_conflict_never_overwrites_locked_split(self):
        fingerprints, reviews = split_fixture()
        prepared = build_split_plan(fingerprints, [], reviews)
        with temporary_workspace() as folder:
            root = Path(folder); (root/'splits').mkdir()
            (root/'data').mkdir()
            rows = {name: [] for name in DATASET_SPECS}
            datasets = {name: {'files_sha256': {}} for name in DATASET_SPECS}
            for image_id in fingerprints:
                dataset, filename = image_id.split('/', 1)
                rows[dataset].append({'filename': filename})
                for kind in ('images', 'masks'):
                    path = root/'data'/dataset/kind/filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(b'fixture-data')
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    datasets[dataset]['files_sha256'][kind+'/'+filename] = digest
                    fingerprints[image_id]['file_sha256'] = digest
            (root/'data/dataset_manifest.json').write_text(json.dumps({'datasets': datasets}), encoding='utf-8')
            locked = root/'splits/kvasir_val.txt'
            locked.write_text('locked-user-split\n', encoding='utf-8')
            before = locked.read_bytes()
            with self.assertRaisesRegex(ValueError, 'refusing'):
                write_split_artifacts(root, rows, fingerprints, reviews, prepared)
            self.assertEqual(locked.read_bytes(), before)
            self.assertFalse((root/'splits/kvasir_train.txt').exists())

    def test_saved_review_is_required_not_invented(self):
        candidate = {'pair_id': 'unseen', 'scope': 'Kvasir-SEG', 'image_a': 'a', 'image_b': 'b'}
        with temporary_workspace() as folder:
            reviews = load_review(Path(folder), [candidate])
            self.assertEqual(reviews[0]['decision'], 'pending')

    def test_zip_rejects_unsafe_paths_and_existing_files(self):
        from src.datasets.download import DatasetDownloader
        with temporary_workspace() as folder:
            root = Path(folder); (root/'src/datasets').mkdir(parents=True)
            (root/'README.md').touch()
            downloader = DatasetDownloader(root)
            archive = root/'unsafe.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../images/x.png', b'payload')
            with zipfile.ZipFile(archive) as z, self.assertRaises(ValueError):
                downloader.archive_members(z, {'extension': '.png'})
            target = root/'existing.png'; target.write_bytes(b'original')
            with self.assertRaises(FileExistsError):
                downloader.write_new_file(target, b'replacement')
            self.assertEqual(target.read_bytes(), b'original')


if __name__ == '__main__':
    unittest.main()
