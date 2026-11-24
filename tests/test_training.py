"""Loss masking, validation weighting and runnable M1 lifecycle on tiny fixtures."""

import copy
import csv
import json
import unittest
from unittest.mock import patch

import torch
import yaml
from PIL import Image
from torch import nn

from test_data_preparation import temporary_workspace
from src.datasets.common import file_sha256
from src.datasets.dataset import collate_samples, create_dataloaders
from src.datasets.transforms import SegmentationTransform
from src.engine import run_epoch
from src.losses import BCEDiceLoss
from src.training import _small_loaders, run_training
from src.utils import load_config


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_padding_has_zero_loss_gradient_and_no_effect_on_loss(self):
        logits = torch.tensor([[[[0., 1., 100., -100.]]]], requires_grad=True)
        truth = torch.tensor([[[[0., 1., 0., 1.]]]])
        valid = torch.tensor([[[[1, 1, 0, 0]]]], dtype=torch.bool)
        criterion = BCEDiceLoss()
        expected = criterion(logits[..., :2], truth[..., :2], valid[..., :2])
        actual = criterion(logits, truth, valid)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        torch.testing.assert_close(logits.grad[..., 2:], torch.zeros_like(logits.grad[..., 2:]))
        self.assertGreater(logits.grad[..., :2].abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            criterion(logits, truth, torch.zeros_like(valid))

    def test_loss_is_numerically_stable_for_extreme_logits(self):
        logits = torch.tensor([[[[-1000., 1000.]]]], requires_grad=True)
        loss = BCEDiceLoss()(logits, torch.tensor([[[[1., 0.]]]]), torch.ones_like(logits))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_validation_is_image_weighted_and_uses_eval_no_grad(self):
        transform = SegmentationTransform(size=16, mean=(0, 0, 0), std=(1, 1, 1))
        samples = []
        for index, foreground in enumerate((True, True, False)):
            image = Image.new('RGB', (19, 13), 'white')
            mask = Image.new('L', image.size, 255 if foreground else 0)
            sample = transform.apply(image, mask)
            sample.update(image_id=str(index), original_size=(13, 19),
                          original_mask=torch.full((1, 13, 19), float(foreground)))
            samples.append(sample)
        class Foreground(nn.Module):
            def forward(self, x):
                assert not self.training
                assert not torch.is_grad_enabled()
                return torch.full_like(x[:, :1], 10)
        loader = torch.utils.data.DataLoader(samples, batch_size=2, collate_fn=collate_samples)
        summary, rows = run_epoch(Foreground(), loader, BCEDiceLoss(), torch.device('cpu'), progress=False)
        self.assertAlmostEqual(summary['dice'], 2 / 3)
        self.assertEqual(len(rows), 3)
        self.assertEqual(summary['samples'], 3)

    def test_training_artifacts_and_checkpoint_roundtrip(self):
        from pathlib import Path
        from src.models.unet import UNet
        config = load_config(Path(__file__).resolve().parents[1] / 'configs/unet.yaml')
        config = copy.deepcopy(config)
        config['data']['image_size'] = [16, 16]
        config['model']['channels'] = [2, 4, 8, 16, 32]
        config['loader']['batch_size'] = 2
        with temporary_workspace() as root:
            (root / 'src/datasets').mkdir(parents=True)
            (root / 'README.md').touch()
            (root / 'splits').mkdir()
            datasets = {'Kvasir-SEG': {'files_sha256': {}}}
            for index in range(4):
                for kind in ('images', 'masks'):
                    path = root / 'data/Kvasir-SEG' / kind / f'{index}.jpg'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new('RGB' if kind == 'images' else 'L', (20, 16), 'white').save(path)
                    datasets['Kvasir-SEG']['files_sha256'][f'{kind}/{index}.jpg'] = file_sha256(path)
            (root / 'data/dataset_manifest.json').write_text(json.dumps({'datasets': datasets}), encoding='utf-8')
            (root / 'splits/kvasir_train.txt').write_text('0\n1\n', encoding='utf-8')
            (root / 'splits/kvasir_val.txt').write_text('2\n3\n', encoding='utf-8')
            (root / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
            overfit, indices = _small_loaders(create_dataloaders(root, config), config, 'overfit', 2)
            train_ids = [sample['image_id'] for sample in overfit['train'].dataset]
            eval_ids = [sample['image_id'] for sample in overfit['val'].dataset]
            self.assertEqual(train_ids, eval_ids)
            self.assertEqual(len(indices), 2)
            self.assertFalse(overfit['train'].dataset.dataset.transform.training)
            self.assertFalse(overfit['val'].dataset.dataset.transform.training)
            self.assertTrue(all('original_mask' in sample for sample in overfit['val'].dataset))
            before = {p: p.read_bytes() for p in (root / 'splits').iterdir()}
            result = run_training(root, 'config.yaml', mode='smoke', samples=2, device='cpu', progress=False)
            run = result['run_dir']
            for filename in ('best.pt', 'last.pt', 'history.csv', 'best_val_metrics.csv', 'config.yaml',
                             'metadata.json', 'predictions_before.png', 'predictions_best.png',
                             'learning_curves.png', 'architecture.png'):
                self.assertTrue((run / filename).is_file(), filename)
            checkpoint = torch.load(run / 'best.pt', map_location='cpu', weights_only=True)
            model = UNet(channels=config['model']['channels'])
            model.load_state_dict(checkpoint['model_state_dict'])
            with torch.no_grad():
                self.assertEqual(tuple(model(torch.randn(1, 3, 16, 16)).shape), (1, 1, 16, 16))
            self.assertEqual(result['metadata']['status'], 'complete')
            self.assertFalse(result['metadata']['precision'] == 'cuda_fp16_amp')
            with (run / 'best_val_metrics.csv').open(encoding='utf-8-sig') as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(p.read_bytes() == content for p, content in before.items()))
            # Tied validation Dice must preserve the earlier best epoch; scheduler
            # and early stopping must both use validation, not training scores.
            config['training']['early_stopping_patience'] = 2
            config['training']['scheduler']['patience'] = 0
            (root / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
            scores = {'loss': 1.0, 'bce': 0.5, 'soft_dice_loss': 0.5,
                      'dice': 0.5, 'iou': 1 / 3, 'precision': 0.5, 'recall': 0.5, 'samples': 2}
            with patch('src.training.run_epoch', return_value=(scores, rows)) as epoch_runner:
                full = run_training(root, 'config.yaml', mode='full', epochs=10, device='cpu', progress=False)
            self.assertEqual(epoch_runner.call_count, 6)
            self.assertEqual(full['metadata']['epochs_completed'], 3)
            self.assertEqual(full['metadata']['best_epoch'], 1)
            self.assertLess(full['history'][2]['learning_rate'], full['history'][0]['learning_rate'])
            best = torch.load(full['run_dir'] / 'best.pt', map_location='cpu', weights_only=True)
            last = torch.load(full['run_dir'] / 'last.pt', map_location='cpu', weights_only=True)
            self.assertEqual(best['epoch'], 1)
            self.assertEqual(last['epoch'], 3)
