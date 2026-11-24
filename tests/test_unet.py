"""Check model/data interface and gradients; no dataset or training run needed."""

import unittest
from pathlib import Path

import torch
import yaml
from torch import nn

from src.models.unet import UNet


class UNetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_project_config_produces_full_resolution_logits(self):
        root = Path(__file__).resolve().parents[1]
        config = yaml.safe_load((root / 'configs/unet.yaml').read_text(encoding='utf-8'))
        model = UNet(channels=config['model']['channels'],
                     output_channels=config['model']['output_channels']).eval()
        height, width = config['data']['image_size']
        with torch.no_grad():
            logits = model(torch.randn(1, 3, height, width))
        self.assertEqual(tuple(logits.shape), (1, 1, height, width))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertIsInstance(model.head, nn.Conv2d)

    def test_bce_backward_reaches_encoder_bottleneck_and_decoder(self):
        model = UNet(channels=(2, 4, 8, 16, 32))
        image = torch.randn(2, 3, 33, 49, requires_grad=True)
        logits = model(image)
        self.assertEqual(tuple(logits.shape), (2, 1, 33, 49))
        nn.BCEWithLogitsLoss()(logits, torch.randint(0, 2, logits.shape).float()).backward()
        self.assertIsNotNone(image.grad)
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_invalid_configuration_and_input_fail_clearly(self):
        for options in ({'channels': (2, 4)}, {'output_channels': 0},
                        {'channels': (2, 4, 8, 16, 0)}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                UNet(**options)
        model = UNet(channels=(2, 4, 8, 16, 32))
        for shape in ((3, 32, 32), (1, 1, 32, 32), (1, 3, 15, 32)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(shape))


if __name__ == '__main__':
    unittest.main()
