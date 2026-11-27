import unittest
import torch
from src.models.pretrained_unet import PretrainedUNet


class PretrainedTests(unittest.TestCase):
    def test_random_variant_shape_gradients_and_deterministic_head(self):
        old = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            model = PretrainedUNet(None, init_seed=42)
            other = PretrainedUNet(None, init_seed=42)
            for key in ('decoder_initial_sha256', 'head_initial_sha256'):
                self.assertEqual(model.provenance[key], other.provenance[key])
            del other
            output = model(torch.randn(2, 3, 64, 64))
            self.assertEqual(tuple(output.shape), (2, 1, 64, 64))
            output.square().mean().backward()
            for parameter in model.parameters():
                self.assertTrue(parameter.requires_grad)
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        finally:
            torch.set_num_threads(old)
