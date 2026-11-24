"""Data/metric contract tests with synthetic masks; no training or downloads."""

import copy
import random
import unittest

import torch
from PIL import Image

from test_data_preparation import temporary_workspace
from src.datasets.dataset import PolypDataset, collate_samples, create_dataloaders
from src.datasets.transforms import SegmentationTransform, binarize_mask, pil_to_tensor, letterbox_geometry
from src.metrics import binary_metrics, metrics_at_original_size, restore_original_logits
from src.utils import load_config


class TransformTests(unittest.TestCase):
    def test_letterbox_centers_landscape_portrait_and_odd_padding(self):
        for source_hw, expected_hw, expected_pad in (
            ((400, 800), (128, 256), (0, 64, 0, 64)),
            ((800, 400), (256, 128), (64, 0, 64, 0)),
            ((3, 5), (154, 256), (0, 51, 0, 51)),
        ):
            with self.subTest(source=source_hw):
                geometry = letterbox_geometry(source_hw, (256, 256))
                self.assertEqual(geometry["resized_size"], expected_hw)
                self.assertEqual(geometry["padding"], expected_pad)
        geometry = letterbox_geometry((3, 5), (8, 8))
        self.assertEqual(geometry["padding"], (0, 1, 0, 2))
        for source_hw in ((3, 5), (5, 3), (1, 10000), (10000, 1), (5, 5)):
            result = SegmentationTransform(8).apply(
                Image.new("RGB", source_hw[::-1], "white"), Image.new("L", source_hw[::-1], 255))
            height, width = result["geometry"]["resized_size"]
            left, top, right, bottom = result["geometry"]["padding"]
            self.assertLessEqual(abs(left - right), 1)
            self.assertLessEqual(abs(top - bottom), 1)
            self.assertEqual(result["valid_mask"].sum().item(), height * width)
            self.assertTrue(torch.equal(result["mask"].bool(), result["valid_mask"]))
            restored_rgb = SegmentationTransform(8).denormalize(result["image"])
            self.assertTrue(torch.allclose(restored_rgb[:, ~result["valid_mask"][0]],
                                           torch.zeros_like(restored_rgb[:, ~result["valid_mask"][0]]), atol=1e-6))

    def test_augmented_nonsquare_content_stays_aligned(self):
        # Resize is identity for the content, padding only: exact pixel matching.
        mask = Image.new("L", (6, 3)); mask.putpixel((1, 0), 255)
        image = Image.merge("RGB", (mask, mask, mask))
        transform = SegmentationTransform(6, training=True, mean=(0, 0, 0), std=(1, 1, 1))
        for seed in range(8):
            random.seed(seed)
            result = transform.apply(image, mask)
            self.assertTrue(torch.equal(result["image"][0], result["mask"][0]))
            self.assertEqual(result["mask"].sum().item(), 1)
            self.assertEqual(result["valid_mask"].sum().item(), 18)

    def test_geometry_is_paired_and_eval_is_deterministic(self):
        mask = Image.new("L", (4, 4)); mask.putpixel((0, 1), 255); mask.putpixel((2, 3), 255)
        image = Image.merge("RGB", (mask, mask, mask))
        train = SegmentationTransform(4, training=True, mean=(0, 0, 0), std=(1, 1, 1))
        for seed in range(8):
            random.seed(seed)
            x, y = train(image, mask)
            self.assertTrue(torch.equal(x[0], y[0]))
            self.assertEqual(y.sum().item(), 2)
        evaluation = SegmentationTransform((3, 5), training=False)
        first = evaluation(image, mask)
        random.seed(999)
        second = evaluation(image, mask)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertEqual(first[1].shape, (1, 3, 5))
        self.assertTrue(set(first[1].unique().tolist()) <= {0, 1})

    def test_mask_threshold_and_rgb_normalization(self):
        mask = Image.new("L", (4, 1)); mask.putdata([0, 127, 128, 255])
        self.assertEqual(pil_to_tensor(binarize_mask(mask)).flatten().tolist(), [0, 0, 1, 1])
        image = Image.new("RGB", (4, 1), (255, 0, 127))
        transform = SegmentationTransform((1, 4))
        normalized, _ = transform(image, mask)
        recovered = transform.denormalize(normalized)
        self.assertTrue(torch.allclose(recovered, pil_to_tensor(image), atol=1e-6))


class DatasetTests(unittest.TestCase):
    def setUp(self):
        self.workspace = temporary_workspace()
        self.root = self.workspace.__enter__()
        (self.root / "src/datasets").mkdir(parents=True)
        (self.root / "README.md").touch()
        (self.root / "splits").mkdir()
        for kind in ("images", "masks"):
            (self.root / "data/Kvasir-SEG" / kind).mkdir(parents=True)
        for name, size in (("a", (5, 3)), ("b", (4, 6))):
            Image.new("RGB", size, "red").save(self.root / f"data/Kvasir-SEG/images/{name}.jpg")
            Image.new("L", size, 255).save(self.root / f"data/Kvasir-SEG/masks/{name}.jpg")
        (self.root / "splits/kvasir_train.txt").write_text("a\nb\n", encoding="utf-8")
        (self.root / "splits/kvasir_val.txt").write_text("b\n", encoding="utf-8")

    def tearDown(self):
        self.workspace.__exit__(None, None, None)

    def test_original_masks_collate_without_resizing(self):
        (self.root / "splits/kvasir_val.txt").write_text("a\nb\n", encoding="utf-8")
        dataset = PolypDataset(self.root, "val", SegmentationTransform(8), return_original_mask=True)
        batch = collate_samples([dataset[0], dataset[1]])
        self.assertEqual(batch["image"].shape, (2, 3, 8, 8))
        self.assertEqual(batch["original_size"], [(3, 5), (6, 4)])
        self.assertEqual(batch["original_mask"][0].shape, (1, 3, 5))
        self.assertEqual(batch["original_mask"][1].shape, (1, 6, 4))
        self.assertEqual(batch["valid_mask"].shape, (2, 1, 8, 8))
        self.assertEqual(batch["geometry"][0]["padding"], (0, 1, 0, 2))
        self.assertEqual(batch["geometry"][1]["padding"], (1, 0, 2, 0))
        scores = metrics_at_original_size(batch["mask"] * 40 - 20,
                                          batch["original_mask"], batch["geometry"])
        self.assertTrue(torch.equal(scores["dice"], torch.ones(2)))

    def test_reject_duplicate_missing_and_augmentation_on_eval(self):
        path = self.root / "splits/kvasir_train.txt"
        for text in ("a\na\n", "nonexistent\n", "../a\n", "a.png\n"):
            path.write_text(text, encoding="utf-8")
            with self.subTest(text=text), self.assertRaises((ValueError, FileNotFoundError)):
                PolypDataset(self.root)
        with self.assertRaises(ValueError):
            PolypDataset(self.root, "val", SegmentationTransform(training=True))

    def test_loader_rejects_overlap_and_has_no_test_by_default(self):
        from pathlib import Path
        config = load_config(Path(__file__).resolve().parents[1] / "configs/unet.yaml")
        with self.assertRaisesRegex(ValueError, "overlap"):
            create_dataloaders(self.root, config)
        (self.root / "splits/kvasir_train.txt").write_text("a\n", encoding="utf-8")
        config = copy.deepcopy(config)
        config["loader"]["batch_size"] = 1
        config["data"]["image_size"] = [8, 8]
        first = create_dataloaders(self.root, config)
        first_batch = next(iter(first["train"]))
        second = create_dataloaders(self.root, config)
        second_batch = next(iter(second["train"]))
        self.assertEqual(set(first), {"train", "val"})
        self.assertTrue(torch.equal(first_batch["image"], second_batch["image"]))


class MetricTests(unittest.TestCase):
    def test_known_overlap_empty_and_image_mean(self):
        target = torch.tensor([[[1., 1., 0.]], [[0., 0., 0.]]])
        predicted = torch.tensor([[[1., 0., 1.]], [[0., 0., 0.]]])
        scores = binary_metrics(predicted, target, from_logits=False)
        self.assertTrue(torch.allclose(scores["dice"], torch.tensor([0.5, 1.])))
        self.assertAlmostEqual(scores["iou"][0].item(), 1 / 3, places=6)
        self.assertEqual(scores["dice"].mean().item(), 0.75)
        self.assertEqual(scores["precision"][1].item(), 0)
        one_empty = binary_metrics(torch.zeros_like(target[:1]), target[:1], from_logits=False)
        self.assertEqual(one_empty["dice"].item(), 0)
        disjoint = binary_metrics(1-target[:1], target[:1], from_logits=False)
        self.assertEqual(disjoint["iou"].item(), 0)

    def test_resize_logits_before_sigmoid_and_keep_original_dimensions(self):
        logits = torch.tensor([[[[-100., 2.]]]])
        original = torch.tensor([[[0., 0., 0., 1.]]])
        geometry = {**letterbox_geometry((1, 4), (1, 2)), "original_size": (1, 4), "augmented": False}
        scores = metrics_at_original_size(logits, [original], [geometry])
        # Interpolating probabilities instead would label the third pixel positive.
        self.assertEqual(scores["dice"].item(), 1)
        self.assertEqual(scores["iou"].item(), 1)

    def test_padded_foreground_never_counts_at_original_resolution(self):
        for original_size in ((3, 5), (5, 3)):
            transformed = SegmentationTransform(8).apply(
                Image.new("RGB", original_size[::-1]), Image.new("L", original_size[::-1]))
            valid = transformed["valid_mask"].unsqueeze(0)
            # All false positives are in padding. Real content is background.
            logits = torch.where(valid, -10., 100.)
            original_mask = torch.zeros((1, *original_size))
            restored = restore_original_logits(logits, [transformed["geometry"]])[0]
            self.assertEqual(tuple(restored.shape), (1, 1, *original_size))
            self.assertTrue(torch.allclose(restored, torch.full_like(restored, -10.)))
            scores = metrics_at_original_size(logits, [original_mask], [transformed["geometry"]])
            self.assertEqual(scores["dice"].item(), 1)
            network_scores = binary_metrics(logits, torch.zeros_like(logits), valid_mask=valid)
            self.assertEqual(network_scores["dice"].item(), 1)
            self.assertEqual(binary_metrics(logits, torch.zeros_like(logits))["dice"].item(), 0)

    def test_restore_supports_low_resolution_decoder_and_rejects_wrong_metadata(self):
        geometry = {**letterbox_geometry((3, 5), (8, 8)), "original_size": (3, 5), "augmented": False}
        restored = restore_original_logits(torch.full((1, 1, 2, 2), -5.), [geometry])[0]
        self.assertEqual(tuple(restored.shape), (1, 1, 3, 5))
        for invalid in ({**geometry, "augmented": True}, {**geometry, "padding": (0, 0, 0, 0)}):
            with self.assertRaises(ValueError):
                restore_original_logits(torch.zeros(1, 1, 8, 8), [invalid])
        with self.assertRaises(ValueError):
            binary_metrics(torch.zeros(1, 1, 8, 8), torch.zeros(1, 1, 8, 8),
                           valid_mask=torch.zeros(1, 1, 8, 8))

    def test_reject_soft_target_and_nonfinite_prediction(self):
        with self.assertRaises(ValueError):
            binary_metrics(torch.zeros(1, 2, 2), torch.full((1, 2, 2), 0.5))
        with self.assertRaises(ValueError):
            binary_metrics(torch.full((1, 2, 2), float("nan")), torch.zeros(1, 2, 2))


if __name__ == "__main__":
    unittest.main()
