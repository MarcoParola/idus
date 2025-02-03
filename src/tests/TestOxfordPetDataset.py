import unittest
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import tempfile
import os
from collections import Counter

from src.datasets.oxford_pet import load_oxford_dataset


class TestOxfordPetDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up the datasets and transformations once for all tests
        cls.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
        ])
        cls.train, cls.val, cls.test = load_oxford_dataset(resize=128, transform=cls.transform)

        # Collect and print dataset statistics
        cls.class_distribution = cls._get_class_distribution()
        print("Class Distribution in Training Set:", cls.class_distribution)

        cls.unique_classes = sorted(cls.class_distribution.keys())
        print("Unique Classes in the Dataset:", cls.unique_classes)

    @classmethod
    def _get_class_distribution(cls):
        class_counter = Counter()
        box_counter = []
        example_targets = []

        # Get first 5 examples for inspection
        for idx in range(min(1000, len(cls.train))):
            _, target = cls.train[idx]
            if len(example_targets) < 5:
                example_targets.append(target)
            box_counter.append(len(target['boxes']))
            labels = target['labels']
            if isinstance(labels, torch.Tensor):
                class_counter.update(labels.tolist())
            else:
                class_counter.update(labels)

        print("\nDataset Analysis:")
        print(f"Number of unique classes: {len(set(class_counter.keys()))}")
        print(f"Class range: {min(class_counter.keys())} to {max(class_counter.keys())}")
        print(f"Average boxes per image: {np.mean(box_counter):.2f}")
        print("\nFirst 5 targets example:")
        for i, target in enumerate(example_targets):
            print(f"\nSample {i}:")
            print(f"Labels: {target['labels'].tolist()}")
            print(f"Number of boxes: {len(target['boxes'])}")
            print(f"Box coordinates (first box): {target['boxes'][0].tolist()}")

        return dict(class_counter)

    def test_dataset_length(self):
        # Test dataset lengths
        self.assertGreater(len(self.train), 0, "Train dataset should not be empty")
        self.assertGreater(len(self.val), 0, "Validation dataset should not be empty")
        self.assertGreater(len(self.test), 0, "Test dataset should not be empty")

    def test_sample_structure(self):
        # Test the structure of a single sample
        image, targets = self.test[0]
        self.assertIsInstance(image, torch.Tensor, "Image should be a tensor")
        self.assertIn('boxes', targets, "Targets should contain 'boxes'")
        self.assertIn('labels', targets, "Targets should contain 'labels'")
        self.assertEqual(image.shape, (3, 128, 128), "Image should have shape (3, 128, 128) after resizing")

    def test_annotations(self):
        # Test that the annotations are properly filtered and processed
        for i in range(5):  # Test a few samples
            _, targets = self.test[i]
            boxes = targets['boxes']
            labels = targets['labels']
            self.assertGreaterEqual(len(boxes), 0, "Bounding boxes should exist")
            self.assertGreaterEqual(len(labels), 0, "Labels should exist")
            self.assertTrue(torch.all(boxes >= 0).item(), "Bounding box coordinates should be non-negative")

    def test_class_distribution(self):
        # Ensure the class distribution is not empty and meaningful
        self.assertGreater(len(self.class_distribution), 0, "There should be at least one class in the dataset")
        for cls_id, count in self.class_distribution.items():
            self.assertGreater(count, 0, f"Class {cls_id} should have at least one instance")

    def test_visualization(self):
        # Test visualization with class labels (temporary output in memory)
        for i in range(2):  # Test visualization for 2 samples
            image, targets = self.test[i]
            image = ToPILImage()(image)
            image = np.array(image)

            plt.imshow(image)
            for bbox, label in zip(targets['boxes'], targets['labels']):
                x, y, w, h = bbox * 128  # Scale back to image size
                plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], color='r', linewidth=2)
                plt.text(x, y - 2, f"Class {label.item()}", color='white', fontsize=8, backgroundcolor='red')

            # Save plot to a temporary file
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, f"test_image_{i}.png")
                plt.savefig(temp_file)
                self.assertTrue(os.path.exists(temp_file), "Visualization file should be created")
            plt.close()

    def test_sample_transformations(self):
        # Test transformations on a sample
        image, _ = self.train[0]
        self.assertEqual(image.shape, (3, 128, 128), "Image should be resized to (3, 128, 128)")
        # Check pixel value range
        self.assertTrue((image >= 0).all() and (image <= 1).all(), "Image tensor values should be in [0, 1]")

    def test_bounding_box_integrity_after_transform(self):
        # Ensure bounding boxes are still within image bounds after transformation
        for i in range(5):
            image, targets = self.test[i]
            boxes = targets['boxes'] * 128  # Scale back to image size
            for bbox in boxes:
                x, y, w, h = bbox
                self.assertTrue(0 <= x <= 128, "Bounding box x-coordinate should be within image bounds")
                self.assertTrue(0 <= y <= 128, "Bounding box y-coordinate should be within image bounds")
                self.assertTrue(w >= 0, "Bounding box width should be non-negative")
                self.assertTrue(h >= 0, "Bounding box height should be non-negative")


if __name__ == "__main__":
    unittest.main()
