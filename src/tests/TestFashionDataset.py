import unittest
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import tempfile
import os
from collections import Counter

from src.datasets.fashion import load_fashion_dataset


class TestFashionDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up the datasets and transformations once for all tests
        cls.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
        ])
        cls.train, cls.val, cls.test = load_fashion_dataset(resize=128, transform=cls.transform)

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
        # Test dataset lengths and split proportions
        total_len = len(self.train) + len(self.val) + len(self.test)
        self.assertGreater(len(self.train), 0, "Train dataset should not be empty")
        self.assertGreater(len(self.val), 0, "Validation dataset should not be empty")
        self.assertGreater(len(self.test), 0, "Test dataset should not be empty")

        # Check approximate split ratios (considering we split first 7 classes for train/val)
        self.assertGreater(len(self.train) / total_len, 0.6, "Training set should be roughly 63% of data")
        self.assertLess(len(self.val) / total_len, 0.1, "Validation set should be roughly 7% of data")
        self.assertLessEqual(len(self.test) / total_len, 0.3,
                             "Test set should be roughly 30% of data")

    def test_sample_structure(self):
        # Test the structure of a single sample
        image, targets = self.train[0]

        # Check image properties
        self.assertIsInstance(image, torch.Tensor, "Image should be a tensor")
        self.assertEqual(image.shape, (3, 128, 128), "Image should have shape (3, 128, 128)")
        self.assertTrue((image >= 0).all() and (image <= 1).all(), "Image values should be normalized between 0 and 1")

        # Check target structure
        self.assertIn('boxes', targets, "Targets should contain 'boxes'")
        self.assertIn('labels', targets, "Targets should contain 'labels'")
        self.assertEqual(len(targets['boxes']), 1, "Should have exactly one box per image")
        self.assertEqual(len(targets['labels']), 1, "Should have exactly one label per image")

    def test_class_distribution(self):
        # Test class distribution
        self.assertEqual(len(set(self.class_distribution.keys())), 7,
                         "Training set should have exactly 7 classes")

        # Verify class ranges
        train_classes = set(self.class_distribution.keys())
        self.assertTrue(all(c < 7 for c in train_classes),
                        "Training classes should all be less than 7")

        # Test distribution in test set using proper iteration
        test_labels = []
        for i in range(len(self.test)):
            _, target = self.test[i]
            test_labels.append(target['labels'].item())
        self.assertTrue(all(label >= 7 for label in test_labels),
                        "Test set should only contain classes 7-9")

    def test_bounding_boxes(self):
        # Test bounding box properties
        for dataset in [self.train, self.val, self.test]:
            for i in range(len(dataset)):
                _, targets = dataset[i]
                boxes = targets['boxes']

                # Check box format
                self.assertEqual(boxes.shape[1], 4, "Boxes should have 4 coordinates")
                self.assertTrue(torch.all(boxes >= 0).item(), "Box coordinates should be non-negative")
                self.assertTrue(torch.all(boxes <= 1).item(), "Box coordinates should be normalized (<=1)")

                # For Fashion-MNIST, boxes should be [0,0,1,1] as we use the whole image
                self.assertTrue(torch.allclose(boxes, torch.tensor([[0., 0., 1., 1.]]), atol=1e-6),
                                "Boxes should cover the whole image")

    def test_visualization(self):
        # Test visualization capabilities
        for i in range(2):
            image, targets = self.train[i]
            image = ToPILImage()(image)
            image = np.array(image)

            plt.figure(figsize=(6, 6))
            plt.imshow(image)

            # Draw bounding boxes and labels
            for bbox, label in zip(targets['boxes'], targets['labels']):
                x, y, w, h = bbox * 128  # Scale back to image size
                plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], color='r', linewidth=2)
                plt.text(x, y - 2, f"Class {label.item()}", color='white', fontsize=8, backgroundcolor='red')

            # Save plot to temporary file
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = os.path.join(temp_dir, f"fashion_test_{i}.png")
                plt.savefig(temp_file)
                self.assertTrue(os.path.exists(temp_file), "Visualization file should be created")
            plt.close()

    def test_transforms(self):
        # Test that transforms are properly applied
        original_transform = self.transform
        special_transform = T.Compose([
            T.Resize((64, 64)),  # Different size to test resize
            T.ToTensor(),
        ])

        # Create new dataset with different transform
        _, _, test_transformed = load_fashion_dataset(resize=64, transform=special_transform)

        # Test original size
        image_orig, _ = self.test[0]
        self.assertEqual(image_orig.shape, (3, 128, 128), "Original image should be 128x128")

        # Test transformed size
        image_trans, _ = test_transformed[0]
        self.assertEqual(image_trans.shape, (3, 64, 64), "Transformed image should be 64x64")


if __name__ == "__main__":
    unittest.main()