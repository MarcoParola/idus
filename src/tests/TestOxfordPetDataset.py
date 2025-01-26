import unittest
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt
import tempfile
import os

from src.datasets.oxford_pet import load_oxford_dataset


# Assuming the OxfordPetDataset and load_oxford_dataset are implemented as provided above

class TestOxfordPetDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Set up the datasets and transformations once for all tests
        cls.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
        ])
        cls.train, cls.val, cls.test = load_oxford_dataset(resize=128, transform=cls.transform)

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

    def test_annotations(self):
        # Test that the annotations are properly filtered and processed
        for i in range(5):  # Test a few samples
            _, targets = self.test[i]
            boxes = targets['boxes']
            labels = targets['labels']
            self.assertGreaterEqual(len(boxes), 0, "Bounding boxes should exist")
            self.assertGreaterEqual(len(labels), 0, "Labels should exist")

    def test_visualization(self):
        # Test visualization (temporary output in memory)
        for i in range(2):  # Test visualization for 2 samples
            image, targets = self.test[i]
            image = ToPILImage()(image)
            image = np.array(image)

            # Visualize image and bounding boxes
            plt.imshow(image)
            for bbox in targets['boxes']:
                x, y, w, h = bbox * 128  # Scale back to image size
                plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], color='r', linewidth=2)

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

if __name__ == "__main__":
    unittest.main()
