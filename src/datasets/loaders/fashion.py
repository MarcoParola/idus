from typing import Tuple
import torch
from torch import Tensor
import numpy as np
from PIL import Image
from torchvision.datasets import FashionMNIST
from src.datasets.dataset import DetectionDataset


def load_fashion_dataset(resize=128, transform=None):
    """Load the Fashion-MNIST dataset directly from torchvision"""
    # Download training data
    train_data = FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=None
    )

    # Download test data
    test_data = FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=None
    )

    # Combine data
    all_images = np.concatenate([
        np.array(train_data.data),
        np.array(test_data.data)
    ])
    all_labels = np.concatenate([
        np.array(train_data.targets),
        np.array(test_data.targets)
    ])

    # Split indices for train/val/test
    train_classes = list(range(7))
    test_classes = list(range(7, 10))

    train_indices = np.nonzero(np.in1d(all_labels, train_classes))[0]
    np.random.shuffle(train_indices)
    val_size = int(len(train_indices) * 0.1)

    # Create datasets
    train_dataset = FashionDataset(
        all_images[train_indices[val_size:]],
        all_labels[train_indices[val_size:]],
        resize=resize,
        transform=transform
    )

    val_dataset = FashionDataset(
        all_images[train_indices[:val_size]],
        all_labels[train_indices[:val_size]],
        resize=resize,
        transform=transform
    )

    test_indices = np.nonzero(np.in1d(all_labels, test_classes))[0]
    test_dataset = FashionDataset(
        all_images[test_indices],
        all_labels[test_indices],
        resize=resize,
        transform=transform
    )

    return train_dataset, val_dataset, test_dataset


class FashionDataset(DetectionDataset):
    """Fashion-MNIST Dataset adapted for object detection"""

    def __init__(self, images, labels, resize=128, transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.resize = resize
        self.transforms = transform

        # Class names for reference
        self.classes = [
            't_shirt_top', 'trouser', 'pullover', 'dress', 'coat',
            'sandal', 'shirt', 'sneaker', 'bag', 'ankle_boots'
        ]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        # Get single image and convert to PIL
        image = Image.fromarray(self.images[idx].astype(np.uint8))

        # Calculate bbox (entire image for single object)
        bbox = [0, 0, 1, 1]

        if self.transforms is not None:
            image = self.transforms(image)

        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        # Prepare target dict
        targets = {
            'boxes': torch.tensor([bbox], dtype=torch.float32),
            'labels': torch.tensor([self.labels[idx]], dtype=torch.int64)
        }

        return image, targets

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]