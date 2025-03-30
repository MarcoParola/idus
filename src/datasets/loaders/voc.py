from typing import Tuple
import torch
from torch import Tensor
from torchvision.datasets import VOCDetection
import numpy as np

from src.datasets.dataset import DetectionDataset


def load_voc_dataset(root="./data", year="2012", resize=224, transform=None):
    """
    Load VOC detection dataset.

    Args:
        root: Root directory for the datasets
        year: VOC dataset year
        resize: Image resize dimension
        transform: Transforms to apply to the images

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, num_classes)
    """

    # Create train, validation and test datasets
    train_dataset = VOCDetection(root=root, year=year, image_set="train", download=True)
    val_dataset = VOCDetection(root=root, year=year, image_set="val", download=True)

    train = VOCDataset(train_dataset, resize=resize, transform=transform)
    val = VOCDataset(val_dataset, resize=resize, transform=transform)
    test = VOCDataset(val_dataset, resize=resize, transform=transform)

    return train, val, test


class VOCDataset(DetectionDataset):
    def __init__(self, torchvision_dataset, resize=224, transform=None):
        super().__init__()
        self.dataset = torchvision_dataset
        self.resize = resize
        self.transforms = transform

        # VOC class names (with background as 0)
        self.classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

        # Filter dataset to only include samples with valid annotations
        self.valid_indices = []
        for idx in range(len(self.dataset)):
            _, target = self.dataset[idx]
            if self._has_valid_annotations(target):
                self.valid_indices.append(idx)

        print(f"Dataset size after filtering empty annotations: {len(self.valid_indices)}")

    def _has_valid_annotations(self, target):
        """Check if the sample has valid annotations."""
        annotations = target['annotation']['object']

        if not isinstance(annotations, list):
            annotations = [annotations]

        for obj in annotations:
            class_name = obj['name']
            class_idx = self.classes.index(class_name) if class_name in self.classes else -1

            # Skip if class is not recognized or is background
            if class_idx > 0:  # Skip background (0)
                return True

        return False

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        # Use the filtered index
        actual_idx = self.valid_indices[idx]
        image, target = self.dataset[actual_idx]

        # Get original image dimensions
        width, height = image.size

        annotations = self.loadAnnotations(target, width / self.resize, height / self.resize)

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(0, 4, dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
            }
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64),
            }

        if self.transforms is not None:
            image = self.transforms(image)
            # Scale boxes if image was resized
            targets['boxes'] = targets['boxes'] / self.resize

        return image, targets

    def loadAnnotations(self, target, imgWidth: float, imgHeight: float) -> np.ndarray:
        ans = []
        objects = target['annotation']['object']

        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            class_name = obj['name']

            # Get class index
            if class_name not in self.classes:
                continue

            cat = self.classes.index(class_name)

            # Skip if class is background
            if cat <= 0:
                continue

            # Get bounding box coordinates
            bbox = obj['bndbox']
            x1 = float(bbox['xmin']) / imgWidth
            y1 = float(bbox['ymin']) / imgHeight
            x2 = float(bbox['xmax']) / imgWidth
            y2 = float(bbox['ymax']) / imgHeight

            ans.append([x1, y1, x2, y2, cat])

        return np.asarray(ans)