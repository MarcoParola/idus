from typing import Tuple
import torch
from torch import Tensor
from torchvision.datasets import VOCDetection
import numpy as np

from src.datasets.dataset import DetectionDataset


def load_voc_dataset(root="./data", year="2012", resize=224, transform=None, exclude_classes=None):
    if exclude_classes is None:
        exclude_classes = []

    # VOC class names
    all_classes = [
        'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
        'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
        'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    ]

    # Create a list of classes excluding the unwanted ones (background is already excluded in VOC)
    kept_classes = [cls for i, cls in enumerate(all_classes[1:], 1) if i not in exclude_classes]

    print(f"Original classes: {len(all_classes) - 1}")  # Subtract 1 for background
    print(f"Excluding classes: {[all_classes[idx] for idx in exclude_classes if idx < len(all_classes)]}")
    print(f"Remaining classes: {len(kept_classes)}")

    # Create train, validation and test datasets
    train_dataset = VOCDetection(root=root, year=year, image_set="train", download=True)
    val_dataset = VOCDetection(root=root, year=year, image_set="val", download=True)

    train = VOCDataset(train_dataset, resize=resize, transform=transform, exclude_classes=exclude_classes)
    val = VOCDataset(val_dataset, resize=resize, transform=transform, exclude_classes=exclude_classes)

    test = VOCDataset(val_dataset, resize=resize, transform=transform, exclude_classes=exclude_classes)

    num_classes = len(kept_classes)

    return train, val, test, num_classes


class VOCDataset(DetectionDataset):
    def __init__(self, torchvision_dataset, resize=224, transform=None, exclude_classes=None):
        super().__init__()
        self.dataset = torchvision_dataset
        self.resize = resize
        self.transforms = transform
        self.exclude_classes = exclude_classes or []

        # VOC class names (including background as 0)
        self.original_classes = [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
            'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

        # Create a remapped class list excluding unwanted classes and background
        self.classes = [cls for i, cls in enumerate(self.original_classes[1:], 1) if i not in self.exclude_classes]

        # Create mapping from original indices to new indices
        self.class_mapping = {}
        new_idx = 0
        for orig_idx in range(1, len(self.original_classes)):  # Skip background (0)
            if orig_idx not in self.exclude_classes:
                self.class_mapping[orig_idx] = new_idx
                new_idx += 1

        print(f"Original class count: {len(self.original_classes) - 1}")  # Subtract 1 for background
        print(f"After excluding {len(self.exclude_classes)} classes: {len(self.classes)} classes")

        # Filter dataset to only include samples with valid annotations
        self.valid_indices = []
        for idx in range(len(self.dataset)):
            _, target = self.dataset[idx]
            if self._has_valid_annotations(target):
                self.valid_indices.append(idx)

        print(f"Dataset size after filtering: {len(self.valid_indices)}")

    def _has_valid_annotations(self, target):
        """Check if the sample has valid annotations."""
        annotations = target['annotation']['object']

        if not isinstance(annotations, list):
            annotations = [annotations]

        for obj in annotations:
            class_name = obj['name']
            class_idx = self.original_classes.index(class_name) if class_name in self.original_classes else -1

            # Skip if class is excluded or not recognized
            if class_idx <= 0 or class_idx in self.exclude_classes:
                continue

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
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor(len(self.classes), dtype=torch.int64),
            }
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64),
            }

        if self.transforms is not None:
            image = self.transforms(image)
            targets['boxes'] = targets['boxes'] / self.resize

        return image, targets

    def loadAnnotations(self, target, imgWidth: float, imgHeight: float) -> np.ndarray:
        ans = []
        objects = target['annotation']['object']

        if not isinstance(objects, list):
            objects = [objects]

        for obj in objects:
            class_name = obj['name']

            # Get original class index
            if class_name not in self.original_classes:
                continue

            orig_cat = self.original_classes.index(class_name)

            # Skip if class is background or excluded
            if orig_cat <= 0 or orig_cat in self.exclude_classes:
                continue

            # Map to new class index
            cat = self.class_mapping[orig_cat]

            # Get bounding box coordinates
            bbox = obj['bndbox']
            x1 = float(bbox['xmin']) / imgWidth
            y1 = float(bbox['ymin']) / imgHeight
            x2 = float(bbox['xmax']) / imgWidth
            y2 = float(bbox['ymax']) / imgHeight

            ans.append([x1, y1, x2, y2, cat])

        return np.asarray(ans)