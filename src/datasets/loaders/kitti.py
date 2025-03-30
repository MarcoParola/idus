from typing import Tuple
import torch
from torch import Tensor
import numpy as np
from torchvision.datasets import Kitti
from torch.utils.data import random_split

from src.datasets.dataset import DetectionDataset


def load_kitti_dataset(root="./data", resize=224, transform=None):
    """
    Load KITTI detection dataset, dividing training data into three equal parts
    for train, validation, and test sets.

    Args:
        root: Root directory for the KITTI dataset
        resize: Image resize dimension
        transform: Transforms to apply to the images

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset)
    """
    # Create train dataset using torchvision Kitti
    # Ignore the original test dataset as it has no annotations
    train_dataset = Kitti(root=root, train=True, download=True)

    print(f"Original train dataset size: {len(train_dataset)}")

    # Wrap with custom dataset class
    full_train = KITTIDataset(train_dataset, resize=resize, transform=transform, name="train")

    # Divide the training set into three equal parts
    total_size = len(full_train)
    train_size = int(total_size / 3)
    val_size = int(total_size / 3)
    test_size = total_size - train_size - val_size  # Account for any rounding

    # Create the splits with fixed random seed for reproducibility
    train, val, test = random_split(
        full_train,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(f"Final training samples: {len(train)}")
    print(f"Final validation samples: {len(val)}")
    print(f"Final test samples: {len(test)}")

    return train, val, test


class KITTIDataset(DetectionDataset):
    def __init__(self, torchvision_dataset, resize=224, transform=None, name="unnamed"):
        super().__init__()
        self.dataset = torchvision_dataset
        self.resize = resize
        self.transforms = transform
        self.name = name  # For debugging

        # KITTI class names (with background as 0)
        self.classes = [
            'background', 'car', 'van', 'truck', 'pedestrian',
            'person_sitting', 'cyclist', 'tram', 'misc'
        ]

        # Debugging counters
        self.valid_count = 0
        self.invalid_count = 0
        self.empty_annotations = 0
        self.no_valid_class = 0

        # Filter dataset to only include samples with valid annotations
        self.valid_indices = []
        for idx in range(len(self.dataset)):
            _, target = self.dataset[idx]
            if self._has_valid_annotations(target):
                self.valid_indices.append(idx)
                self.valid_count += 1
            else:
                self.invalid_count += 1

        print(f"{self.name} dataset stats:")
        print(f"  - Total samples: {len(self.dataset)}")
        print(f"  - Valid samples: {self.valid_count}")
        print(f"  - Invalid samples: {self.invalid_count}")
        print(f"  - Empty annotations: {self.empty_annotations}")
        print(f"  - No valid class: {self.no_valid_class}")
        print(f"  - Final dataset size: {len(self.valid_indices)}")

        # If no valid samples, adjust the filtering criteria in debug mode
        if len(self.valid_indices) == 0 and self.name == "test":
            print("WARNING: No valid test samples found with current criteria. Accepting all samples.")
            self.valid_indices = list(range(len(self.dataset)))

    def _has_valid_annotations(self, target):
        """Check if the sample has valid annotations."""
        # Check if there are any objects at all
        if not target or len(target) == 0:
            self.empty_annotations += 1
            return False

        has_valid_class = False
        for obj in target:
            class_name = obj['type'].lower()
            # Handle case differences or formatting issues
            for c in self.classes[1:]:  # Skip background
                if c in class_name or class_name in c:
                    has_valid_class = True
                    break

        if not has_valid_class:
            self.no_valid_class += 1

        return has_valid_class

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

        for obj in target:
            class_name = obj['type'].lower()

            # More flexible class matching
            cat = -1
            for i, c in enumerate(self.classes):
                if c in class_name or class_name in c:
                    cat = i
                    break

            # Skip if class is background
            if cat <= 0:
                continue

            # KITTI bbox is [x_min, y_min, x_max, y_max]
            bbox = obj['bbox']
            x1 = float(bbox[0]) / imgWidth
            y1 = float(bbox[1]) / imgHeight
            x2 = float(bbox[2]) / imgWidth
            y2 = float(bbox[3]) / imgHeight

            ans.append([x1, y1, x2, y2, cat])

        return np.asarray(ans)