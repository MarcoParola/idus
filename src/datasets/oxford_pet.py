from datasets import load_dataset
from typing import Tuple
from torch import Tensor
import torch
from PIL import Image
import numpy as np

from src.datasets.dataset import DetectionDataset


def load_oxford_dataset(resize=224, transform=None):
    """
    Load Oxford-IIIT Pet dataset.

    Args:
        resize: Image resize dimension
        transform: Transforms to apply to the images

    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, num_classes)
    """
    dataset = load_dataset("visual-layer/oxford-iiit-pet-vl-enriched")
    print(dataset)

    # Get the full class list
    all_classes = dataset['train'].unique('label_breed')
    print(f"Total classes: {len(all_classes)}")

    # Create validation set from train
    train_valid_split = dataset['train'].train_test_split(test_size=0.1)

    # Access the new train and validation sets
    train = train_valid_split['train']
    val = train_valid_split['test']
    test = dataset['test']

    # Create datasets
    train = OxfordPetDataset(train, resize=resize, transform=transform)
    val = OxfordPetDataset(val, resize=resize, transform=transform)
    test = OxfordPetDataset(test, resize=resize, transform=transform)

    return train, val, test


class OxfordPetDataset(DetectionDataset):
    def __init__(self, hf_dataset, resize=224, transform=None):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.resize = resize
        self.transforms = transform

        # Get full class list
        self.classes = hf_dataset.unique('label_breed')
        print(f"Class count: {len(self.classes)}")

        # Filter dataset to remove entries without valid bounding boxes
        self.hf_dataset = self.hf_dataset.filter(self._has_valid_annotations)
        print(f"Dataset size after filtering invalid annotations: {len(self.hf_dataset)}")

    def _has_valid_annotations(self, sample):
        """Check if the sample has valid annotations."""
        if sample['label_bbox_enriched'] is None:
            return False

        # Check if the bbox contains cat or dog annotations
        bbox_labels = [annotation['label'] for annotation in sample['label_bbox_enriched']]
        has_cat_or_dog = 'cat' in bbox_labels or 'dog' in bbox_labels

        return has_cat_or_dog

    def __len__(self) -> int:
        return self.hf_dataset.num_rows

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        sample = self.hf_dataset.__getitem__(idx)
        image = sample['image']
        image = np.array(image)

        annotations = self.loadAnnotations(sample, image.shape[1] / self.resize, image.shape[0] / self.resize)

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
            image = Image.fromarray(np.array(image))
            image = self.transforms(image)
            targets['boxes'] = targets['boxes'] / self.resize

        return image, targets

    def loadAnnotations(self, sample, imgWidth: float, imgHeight: float) -> np.ndarray:
        ans = []
        for annotation in sample['label_bbox_enriched']:
            if annotation['label'] not in ['cat', 'dog']:
                continue

            # Get class index directly
            cat_idx = self.classes.index(sample['label_breed'])

            bbox = annotation['bbox']
            bbox = [bbox[0] / imgWidth, bbox[1] / imgHeight, bbox[2] / imgWidth, bbox[3] / imgHeight]
            ans.append(bbox + [cat_idx])

        return np.asarray(ans)