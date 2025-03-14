from datasets import load_dataset
from typing import Tuple
from torch import Tensor
import torch
from PIL import Image
import numpy as np

from src.datasets.dataset import DetectionDataset


def load_oxford_dataset(resize=224, transform=None, exclude_classes=None):
    if exclude_classes is None:
        exclude_classes = []

    dataset = load_dataset("visual-layer/oxford-iiit-pet-vl-enriched")
    print(dataset)

    # Get the full class list before any filtering
    all_classes = dataset['train'].unique('label_breed')

    # Create a list of classes excluding the unwanted ones
    kept_classes = [cls for i, cls in enumerate(all_classes) if i not in exclude_classes]

    print(f"Original classes: {len(all_classes)}")
    print(f"Excluding classes: {[all_classes[idx] for idx in exclude_classes if idx < len(all_classes)]}")
    print(f"Remaining classes: {len(kept_classes)}")

    # Create validation set from train
    train_valid_split = dataset['train'].train_test_split(test_size=0.1)

    # Access the new train and validation sets
    train = train_valid_split['train']
    val = train_valid_split['test']
    test = dataset['test']

    # Create all datasets with the same exclusion list and original class list
    train = OxfordPetDataset(train, resize=resize, transform=transform, exclude_classes=exclude_classes)
    val = OxfordPetDataset(val, resize=resize, transform=transform, exclude_classes=exclude_classes)
    test = OxfordPetDataset(test, resize=resize, transform=transform, exclude_classes=exclude_classes)

    num_classes = len(kept_classes)

    return train, val, test, num_classes


class OxfordPetDataset(DetectionDataset):
    def __init__(self, hf_dataset, resize=224, transform=None, exclude_classes=None):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.resize = resize
        self.transforms = transform
        self.exclude_classes = exclude_classes or []

        # Get full class list
        all_classes = hf_dataset.unique('label_breed')

        # Create a remapped class list excluding unwanted classes
        self.original_classes = all_classes
        self.classes = [cls for i, cls in enumerate(all_classes) if i not in self.exclude_classes]

        # Create mapping from original indices to new indices
        self.class_mapping = {}
        new_idx = 0
        for orig_idx, cls in enumerate(all_classes):
            if orig_idx not in self.exclude_classes:
                self.class_mapping[orig_idx] = new_idx
                new_idx += 1

        print(f"Original class count: {len(all_classes)}")
        print(f"After excluding {len(self.exclude_classes)} classes: {len(self.classes)} classes")

        # Filter dataset
        def has_valid_annotations(sample):
            breed_idx = self.original_classes.index(sample['label_breed'])
            if breed_idx in self.exclude_classes:
                return False

            if sample['label_bbox_enriched'] is None:
                return False

            bbox_labels = [annotation['label'] for annotation in sample['label_bbox_enriched']]
            has_cat_or_dog = 'cat' in bbox_labels or 'dog' in bbox_labels

            return has_cat_or_dog

        self.hf_dataset = self.hf_dataset.filter(has_valid_annotations)
        print(f"Dataset size after filtering: {len(self.hf_dataset)}")

    def __len__(self) -> int:
        return self.hf_dataset.num_rows

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        sample = self.hf_dataset.__getitem__(idx)
        image = sample['image']
        image = np.array(image)

        annotations = self.loadAnnotations(sample, image.shape[1]/self.resize, image.shape[0]/self.resize)

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor(len(self.classes), dtype=torch.int64),}
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64),}

        if self.transforms is not None:
            image = Image.fromarray(np.array(image))
            image = self.transforms(image)
            targets['boxes'] = targets['boxes'] / self.resize

        return image, targets

    def loadAnnotations(self, sample, imgWidth: int, imgHeight: int) -> np.ndarray:
        ans = []
        for annotation in sample['label_bbox_enriched']:
            if annotation['label'] not in ['cat', 'dog']:
                continue

            # Get original class index
            orig_cat = self.original_classes.index(sample['label_breed'])

            # Map to new class index
            cat = self.class_mapping[orig_cat]  # This remaps to a sequential index

            bbox = annotation['bbox']
            bbox = [bbox[0] / imgWidth, bbox[1] / imgHeight, bbox[2] / imgWidth, bbox[3] / imgHeight]
            ans.append(bbox + [cat])

        return np.asarray(ans)

    
