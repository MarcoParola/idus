from datasets import load_dataset
from typing import Tuple
from torch import Tensor
import torch
from PIL import Image
import numpy as np

from src.datasets.dataset import DetectionDataset


def load_oxford_superclass_dataset(resize=224, transform=None):
    """
    Load the Oxford-IIIT Pet Dataset with only cat and dog superclasses
    """
    dataset = load_dataset("visual-layer/oxford-iiit-pet-vl-enriched")
    print(dataset)
    # create validation set from train
    train_valid_split = dataset['train'].train_test_split(test_size=0.1)

    # Access the new train and validation sets
    train = train_valid_split['train']
    val = train_valid_split['test']
    test = dataset['test']
    train = OxfordPetSuperclassDataset(train, resize=resize, transform=transform)
    val = OxfordPetSuperclassDataset(val, resize=resize, transform=transform)
    test = OxfordPetSuperclassDataset(test, resize=resize, transform=transform)
    return train, val, test


class OxfordPetSuperclassDataset(DetectionDataset):
    """Oxford-IIIT Pet Dataset with only cat and dog superclasses
    Args:
        hf_dataset: Hugging Face Dataset object
        resize: Resize the image to this size
        transform: Optional transform to be applied on a sample.

        This class is similar to OxfordPetDataset but uses only cat and dog
        superclasses as labels instead of the 37 breed classes.
    """

    def __init__(self, hf_dataset, resize=224, transform=None):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.resize = resize
        self.transforms = transform
        # Only two superclasses: cat and dog
        self.classes = ['cat', 'dog']

        # Create a mapping from breed to superclass
        # In Oxford-IIIT dataset, classes 1-12 are dogs, 13-37 are cats
        self.breed_to_superclass = {}
        for i, breed in enumerate(hf_dataset.unique('label_breed')):
            # Extract superclass information from metadata or name
            # Assuming first 12 breeds are dogs, rest are cats
            # You might need to adjust this logic based on your actual data
            if 'cat' in breed.lower() or i >= 12:
                self.breed_to_superclass[breed] = 0  # cat index
            else:
                self.breed_to_superclass[breed] = 1  # dog index

        def has_cat_or_dog(sample):
            if sample['label_bbox_enriched'] is None:
                return False
            bbox_labels = [annotation['label'] for annotation in sample['label_bbox_enriched']]
            return 'cat' in bbox_labels or 'dog' in bbox_labels

        self.hf_dataset = self.hf_dataset.filter(has_cat_or_dog)

    def __len__(self) -> int:
        return self.hf_dataset.num_rows

    def __getitem__(self, idx: int) -> Tuple[Tensor, dict]:
        sample = self.hf_dataset.__getitem__(idx)
        image = sample['image']
        image = np.array(image)

        annotations = self.loadAnnotations(sample, image.shape[1] / self.resize, image.shape[0] / self.resize)

        if len(annotations) == 0:
            targets = {
                'boxes': torch.zeros(1, 4, dtype=torch.float32),
                'labels': torch.as_tensor(len(self.classes), dtype=torch.int64), }
        else:
            targets = {
                'boxes': torch.as_tensor(annotations[..., :-1], dtype=torch.float32),
                'labels': torch.as_tensor(annotations[..., -1], dtype=torch.int64), }

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

            # Map directly to superclass index (0 for cat, 1 for dog)
            superclass_idx = 0 if annotation['label'] == 'cat' else 1

            bbox = annotation['bbox']
            bbox = [bbox[0] / imgWidth, bbox[1] / imgHeight, bbox[2] / imgWidth, bbox[3] / imgHeight]
            ans.append(bbox + [superclass_idx])

        return np.asarray(ans)