from datasets import load_dataset
from typing import Tuple
from torch import Tensor
import torch
from PIL import Image
import numpy as np

from src.datasets.dataset import DetectionDataset


def load_oxford_dataset(resize=224, transform=None):

    dataset = load_dataset("visual-layer/oxford-iiit-pet-vl-enriched")
    print(dataset)
    # create validation set from train
    train_valid_split = dataset['train'].train_test_split(test_size=0.1)

    # Access the new train and validation sets
    train = train_valid_split['train']
    val = train_valid_split['test']
    test = dataset['test']
    train = OxfordPetDataset(train, resize=resize, transform=transform)
    val = OxfordPetDataset(val, resize=resize, transform=transform)
    test = OxfordPetDataset(test, resize=resize, transform=transform)
    return train, val, test


class OxfordPetDataset(DetectionDataset):
    """Oxford-IIIT Pet Dataset
    Args:
        hf_dataset: Hugging Face Dataset object
        resize: Resize the image to this size
        transform: Optional transform to be applied on a sample.

        Oxford-IIIT-Pet-enriched is a dataset of images of cats and dogs derived from the Oxford-IIIT Pet Dataset
        enriched with bboxes for cats, dogs, and other classes that will be skipped (e.g., person, blanket, etc.)
        __getitem__ returns a tuple of image and target (bbox and label) filtering out the other classes
    """
    def __init__(self, hf_dataset, resize=224, transform=None):
        super().__init__()
        self.hf_dataset = hf_dataset
        self.resize = resize
        self.transforms = transform
        self.classes = hf_dataset.unique('label_breed')

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
            # get index of the class from self.classes
            cat = self.classes.index(sample['label_breed'])
            bbox = annotation['bbox']
            bbox = [bbox[0]/imgWidth, bbox[1]/imgHeight, bbox[2]/imgWidth, bbox[3]/imgHeight]
            ans.append(bbox + [cat])

        return np.asarray(ans)

    
