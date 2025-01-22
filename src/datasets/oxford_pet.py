from datasets import load_dataset
from typing import Tuple
from torch import Tensor
import torchvision
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
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


if __name__ == "__main__":

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 128)),
        torchvision.transforms.ToTensor(),
    ])

    train, val, test = load_oxford_dataset(resize=128, transform=transform)

    import numpy as np
    from PIL import Image
    from torchvision.transforms import ToPILImage, ToTensor
    import matplotlib.pyplot as plt
    
    for i in range(50):
        image_dataset, target_dataset = test.__getitem__(i)
        # plot
        image = ToPILImage()(image_dataset)
        image = np.array(image)
        plt.imshow(image)

        for bbox in target_dataset['boxes']:
            x, y, w, h = bbox
            x, y, w, h = x*128, y*128, w*128, h*128
            # Draw the bounding box on the image using matplotlib
            plt.plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], color='r', linewidth=2)

        plt.show()
    

    '''
    for i in range(10):
        tmp = test.__getitem__(i)
        pippo = dataset_train.__getitem__(i)
        print(tmp)       
        print(type(tmp['image']))

        image = tmp['image']
        bboxes = tmp['label_bbox_enriched']
        
        image = ToTensor()(image)  # Convert to Tensor first
        image = ToPILImage()(image)  
        image = np.array(image)

        image_train = pippo[0]
        target_train = pippo[1]

        # plot 2 images in subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image)
        axs[0].set_title('Original image')
        for bbox in bboxes:
            x, y, w, h = bbox['bbox']
            # Draw the bounding box on the image using matplotlib
            axs[0].plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], color='r', linewidth=2)

        axs[1].imshow(image_train.permute(1, 2, 0))
        axs[1].set_title('Transformed image')
        for bbox in target_train['boxes']:
            x, y, w, h = bbox
            # Draw the bounding box on the image using matplotlib
            axs[1].plot([x, x + w, x + w, x, x], [y, y, y + h, y + h, y], color='r', linewidth=2)
            
        plt.show()
    '''

    
