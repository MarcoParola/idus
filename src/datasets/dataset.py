from typing import Tuple, List, Dict

import hydra
import torch
import torchvision
from omegaconf import DictConfig
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
import numpy as np


class DetectionDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        return 10  # Dummy length for testing

    def __getitem__(self, idx):
        # Generate single channel image
        image = torch.randn(1, 224, 224)

        # Ensure all tensors have correct dtypes
        target = {
            'labels': torch.tensor([1], dtype=torch.long),  # Explicitly using long dtype for labels
            'boxes': torch.tensor([[0.0, 0.0, 100.0, 100.0]], dtype=torch.float32),
            'masks': torch.ones((1, 224, 224), dtype=torch.bool),
            'image_id': torch.tensor([idx], dtype=torch.long)
        }

        return image.float(), target  # Ensure image is float

    def __iter__(self):
        return self


def load_datasets(args):
    train, val, test = None, None, None

    if args.dataset == "oxford-pet":
        from src.datasets.oxford_pet import load_oxford_dataset
        from torchvision.transforms import Grayscale

        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((128, 128)),
            Grayscale(num_output_channels=1),  # Convert RGB to grayscale
            torchvision.transforms.ToTensor(),
        ])

        train, val, test = load_oxford_dataset(resize=128, transform=transform)

    return train, val, test


def collateFunction(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    images, targets = zip(*batch)

    # Convert any numpy arrays to tensors with correct dtypes
    images = [torch.from_numpy(img).float() if isinstance(img, np.ndarray) else img.float() for img in images]

    # For RGB images, convert to grayscale if needed
    images = [img.mean(dim=0, keepdim=True) if img.size(0) == 3 else img for img in images]

    # Stack images and ensure they're float tensors
    images_tensor = torch.stack(images).float()

    # Convert any numpy arrays in targets to tensors with correct dtypes
    targets_list = []
    for target in targets:
        tensor_target = {}
        for k, v in target.items():
            if isinstance(v, np.ndarray):
                if k == 'labels':
                    tensor_target[k] = torch.from_numpy(v).long()
                elif k == 'masks':
                    tensor_target[k] = torch.from_numpy(v).bool()
                else:
                    tensor_target[k] = torch.from_numpy(v).float()
            elif isinstance(v, Tensor):
                if k == 'labels':
                    tensor_target[k] = v.long()
                elif k == 'masks':
                    tensor_target[k] = v.bool()
                else:
                    tensor_target[k] = v.float()
            else:
                if k == 'labels':
                    tensor_target[k] = torch.tensor(v, dtype=torch.long)
                elif k == 'masks':
                    tensor_target[k] = torch.tensor(v, dtype=torch.bool)
                else:
                    tensor_target[k] = torch.tensor(v, dtype=torch.float32)
        targets_list.append(tensor_target)

    return images_tensor, targets_list


@hydra.main(config_path="C:\\Users\\pietr\\PycharmProjects\\idus\\config", config_name="config")
def main(cfg: DictConfig):
    # Step 1: Initialize the dataset
    dataset = DetectionDataset()

    # Step 2: Test the length of the dataset
    dataset_len = len(dataset)
    print(f"Dataset Length: {dataset_len}")
    assert dataset_len > 0, "Dataset length should be greater than 0"

    # Step 3: Test the `__getitem__` method
    sample = dataset[0]  # Fetch the first sample
    assert isinstance(sample, Tuple), "Sample should be a tuple"
    assert isinstance(sample[0], Tensor), "First element of the sample should be a Tensor"
    assert isinstance(sample[1], dict), "Second element of the sample should be a dictionary"
    print(f"Sample {0}: {sample}")

    # Step 4: Load datasets (train, val, test)
    train, val, test = load_datasets(cfg)
    assert train is not None, "Train dataset should not be None"
    assert val is not None, "Validation dataset should not be None"
    assert test is not None, "Test dataset should not be None"
    print("Datasets loaded successfully!")

    # Step 5: Test the `collateFunction` with a dummy batch
    dummy_batch = [(torch.randn(3, 224, 224), {'label': torch.tensor(1)}),
                   (torch.randn(3, 224, 224), {'label': torch.tensor(2)})]
    collated_data = collateFunction(dummy_batch)
    assert isinstance(collated_data, tuple), "Collate function should return a tuple"
    assert isinstance(collated_data[0], Tensor), "First element of collated data should be a Tensor"
    assert isinstance(collated_data[1], list), "Second element of collated data should be a list"
    print("Collate function works successfully!")

    # Optional: Test DataLoader with collate function
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collateFunction)
    for data in dataloader:
        print(f"Data: {data}")
        break  # Just print the first batch for testing

    print("Testing completed successfully.")


if __name__ == "__main__":
    main()
