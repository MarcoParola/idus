from typing import Tuple, List, Dict
import hydra
from omegaconf import DictConfig
from torch import Tensor
import torch
from torch.utils.data import Dataset, DataLoader


class DetectionDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        # Return the length of the dataset (for testing purposes, returning a fixed value)
        return 10  # Dummy length for testing

    def __getitem__(self, idx):
        # Return a dummy sample for testing
        return torch.randn(3, 224, 224), {'label': torch.tensor(1)}  # Dummy image and label

    def __iter__(self):
        return self


def load_datasets(args):
    train, val, test = None, None, None

    if args.dataset == "oxford-pet":
        from src.datasets.oxford_pet import load_oxford_dataset
        train, val, test = load_oxford_dataset()

    return train, val, test


def collateFunction(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    images, targets = zip(*batch)  # Unzip the batch into images and targets
    images_tensor = torch.stack(images)  # Stack the images into a tensor
    targets_list = list(targets)  # Convert the targets to a list

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
