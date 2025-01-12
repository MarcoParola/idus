from typing import Tuple, List, Dict
from torch import Tensor
import torch

class DetectionDataset(torch.utils.data.Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __iter__(self):
        return self

def load_datasets(args):
    train, val, test = None, None, None

    if args.dataset == "oxford-pet":
        from src.datasets.oxford_pet import load_oxford_dataset
        train, val, test = load_oxford_dataset()

    return train, val, test
        

def collateFunction(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, Tuple[Dict[str, Tensor]]]:
    batch = tuple(zip(*batch))
    return torch.stack(batch[0]), batch[1]