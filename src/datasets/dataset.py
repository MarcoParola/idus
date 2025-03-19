from typing import Tuple, List, Dict

import torch
from torch import Tensor
from torch.utils.data import Dataset
import numpy as np


class DetectionDataset(Dataset):
    def __init__(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

    def __iter__(self):
        return self


def load_datasets(args):
    import torchvision.transforms as transforms

    train, val, test = None, None, None

    if args.dataset == "oxford-pet":
        from src.datasets.oxford_pet import load_oxford_dataset

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Lambda(remove_alpha_channel),
        ])

        train, val, test = load_oxford_dataset(
            resize=128,
            transform=transform,
        )

    elif args.dataset == "oxford-pet-superclass":
        from src.datasets.oxford_pet_superclass import load_oxford_superclass_dataset

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Lambda(remove_alpha_channel),
        ])

        train, val, test= load_oxford_superclass_dataset(
            resize=128,
            transform=transform,
        )

    elif args.dataset == "fashion":
        from src.datasets.fashion import load_fashion_dataset

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        train, val, test= load_fashion_dataset(
            resize=128,
            transform=transform,)

    elif args.dataset == "voc":
        from src.datasets.voc import load_voc_dataset

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        train, val, test= load_voc_dataset(
            root=args.dataDir if hasattr(args, 'dataDir') else "./data",
            year=args.vocYear if hasattr(args, 'vocYear') else "2012",
            resize=128,
            transform=transform,
        )


    return train, val, test


def remove_alpha_channel(img):
    """Remove the alpha channel if present."""
    return img[:3] if img.shape[0] == 4 else img


def collateFunction(batch: List[Tuple[Tensor, dict]]) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    images, targets = zip(*batch)

    # Convert any numpy arrays to tensors with correct dtypes.
    images = [torch.from_numpy(img).float() if isinstance(img, np.ndarray) else img.float() for img in images]

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