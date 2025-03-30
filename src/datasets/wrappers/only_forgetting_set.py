from typing import Tuple, Dict
import torch
from torch import Tensor
from torch.utils.data import Dataset


class OnlyForgettingSet(Dataset):
    """
    A dataset wrapper that only includes specified classes but handles background specially.
    This is the inverse of OnlyRetainingSet, used for negative gradient approaches.
    """

    def __init__(self, dataset, forgetting_set=None, removal='class'):
        """
        Initialize the wrapper with a dataset and classes to specifically include.

        Args:
            dataset: A detection dataset with classes attribute
            forgetting_set: List of class indices to specifically include
            removal: Type of filtering strategy ('class' for class-based filtering)
        """
        self.dataset = dataset
        self.removal = removal

        # Make a copy of forgetting_set to avoid modifying the original
        self.forgetting_set = forgetting_set.copy() if forgetting_set else []

        # Original classes from the dataset
        self.original_classes = dataset.classes

        # Check if background is present as the first class
        self.has_background = self.original_classes[0].lower() == 'background'
        self.preserve_background = True

        self.classes = self.original_classes.copy()

        self.class_mapping = {i: i for i in range(len(self.original_classes))}

        # Print information about the filtering
        forgetting_classes = [self.original_classes[idx] for idx in self.forgetting_set
                              if idx < len(self.original_classes)]

        print(f"Original class count: {len(self.original_classes) - (1 if self.has_background else 0)}")
        print(f"Focusing on forgetting classes: {forgetting_classes}")
        print(f"Retaining only annotations for these classes")

        # Include all indices to keep all images
        self.valid_indices = list(range(len(dataset)))
        print(f"Dataset size (all images): {len(self.valid_indices)}")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict]:
        # Use the filtered index
        actual_idx = self.valid_indices[idx]
        image, targets = self.dataset[actual_idx]

        # Filter the annotations to keep only those in the forgetting set
        if 'labels' in targets:
            labels = targets['labels']

            if isinstance(labels, torch.Tensor):
                # Find indices of labels in the forgetting set
                if labels.numel() > 0:
                    mask = torch.zeros_like(labels, dtype=torch.bool)
                    for forgetting_label in self.forgetting_set:
                        mask |= (labels == forgetting_label)

                    # If there are annotations in the forgetting set
                    if mask.any():
                        # Keep only the annotations for classes in the forgetting set
                        for key in targets:
                            if targets[key].shape[0] == labels.shape[0]:
                                targets[key] = targets[key][mask]
                    else:
                        # No annotations in forgetting set
                        targets = {
                            'boxes': torch.zeros(0, 4, dtype=torch.float32),
                            'labels': torch.tensor([], dtype=torch.int64),
                        }
                else:
                    # Empty label tensor
                    targets = {
                        'boxes': torch.zeros(0, 4, dtype=torch.float32),
                        'labels': torch.tensor([], dtype=torch.int64),
                    }
            else:
                # Handle non-tensor labels (e.g., lists)
                mask = [i for i, label in enumerate(labels) if label in self.forgetting_set]

                if mask:
                    for key in list(targets.keys()):
                        if hasattr(targets[key], '__len__') and len(targets[key]) == len(labels):
                            if isinstance(targets[key], list):
                                targets[key] = [targets[key][i] for i in mask]
                            elif isinstance(targets[key], torch.Tensor):
                                targets[key] = targets[key][torch.tensor(mask)]
                            # Add other types as needed
                else:
                    # No annotations in forgetting set
                    targets = {
                        'boxes': torch.zeros(0, 4, dtype=torch.float32),
                        'labels': torch.tensor([], dtype=torch.int64),
                    }

        return image, targets