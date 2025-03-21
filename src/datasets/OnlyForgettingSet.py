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

        # Filter dataset to only include samples with annotations from forgetting set
        self.valid_indices = []
        for idx in range(len(dataset)):
            if self._has_forgetting_class_annotations(idx):
                self.valid_indices.append(idx)

        print(f"Dataset size after filtering: {len(self.valid_indices)}")

    def _has_forgetting_class_annotations(self, idx):
        """Check if the sample has any annotations from the forgetting set."""
        _, targets = self.dataset[idx]

        if 'labels' in targets:
            # For datasets that return tensor labels
            labels = targets['labels'].numpy() if isinstance(targets['labels'], torch.Tensor) else targets['labels']

            # Check if any label is in the forgetting set
            for label in labels:
                if label in self.forgetting_set:
                    return True

        return False

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
                mask = torch.zeros_like(labels, dtype=torch.bool)
                for forgetting_label in self.forgetting_set:
                    mask |= (labels == forgetting_label)

                # Keep only the annotations for classes in the forgetting set
                for key in targets:
                    if targets[key].shape[0] == labels.shape[
                        0]:  # Only filter tensors with same first dimension as labels
                        targets[key] = targets[key][mask]
            else:
                # Handle non-tensor labels (e.g., lists)
                mask = [i for i, label in enumerate(labels) if label in self.forgetting_set]

                for key in list(targets.keys()):
                    if hasattr(targets[key], '__len__') and len(targets[key]) == len(labels):
                        if isinstance(targets[key], list):
                            targets[key] = [targets[key][i] for i in mask]
                        elif isinstance(targets[key], torch.Tensor):
                            targets[key] = targets[key][torch.tensor(mask)]
                        # Add other types as needed

        return image, targets