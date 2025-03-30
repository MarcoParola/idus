from typing import Tuple, Dict
import torch
from torch import Tensor
from torch.utils.data import Dataset
import random


class RandomRelabellingSet(Dataset):
    """
    A dataset wrapper that randomly relabels instances of specified classes to other valid classes.
    This is used for the random relabelling machine unlearning approach.
    """

    def __init__(self, dataset, forgetting_set=None, removal='CR'):
        """
        Initialize the wrapper with a dataset and classes to relabel.

        Args:
            dataset: A detection dataset with classes attribute
            forgetting_set: List of class indices to randomly relabel
            removal: Type of unlearning strategy
        """
        self.dataset = dataset
        self.removal = removal

        # Make a copy of forgetting_set to avoid modifying the original
        self.forgetting_set = forgetting_set.copy() if forgetting_set else []

        # Original classes from the dataset
        self.original_classes = dataset.classes
        self.classes = self.original_classes.copy()

        # Check if background is present as the first class
        self.has_background = self.original_classes[0].lower() == 'background'
        self.background_idx = 0 if self.has_background else None

        # Remove background from forgetting set if it's included (it doesn't make sense to "forget" background)
        if self.has_background and self.background_idx in self.forgetting_set:
            self.forgetting_set.remove(self.background_idx)
            print("Warning: Background class was in forgetting set and has been removed.")

        # Create a list of available classes for relabelling (excluding forget classes and background)
        self.available_classes = []
        for i in range(len(self.classes)):
            # Skip forgetting classes
            if i in self.forgetting_set:
                continue

            # Skip background class
            if self.has_background and i == self.background_idx:
                continue

            self.available_classes.append(i)

        # If we have no available classes to relabel to (extreme case), use any non-background class
        if not self.available_classes:
            if self.has_background:
                # Use all classes except background if all classes are to be forgotten
                self.available_classes = [i for i in range(len(self.classes)) if i != self.background_idx]
            else:
                # If no background and no available classes, just use all indices
                self.available_classes = list(range(len(self.classes)))

        # Class mapping remains the same
        self.class_mapping = {i: i for i in range(len(self.original_classes))}

        # Print information about the relabelling
        forgetting_classes = [self.original_classes[idx] for idx in self.forgetting_set
                              if idx < len(self.original_classes)]
        available_class_names = [self.original_classes[idx] for idx in self.available_classes
                                 if idx < len(self.original_classes)]

        print(f"Original class count: {len(self.original_classes) - (1 if self.has_background else 0)}")
        print(f"Classes to be randomly relabelled: {forgetting_classes}")
        print(f"Will be relabelled to one of: {available_class_names}")

        # Add debug print showing re-labeled samples count
        relabeled_count = 0
        for idx in range(len(dataset)):
            image, targets = dataset[idx]
            if 'labels' in targets:
                labels = targets['labels']
                for cls in self.forgetting_set:
                    if isinstance(labels, torch.Tensor):
                        relabeled_count += torch.sum(labels == cls).item()
                    else:
                        relabeled_count += labels.count(cls)

        print(f"Total instances that will be relabelled: {relabeled_count}")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict]:
        image, targets = self.dataset[idx]

        # Apply random relabelling to targets
        if 'labels' in targets:
            labels = targets['labels']

            if isinstance(labels, torch.Tensor):
                # Create a mask for forgetting classes
                forget_mask = torch.zeros_like(labels, dtype=torch.bool)
                for cls in self.forgetting_set:
                    forget_mask |= (labels == cls)

                # For each label in forget classes, relabel it randomly to another class
                if forget_mask.any() and len(self.available_classes) > 0:
                    # Get number of labels to relabel
                    num_to_relabel = forget_mask.sum().item()

                    # Generate random labels from available classes
                    random_labels = torch.tensor(
                        random.choices(self.available_classes, k=num_to_relabel),
                        device=labels.device,
                        dtype=labels.dtype
                    )

                    # Apply random labels
                    targets['labels'][forget_mask] = random_labels
            else:
                # Handle non-tensor labels (e.g., lists)
                for i, label in enumerate(labels):
                    if label in self.forgetting_set and self.available_classes:
                        # Replace with a random label from available classes
                        targets['labels'][i] = random.choice(self.available_classes)

        return image, targets