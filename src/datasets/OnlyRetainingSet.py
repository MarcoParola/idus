from typing import Tuple, Dict
import torch
from torch import Tensor
from torch.utils.data import Dataset


class OnlyRetainingSet(Dataset):
    """
    A dataset wrapper that excludes specified classes but handles background specially.
    """

    def __init__(self, dataset, forgetting_set=None, removal='class'):
        """
        Initialize the wrapper with a dataset and classes to forget/exclude.

        Args:
            dataset: A detection dataset with classes attribute
            forgetting_set: List of class indices to exclude/forget
            removal: Type of removal strategy ('class' for class-based filtering)
        """
        self.dataset = dataset
        self.removal = removal

        # Make a copy of forgetting_set to avoid modifying the original
        self.forgetting_set = forgetting_set.copy() if forgetting_set else []

        # Original classes from the dataset
        self.original_classes = dataset.classes

        # Check if background is present as the first class
        self.has_background = self.original_classes[0].lower() == 'background'

        # Always preserve background class if it exists (remove it from forgetting_set)
        if self.has_background and 0 in self.forgetting_set:
            self.forgetting_set.remove(0)
            print("Note: Background class (0) removed from forgetting set to preserve detection functionality")

        # Create a filtered list of classes
        if self.has_background:
            # Always keep background as the first class
            self.classes = [self.original_classes[0]]

            # Add other classes that aren't in the forgetting set
            for i, cls in enumerate(self.original_classes[1:], 1):
                if i not in self.forgetting_set:
                    self.classes.append(cls)
        else:
            # No background, just filter normally
            self.classes = [
                cls for i, cls in enumerate(self.original_classes)
                if i not in self.forgetting_set
            ]

        # Create mapping from original indices to new indices
        self.class_mapping = {}
        new_idx = 0

        # If background exists, it always maps to index 0
        if self.has_background:
            self.class_mapping[0] = 0
            new_idx = 1
            start_idx = 1
        else:
            start_idx = 0

        # Map the remaining classes to new consecutive indices
        for orig_idx in range(start_idx, len(self.original_classes)):
            if orig_idx not in self.forgetting_set:
                self.class_mapping[orig_idx] = new_idx
                new_idx += 1

        # Print information about the filtering
        forgetting_classes = [self.original_classes[idx] for idx in self.forgetting_set
                              if idx < len(self.original_classes)]

        print(f"Original class count: {len(self.original_classes) - (1 if self.has_background else 0)}")
        print(f"Forgetting classes: {forgetting_classes}")
        print(f"After excluding {len(self.forgetting_set)} classes: "
              f"{len(self.classes) - (1 if self.has_background else 0)} classes")

        # Filter dataset to only include samples with valid annotations after exclusion
        self.valid_indices = []
        for idx in range(len(dataset)):
            if self._has_valid_annotations(idx):
                self.valid_indices.append(idx)

        print(f"Dataset size after filtering: {len(self.valid_indices)}")

    def _has_valid_annotations(self, idx):
        """Check if the sample has valid annotations after exclusion."""
        _, targets = self.dataset[idx]

        if 'labels' in targets:
            # For datasets that return tensor labels
            labels = targets['labels'].numpy() if isinstance(targets['labels'], torch.Tensor) else targets['labels']

            # Check if any label is valid (not in forgetting_set)
            for label in labels:
                # If it's background (0) and we're keeping background, it's valid
                if self.has_background and label == 0:
                    return True

                # If it's not a class we're forgetting, it's valid
                if label not in self.forgetting_set:
                    return True

        return False

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict]:
        # Use the filtered index
        actual_idx = self.valid_indices[idx]
        image, targets = self.dataset[actual_idx]

        # Process boxes and labels
        if 'labels' in targets:
            if isinstance(targets['labels'], torch.Tensor):
                # Get mask for valid classes (considering background preservation)
                valid_mask = torch.tensor([
                    (label.item() == 0 and self.has_background and self.preserve_background) or
                    (label.item() not in self.forgetting_set)
                    for label in targets['labels']
                ])

                if 'boxes' in targets and valid_mask.any():
                    # Filter boxes and labels
                    boxes = targets['boxes'][valid_mask]
                    original_labels = targets['labels'][valid_mask]

                    # Remap the class labels
                    remapped_labels = torch.tensor([
                        self.class_mapping[label.item()]
                        for label in original_labels
                    ], dtype=torch.int64)

                    new_targets = targets.copy()
                    new_targets['boxes'] = boxes
                    new_targets['labels'] = remapped_labels

                    # Handle other target attributes if present
                    for key in targets:
                        if key not in ['boxes', 'labels'] and hasattr(targets[key], '__getitem__'):
                            try:
                                new_targets[key] = targets[key][valid_mask]
                            except:
                                # Keep original if slicing fails
                                new_targets[key] = targets[key]
                else:
                    # No valid annotations left
                    new_targets = {
                        'boxes': torch.zeros(0, 4, dtype=torch.float32),
                        'labels': torch.tensor([], dtype=torch.int64),
                    }
                    # Preserve other keys from original targets
                    for key in targets:
                        if key not in ['boxes', 'labels']:
                            new_targets[key] = targets[key]
            else:
                # For non-tensor labels, create a custom implementation
                # This is a placeholder for datasets with different formats
                new_targets = targets
        else:
            # If there are no labels, pass through the targets unchanged
            new_targets = targets

        return image, new_targets