from src.models.BaseCriterion import BaseCriterion
from src.utils.boxOps import boxCxcywh2Xyxy, gIoU

from typing import Dict, List
import torch
from torch import nn, Tensor
import random


class RandomRelabellingCriterion(BaseCriterion):
    """
    RandomRelabellingCriterion for object detection - randomly relabels objects from forgetting classes
    to other valid classes during training to induce forgetting
    """

    def __init__(self, args):
        super(RandomRelabellingCriterion, self).__init__(args)
        self.forget_classes = args.excludeClasses if hasattr(args, 'excludeClasses') else []

        # Create a list of available classes for relabelling (excluding forget classes)
        self.available_classes = [i for i in range(self.numClass) if i not in self.forget_classes]

        # If we have no available classes to relabel to (e.g., all classes are to be forgotten),
        # we'll just use random integers within numClass range
        if not self.available_classes:
            self.available_classes = list(range(self.numClass))

    def computeLoss(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        try:
            if not hasattr(self, 'step_counter'):
                self.step_counter = 0

            # First, modify the targets by random relabelling the forgetting classes
            modified_y = []

            for target in y:
                new_target = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in target.items()}
                labels = new_target['labels']

                # Create a mask for forgetting classes
                forget_mask = torch.zeros_like(labels, dtype=torch.bool)
                for cls in self.forget_classes:
                    forget_mask |= (labels == cls)

                # For each label in forget classes, relabel it randomly to another class
                if forget_mask.any():
                    # Get number of labels to relabel
                    num_to_relabel = forget_mask.sum().item()

                    # Generate random labels (excluding forgetting classes)
                    random_labels = torch.tensor(
                        random.choices(self.available_classes, k=num_to_relabel),
                        device=labels.device,
                        dtype=labels.dtype
                    )

                    # Apply random labels
                    new_target['labels'][forget_mask] = random_labels

                modified_y.append(new_target)

            # Now compute loss using the modified targets
            ids = self.matcher(x, modified_y)
            idx = self.getPermutationIdx(ids)

            # Classification loss computation
            logits = x['class']
            targetClassO = torch.cat([t['labels'] for t, (_, J) in zip(modified_y, ids)])
            targetClass = torch.full(logits.shape[:2], self.numClass, dtype=torch.int64, device=logits.device)
            targetClass[idx] = targetClassO

            classificationLoss = nn.functional.cross_entropy(logits.transpose(1, 2), targetClass, self.emptyWeight)
            classificationLoss *= self.classCost

            # Box loss computation
            mask = targetClassO != self.numClass
            boxes = x['bbox'][idx][mask]
            targetBoxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(modified_y, ids)], 0)[mask]

            numBoxes = len(targetBoxes) + 1e-6

            bboxLoss = nn.functional.l1_loss(boxes, targetBoxes, reduction='none')
            bboxLoss = bboxLoss.sum() / numBoxes
            bboxLoss *= self.bboxCost

            giouLoss = 1 - torch.diag(gIoU(boxCxcywh2Xyxy(boxes), boxCxcywh2Xyxy(targetBoxes)))
            giouLoss = giouLoss.sum() / numBoxes
            giouLoss *= self.giouCost

            metrics = {
                'classification loss': classificationLoss,
                'bbox loss': bboxLoss,
                'gIoU loss': giouLoss,
            }

            # Debug print every 100 steps
            if self.step_counter % 100 == 0:
                print(f"[Step {self.step_counter}] RandomRelabelling Losses: Class={classificationLoss.item():.4f}, "
                      f"BBox={bboxLoss.item():.4f}, gIoU={giouLoss.item():.4f}")

            self.step_counter += 1

            return metrics

        except Exception as e:
            print(f"Error in computeLoss: {str(e)}")
            # Return empty metrics to continue training
            device = x['class'].device if 'class' in x else 'cpu'
            return {
                'classification loss': torch.tensor(0.0, device=device),
                'bbox loss': torch.tensor(0.0, device=device),
                'gIoU loss': torch.tensor(0.0, device=device)
            }