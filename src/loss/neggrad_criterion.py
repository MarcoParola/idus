from src.loss.base_criterion import BaseCriterion
from src.utils.box_ops import boxCxcywh2Xyxy, gIoU

from typing import Dict, List
import torch
from torch import nn, Tensor


class NegGradCriterion(BaseCriterion):
    """
    NegGradLoss adapted for object detection - applies negative gradient to classification,
    bbox, and IoU losses for forgetting classes
    """

    def __init__(self, args):
        super(NegGradCriterion, self).__init__(args)
        self.forget_classes = args.excludeClasses if hasattr(args, 'excludeClasses') else []

    def computeLoss(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        try:
            if not hasattr(self, 'step_counter'):
                self.step_counter = 0

            # Filter out targets with no annotations
            valid_targets = [t for t in y if len(t['labels']) > 0]

            # If no valid targets, return zero losses
            if not valid_targets:
                device = x['class'].device
                return {
                    'classification loss': torch.tensor(0.0, device=device, requires_grad=True),
                    'bbox loss': torch.tensor(0.0, device=device, requires_grad=True),
                    'gIoU loss': torch.tensor(0.0, device=device, requires_grad=True)
                }

            # Use only valid targets
            y = valid_targets

            ids = self.matcher(x, y)
            idx = self.getPermutationIdx(ids)

            # Classification loss computation
            logits = x['class']
            targetClassO = torch.cat([t['labels'] for t, (_, J) in zip(y, ids)])
            targetClass = torch.full(logits.shape[:2], self.numClass, dtype=torch.int64, device=logits.device)
            targetClass[idx] = targetClassO

            classificationLoss = nn.functional.cross_entropy(logits.transpose(1, 2), targetClass, self.emptyWeight)
            classificationLoss *= self.classCost

            # Box loss computation
            mask = targetClassO != self.numClass

            # Ensure we have boxes to compute loss
            if mask.any():
                boxes = x['bbox'][idx][mask]
                targetBoxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(y, ids)], 0)[mask]

                numBoxes = len(targetBoxes) + 1e-6

                bboxLoss = nn.functional.l1_loss(boxes, targetBoxes, reduction='none')
                bboxLoss = bboxLoss.sum() / numBoxes
                bboxLoss *= self.bboxCost

                giouLoss = 1 - torch.diag(gIoU(boxCxcywh2Xyxy(boxes), boxCxcywh2Xyxy(targetBoxes)))
                giouLoss = giouLoss.sum() / numBoxes
                giouLoss *= self.giouCost
            else:
                # If no valid boxes, use zero losses
                device = x['class'].device
                bboxLoss = torch.tensor(0.0, device=device, requires_grad=True)
                giouLoss = torch.tensor(0.0, device=device, requires_grad=True)

            # Apply negative gradient since this is the forgetting set
            classificationLoss = -classificationLoss
            bboxLoss = -bboxLoss
            giouLoss = -giouLoss

            metrics = {
                'classification loss': classificationLoss,
                'bbox loss': bboxLoss,
                'gIoU loss': giouLoss,
            }

            # Debug print every 100 steps
            if self.step_counter % 100 == 0:
                print(f"[Step {self.step_counter}] NegGrad Losses: Class={classificationLoss.item():.4f}, "
                      f"BBox={bboxLoss.item():.4f}, gIoU={giouLoss.item():.4f}")

            self.step_counter += 1

            return metrics

        except Exception as e:
            print(f"Error in computeLoss: {str(e)}")
            # Return zero tensors with gradient
            device = x['class'].device
            return {
                'classification loss': torch.tensor(0.0, device=device, requires_grad=True),
                'bbox loss': torch.tensor(0.0, device=device, requires_grad=True),
                'gIoU loss': torch.tensor(0.0, device=device, requires_grad=True)
            }


class NegGradPlusCriterion(BaseCriterion):
    """
    NegGradPlusLoss adapted for object detection - applies positive gradient to retaining classes
    and negative gradient to forgetting classes
    """

    def __init__(self, args):
        super(NegGradPlusCriterion, self).__init__(args)
        self.forget_classes = args.excludeClasses if hasattr(args, 'excludeClasses') else []

    def computeLoss(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        try:
            if not hasattr(self, 'step_counter'):
                self.step_counter = 0

            # Filter out targets with no annotations
            valid_targets = [t for t in y if len(t['labels']) > 0]

            # If no valid targets, return zero losses
            if not valid_targets:
                device = x['class'].device
                return {
                    'classification loss': torch.tensor(0.0, device=device, requires_grad=True),
                    'bbox loss': torch.tensor(0.0, device=device, requires_grad=True),
                    'gIoU loss': torch.tensor(0.0, device=device, requires_grad=True)
                }

            # Use only valid targets
            y = valid_targets

            ids = self.matcher(x, y)
            idx = self.getPermutationIdx(ids)

            # Classification loss computation
            logits = x['class']
            targetClassO = torch.cat([t['labels'] for t, (_, J) in zip(y, ids)])
            targetClass = torch.full(logits.shape[:2], self.numClass, dtype=torch.int64, device=logits.device)
            targetClass[idx] = targetClassO

            # Use cross entropy with reduction='none' to get per-element losses
            classLoss = nn.functional.cross_entropy(
                logits.transpose(1, 2),
                targetClass,
                self.emptyWeight,
                reduction='none'
            )

            # Apply class-specific signs to losses
            batchIdx = idx[0]
            srcIdx = idx[1]

            # Default multiplier is 1.0 (positive gradient)
            classMultiplier = torch.ones_like(classLoss)

            # Create a mask for the positions where we have matched predictions
            valid_positions = torch.zeros_like(classLoss, dtype=torch.bool)
            valid_positions[batchIdx, srcIdx] = True

            # For forget classes, set multiplier to -1.0 (negative gradient)
            for i, (batch_i, src_i) in enumerate(zip(batchIdx, srcIdx)):
                if int(targetClassO[i]) in self.forget_classes:
                    classMultiplier[batch_i, src_i] = -1.0

            # Apply the multiplier and take mean
            classificationLoss = (classLoss * classMultiplier).mean() * self.classCost

            # Box loss computation
            mask = targetClassO != self.numClass

            # Ensure we have boxes to compute loss
            if mask.any():
                boxes = x['bbox'][idx][mask]
                targetBoxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(y, ids)], 0)[mask]

                # Get classes for the matched boxes
                boxClasses = targetClassO[mask]

                numBoxes = len(targetBoxes) + 1e-6

                # Compute individual box losses
                bboxLosses = nn.functional.l1_loss(boxes, targetBoxes, reduction='none')
                giouLosses = 1 - torch.diag(gIoU(boxCxcywh2Xyxy(boxes), boxCxcywh2Xyxy(targetBoxes)))

                # Create multipliers for box losses (-1 for forget classes, 1 for retain classes)
                boxMultipliers = torch.ones(len(boxClasses), device=boxClasses.device)
                for i, cls in enumerate(boxClasses):
                    if int(cls) in self.forget_classes:
                        boxMultipliers[i] = -1.0

                # Apply multipliers and average
                bboxLoss = (bboxLosses.sum(dim=1) * boxMultipliers).sum() / numBoxes * self.bboxCost
                giouLoss = (giouLosses * boxMultipliers).sum() / numBoxes * self.giouCost
            else:
                # If no valid boxes, use zero losses
                device = x['class'].device
                bboxLoss = torch.tensor(0.0, device=device, requires_grad=True)
                giouLoss = torch.tensor(0.0, device=device, requires_grad=True)

            metrics = {
                'classification loss': classificationLoss,
                'bbox loss': bboxLoss,
                'gIoU loss': giouLoss,
            }

            # Debug print every 100 steps
            if self.step_counter % 100 == 0:
                print(f"[Step {self.step_counter}] NegGrad+ Losses: Class={classificationLoss.item():.4f}, "
                      f"BBox={bboxLoss.item():.4f}, gIoU={giouLoss.item():.4f}")

            self.step_counter += 1

            return metrics

        except Exception as e:
            print(f"Error in computeLoss: {str(e)}")
            # Return zero tensors with gradient
            device = x['class'].device
            return {
                'classification loss': torch.tensor(0.0, device=device, requires_grad=True),
                'bbox loss': torch.tensor(0.0, device=device, requires_grad=True),
                'gIoU loss': torch.tensor(0.0, device=device, requires_grad=True)
            }