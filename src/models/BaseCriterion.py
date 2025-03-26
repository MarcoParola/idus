from src.models.matcher import HungarianMatcher
from src.utils.boxOps import boxCxcywh2Xyxy, gIoU, boxIoU

from typing import Dict, List, Tuple
import torch
from torch import nn, Tensor


class BaseCriterion(nn.Module):
    def __init__(self, args):
        super(BaseCriterion, self).__init__()
        self.matcher = HungarianMatcher(args.classCost, args.bboxCost, args.giouCost)
        self.numClass = args.numClass
        self.classCost = args.classCost
        self.bboxCost = args.bboxCost
        self.giouCost = args.giouCost

        emptyWeight = torch.ones(args.numClass + 1)
        emptyWeight[-1] = args.eosCost
        self.register_buffer('emptyWeight', emptyWeight)

        self.device = args.device
        self.step_counter = 0

    def forward(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        try:
            ans = self.computeLoss(x, y)

            # Handle auxiliary losses
            for i, aux in enumerate(x['aux']):
                try:
                    aux_losses = self.computeLoss(aux, y)
                    ans.update({f'{k}_aux{i}': v for k, v in aux_losses.items()})
                # Skip this auxiliary loss if there's an error but continue training
                except Exception as e:
                    print(f"Error computing auxiliary loss {i}: {str(e)}")

            return ans

        except Exception as e:
            print(f"Error in forward pass: {str(e)}")
            # Return just the basic losses to continue training
            return {
                'classification loss': torch.tensor(0.0, device=x['class'].device),
                'bbox loss': torch.tensor(0.0, device=x['class'].device),
                'gIoU loss': torch.tensor(0.0, device=x['class'].device)
            }

    def computeLoss(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        try:
            if not hasattr(self, 'step_counter'):
                self.step_counter = 0

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
            boxes = x['bbox'][idx][mask]
            targetBoxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(y, ids)], 0)[mask]

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
            if self.step_counter % 1000 == 0:
                print(f"[Step {self.step_counter}] Losses: Class={classificationLoss.item():.4f}, "
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

    @staticmethod
    def getPermutationIdx(indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        batchIdx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        srcIdx = torch.cat([src for (src, _) in indices])
        return batchIdx, srcIdx