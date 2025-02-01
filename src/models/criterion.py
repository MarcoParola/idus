from typing import Dict, List, Tuple
import torch
from torch import nn, Tensor

from src.utils.boxOps import boxCxcywh2Xyxy, gIoU, boxIoU
from .matcher import HungarianMatcher


class SetCriterion(nn.Module):
    def __init__(self, args):
        super(SetCriterion, self).__init__()
        self.matcher = HungarianMatcher(args.classCost, args.bboxCost, args.giouCost)
        self.numClass = args.numClass
        self.classCost = args.classCost
        self.bboxCost = args.bboxCost
        self.giouCost = args.giouCost

        emptyWeight = torch.ones(args.numClass + 1)
        emptyWeight[-1] = args.eosCost
        self.register_buffer('emptyWeight', emptyWeight)

        self.debug_counter = 0

    def forward(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        ans = self.computeLoss(x, y)

        for i, aux in enumerate(x['aux']):
            ans.update({f'{k}_aux{i}': v for k, v in self.computeLoss(aux, y).items()})

        return ans

    def computeLoss(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        ids = self.matcher(x, y)
        idx = self.getPermutationIdx(ids)

        # Debug print for matcher outputs
        if self.debug_counter % 100 == 0:
            print("\n=== Debug Info (Step {}) ===".format(self.debug_counter))
            print(f"Number of matched pairs: {sum(len(i[0]) for i in ids)}")

        # Classification loss computation with debug
        logits = x['class']
        targetClassO = torch.cat([t['labels'] for t, (_, J) in zip(y, ids)])
        targetClass = torch.full(logits.shape[:2], self.numClass, dtype=torch.int64, device=logits.device)
        targetClass[idx] = targetClassO

        if self.debug_counter % 100 == 0:
            print(f"Unique target classes: {torch.unique(targetClass).tolist()}")
            print(f"Class distribution: {torch.bincount(targetClass.flatten())}")
            print(f"Logits shape: {logits.shape}")
            print(f"Target class tensor shape: {targetClass.shape}")
            print(f"Indices for target assignment: {idx}")
            print(f"Assigned target classes (targetClass[idx]): {targetClass[idx]}")

        classificationLoss = nn.functional.cross_entropy(logits.transpose(1, 2), targetClass, self.emptyWeight)
        classificationLoss *= self.classCost

        # Box loss computation with debug
        mask = targetClassO != self.numClass
        boxes = x['bbox'][idx][mask]
        targetBoxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(y, ids)], 0)[mask]

        if self.debug_counter % 100 == 0:
            print(f"Number of valid boxes: {len(boxes)}")
            if len(boxes) > 0:
                print(f"Box coordinates range: min={boxes.min().item():.3f}, max={boxes.max().item():.3f}")

        numBoxes = len(targetBoxes) + 1e-6

        bboxLoss = nn.functional.l1_loss(boxes, targetBoxes, reduction='none')
        bboxLoss = bboxLoss.sum() / numBoxes
        bboxLoss *= self.bboxCost

        giouLoss = 1 - torch.diag(gIoU(boxCxcywh2Xyxy(boxes), boxCxcywh2Xyxy(targetBoxes)))
        giouLoss = giouLoss.sum() / numBoxes
        giouLoss *= self.giouCost

        # mAP computation with detailed debugging
        with torch.no_grad():
            # Get raw logits for matched predictions
            matched_logits = logits[idx]

            # Get predictions before and after softmax
            raw_max_vals, raw_pred = matched_logits.max(-1)
            softmaxed = nn.functional.softmax(matched_logits, -1)
            softmax_vals, pred_after_softmax = softmaxed.max(-1)

            # Original classification mask computation
            predClass = pred_after_softmax
            classMask = (predClass == targetClassO) & mask

            if self.debug_counter % 100 == 0:

                print("\nClassification Debug:")
                print(f"Raw predictions shape: {raw_pred.shape}")
                print(f"Raw predictions unique values: {torch.unique(raw_pred).tolist()}")
                print(f"Softmax predictions unique values: {torch.unique(pred_after_softmax).tolist()}")
                print(f"Target unique values: {torch.unique(targetClassO).tolist()}")
                print(f"Correct predictions: {(raw_pred == targetClassO).sum().item()}")
                print(f"Max confidence values: {softmax_vals.mean().item():.3f}")
                print(f"First prediction logits: {matched_logits[0].tolist()[:5]}...{matched_logits[0].tolist()[-5:]}")
                print(f"Logits range for first prediction: min={matched_logits[0].min().item():.3f}, "
                      f"max={matched_logits[0].max().item():.3f}")

            if len(boxes) > 0:
                iou = torch.diag(boxIoU(boxCxcywh2Xyxy(boxes), boxCxcywh2Xyxy(targetBoxes))[0])

                if self.debug_counter % 100 == 0:
                    print(
                        f"IoU stats: min={iou.min().item():.3f}, max={iou.max().item():.3f}, mean={iou.mean().item():.3f}")

                iou_th = [50, 75, 95]
                map_th = []
                ap = []

                for threshold in range(50, 100, 5):
                    ap_th = ((iou >= threshold / 100) * classMask).sum() / numBoxes
                    ap.append(ap_th)
                    if threshold in iou_th:
                        map_th.append(ap_th)
                        if self.debug_counter % 100 == 0:
                            print(f"AP@{threshold}: {ap_th.item():.3f}")

                ap = torch.mean(torch.stack(ap))
            else:
                ap = torch.tensor(0.0, device=logits.device)
                map_th = [torch.tensor(0.0, device=logits.device) for _ in range(3)]
                if self.debug_counter % 100 == 0:
                    print("WARNING: No valid boxes found for mAP computation")

        self.debug_counter += 1

        return {
            'classification loss': classificationLoss,
            'bbox loss': bboxLoss,
            'gIoU loss': giouLoss,
            'mAP': ap,
            'mAP_50': map_th[0],
            'mAP_75': map_th[1],
            'mAP_95': map_th[2]
        }

    @staticmethod
    def getPermutationIdx(indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        batchIdx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        srcIdx = torch.cat([src for (src, _) in indices])
        return batchIdx, srcIdx