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

        #for i, aux in enumerate(x['aux']):
            #ans.update({f'{k}_aux{i}': v for k, v in self.computeLoss(aux, y).items()})

        return ans

    def computeLoss(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
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

        # mAP computation
        with torch.no_grad():
            matched_logits = logits[idx]
            softmaxed = nn.functional.softmax(matched_logits, -1)
            pred_after_softmax = softmaxed.argmax(-1)
            predClass = pred_after_softmax
            classMask = (predClass == targetClassO) & mask

            if len(boxes) > 0:
                iou = torch.diag(boxIoU(boxCxcywh2Xyxy(boxes), boxCxcywh2Xyxy(targetBoxes))[0])

                iou_th = [50, 75, 95]
                map_th = []
                ap = []
                for threshold in range(50, 100, 5):
                    ap_th = ((iou >= threshold / 100) * classMask).sum() / numBoxes
                    ap.append(ap_th)
                    if threshold in iou_th:
                        map_th.append(ap_th)
                ap = torch.mean(torch.stack(ap))

                # Compute per-class AP including mAP_50, 75, 95
                per_class_ap = {}
                per_class_ap_50 = {}
                per_class_ap_75 = {}
                per_class_ap_95 = {}

                for c in range(self.numClass):
                    class_mask = (targetClassO == c) & mask
                    num_class_instances = class_mask.sum().item()
                    if num_class_instances == 0:
                        per_class_ap[c] = torch.tensor(0.0, device=logits.device)
                        per_class_ap_50[c] = torch.tensor(0.0, device=logits.device)
                        per_class_ap_75[c] = torch.tensor(0.0, device=logits.device)
                        per_class_ap_95[c] = torch.tensor(0.0, device=logits.device)
                        continue

                    class_iou = iou[class_mask]
                    class_pred_correct = (predClass[class_mask] == c)

                    ap_values = []
                    ap_50, ap_75, ap_95 = 0, 0, 0
                    for threshold in range(50, 100, 5):
                        th = threshold / 100
                        tp = ((class_iou >= th) & class_pred_correct).sum().float()
                        ap_val = tp / num_class_instances
                        ap_values.append(ap_val)
                        if threshold == 50:
                            ap_50 = ap_val
                        elif threshold == 75:
                            ap_75 = ap_val
                        elif threshold == 95:
                            ap_95 = ap_val

                    ap_mean = torch.mean(torch.stack(ap_values)) if ap_values else torch.tensor(0.0)
                    per_class_ap[c] = ap_mean
                    per_class_ap_50[c] = ap_50
                    per_class_ap_75[c] = ap_75
                    per_class_ap_95[c] = ap_95
            else:
                ap = torch.tensor(0.0, device=logits.device)
                map_th = [torch.tensor(0.0, device=logits.device) for _ in range(3)]
                per_class_ap = {c: torch.tensor(0.0, device=logits.device) for c in range(self.numClass)}
                per_class_ap_50 = per_class_ap.copy()
                per_class_ap_75 = per_class_ap.copy()
                per_class_ap_95 = per_class_ap.copy()

        self.debug_counter += 1

        # Prepare metrics dictionary
        metrics = {
            'classification loss': classificationLoss,
            'bbox loss': bboxLoss,
            'gIoU loss': giouLoss,
            'mAP': ap,
            'mAP_50': map_th[0],
            'mAP_75': map_th[1],
            'mAP_95': map_th[2]
        }

        # Add per-class AP metrics
        for c in range(self.numClass):
            metrics[f'AP_class_{c}'] = per_class_ap[c]
            metrics[f'AP_class_{c}_50'] = per_class_ap_50[c]
            metrics[f'AP_class_{c}_75'] = per_class_ap_75[c]
            metrics[f'AP_class_{c}_95'] = per_class_ap_95[c]

        return metrics

    @staticmethod
    def getPermutationIdx(indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        batchIdx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        srcIdx = torch.cat([src for (src, _) in indices])
        return batchIdx, srcIdx