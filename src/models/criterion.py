from typing import Dict, List, Tuple
import torch
from torch import nn, Tensor
from collections import defaultdict

from src.models.matcher import HungarianMatcher
from src.utils.boxOps import boxCxcywh2Xyxy, gIoU, boxIoU


class APCalculator:
    def __init__(self, num_classes, device):
        self.num_classes = num_classes
        self.device = device
        self.reset()

    def reset(self):
        self.predictions = {
            'class_ids': [],
            'confidences': [],
            'is_correct': [],
            'ious': []
        }
        self.gt_counts = torch.zeros(self.num_classes, device=self.device)

    def update(self, pred_classes, target_classes, confidences, ious):
        """Update detection statistics using tensor operations"""
        # Ensure all inputs are on the correct device
        pred_classes = pred_classes.to(self.device)
        target_classes = target_classes.to(self.device)
        confidences = confidences.to(self.device)
        ious = ious.to(self.device)

        # Update ground truth counts
        unique_classes, class_counts = torch.unique(target_classes, return_counts=True)
        for cls, count in zip(unique_classes, class_counts):
            if cls != self.num_classes:  # Ignore background class
                self.gt_counts[cls] += count

        # Store predictions as tensors
        mask = pred_classes != self.num_classes
        self.predictions['class_ids'].append(pred_classes[mask])
        self.predictions['confidences'].append(confidences[mask])
        self.predictions['is_correct'].append(pred_classes[mask] == target_classes[mask])
        self.predictions['ious'].append(ious)

    def compute_ap(self, class_id, threshold):
        """Compute AP for a specific class and IoU threshold"""
        if not self.predictions['class_ids']:
            return torch.tensor(0.0, device=self.device)

        # Concatenate all predictions
        class_ids = torch.cat(self.predictions['class_ids'])
        confidences = torch.cat(self.predictions['confidences'])
        is_correct = torch.cat(self.predictions['is_correct'])
        ious = torch.cat(self.predictions['ious'])

        # Get predictions for this class
        class_mask = class_ids == class_id
        if not class_mask.any() or self.gt_counts[class_id] == 0:
            return torch.tensor(0.0, device=self.device)

        class_confidences = confidences[class_mask]
        class_is_correct = is_correct[class_mask]
        class_ious = ious[class_mask]

        # Sort by confidence
        sorted_indices = torch.argsort(class_confidences, descending=True)
        class_is_correct = class_is_correct[sorted_indices]
        class_ious = class_ious[sorted_indices]

        # Compute true positives and false positives for this threshold
        valid_detections = class_ious >= threshold
        true_positives = (valid_detections & class_is_correct).float()
        false_positives = (valid_detections & ~class_is_correct).float()

        # Compute precision and recall
        cum_true_positives = torch.cumsum(true_positives, dim=0)
        cum_false_positives = torch.cumsum(false_positives, dim=0)
        recalls = cum_true_positives / self.gt_counts[class_id]
        precisions = cum_true_positives / (cum_true_positives + cum_false_positives + 1e-6)

        # Compute AP using 11-point interpolation
        ap = torch.tensor(0.0, device=self.device)
        for recall_point in torch.arange(0, 1.1, 0.1, device=self.device):
            mask = recalls >= recall_point
            if mask.any():
                ap += torch.max(precisions[mask]) / 11.0

        return ap

    def compute_metrics(self, include_per_class=False):
        """Compute AP metrics, optionally including per-class metrics"""
        metrics = {}

        # Initialize tensors for storing APs
        class_aps = torch.zeros((self.num_classes, 3), device=self.device)  # 3 thresholds: 0.5, 0.75, 0.95

        # Compute AP for each class and threshold
        for class_id in range(self.num_classes):
            # Compute AP for standard thresholds
            class_aps[class_id, 0] = self.compute_ap(class_id, 0.5)  # AP@50
            class_aps[class_id, 1] = self.compute_ap(class_id, 0.75)  # AP@75
            class_aps[class_id, 2] = self.compute_ap(class_id, 0.95)  # AP@95

            if include_per_class:
                # Compute mean AP across thresholds for this class
                thresholds = torch.arange(50, 100, 5, device=self.device) / 100
                class_ap_mean = torch.mean(torch.stack([
                    self.compute_ap(class_id, th) for th in thresholds
                ]))
                metrics[f'AP_class_{class_id}'] = class_ap_mean

                # Store individual threshold APs
                metrics[f'AP_class_{class_id}_50'] = class_aps[class_id, 0]
                metrics[f'AP_class_{class_id}_75'] = class_aps[class_id, 1]
                metrics[f'AP_class_{class_id}_95'] = class_aps[class_id, 2]

        # Always compute mAP metrics
        metrics['mAP_50'] = class_aps[:, 0].mean()
        metrics['mAP_75'] = class_aps[:, 1].mean()
        metrics['mAP_95'] = class_aps[:, 2].mean()

        # Compute overall mAP (across all classes and thresholds)
        all_thresholds = torch.arange(50, 100, 5, device=self.device) / 100
        all_aps = []
        for class_id in range(self.num_classes):
            class_aps = torch.stack([
                self.compute_ap(class_id, th) for th in all_thresholds
            ])
            all_aps.append(class_aps.mean())
        metrics['mAP'] = torch.stack(all_aps).mean()

        return metrics


class SetCriterion(nn.Module):
    def __init__(self, args, train_mode=True):
        super(SetCriterion, self).__init__()
        self.matcher = HungarianMatcher(args.classCost, args.bboxCost, args.giouCost)
        self.numClass = args.numClass
        self.classCost = args.classCost
        self.bboxCost = args.bboxCost
        self.giouCost = args.giouCost
        self.train_mode = train_mode

        emptyWeight = torch.ones(args.numClass + 1)
        emptyWeight[-1] = args.eosCost
        self.register_buffer('emptyWeight', emptyWeight)

        # Initialize AP calculator for both training and testing
        self.ap_calculator = APCalculator(self.numClass, args.device)
        self.step_counter = 0


    def forward(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        ans = self.computeLoss(x, y)

        for i, aux in enumerate(x['aux']):
            ans.update({f'{k}_aux{i}': v for k, v in self.computeLoss(aux, y).items()})

        return ans

    def computeLoss(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        if not hasattr(self, 'step_counter'):
            self.step_counter = 0

        ids = self.matcher(x, y)
        idx = self.getPermutationIdx(ids)

        logits = x['class']
        targetClassO = torch.cat([t['labels'] for t, (_, J) in zip(y, ids)])
        targetClass = torch.full(logits.shape[:2], self.numClass, dtype=torch.int64, device=logits.device)
        targetClass[idx] = targetClassO

        classificationLoss = nn.functional.cross_entropy(logits.transpose(1, 2), targetClass, self.emptyWeight)
        classificationLoss *= self.classCost

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

        with torch.no_grad():
            matched_logits = logits[idx]
            softmaxed = nn.functional.softmax(matched_logits, -1)
            confidences, predClass = softmaxed.max(-1)

            if len(boxes) > 0:
                ious = torch.diag(boxIoU(boxCxcywh2Xyxy(boxes), boxCxcywh2Xyxy(targetBoxes))[0])

                if self.step_counter % 100 == 0:
                    print(f"[Step {self.step_counter}] Losses: Class={classificationLoss.item():.4f}, "
                          f"BBox={bboxLoss.item():.4f}, gIoU={giouLoss.item():.4f}")

                self.step_counter += 1

                # Update AP calculator in both training and testing modes
                self.ap_calculator.update(
                    predClass[mask],
                    targetClassO[mask],
                    confidences[mask],
                    ious
                )

                # Compute metrics with per-class AP only in test mode
                ap_metrics = self.ap_calculator.compute_metrics(include_per_class=not self.train_mode)
                metrics.update(ap_metrics)

        return metrics

    @staticmethod
    def getPermutationIdx(indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        batchIdx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        srcIdx = torch.cat([src for (src, _) in indices])
        return batchIdx, srcIdx