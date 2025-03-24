from src.models.matcher import HungarianMatcher
from src.utils.boxOps import boxCxcywh2Xyxy, boxIoU

from typing import Dict, List, Tuple
import torch
from torch import nn, Tensor


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
        # Track IoU statistics
        self.class_ious = [[] for _ in range(self.num_classes)]
        self.global_ious = []

    def update(self, pred_classes, target_classes, confidences, ious):
        """Update detection statistics using tensor operations"""
        # Ensure all inputs are on the correct device and have the same size
        pred_classes = pred_classes.to(self.device)
        target_classes = target_classes.to(self.device)
        confidences = confidences.to(self.device)
        ious = ious.to(self.device)

        # Safety check to ensure all tensors have compatible sizes (had problems)
        if len(pred_classes) != len(target_classes) or len(pred_classes) != len(confidences) or len(
                pred_classes) != len(ious):
            print(f"Warning: Mismatched tensor sizes - pred_classes: {pred_classes.shape}, "
                  f"target_classes: {target_classes.shape}, confidences: {confidences.shape}, ious: {ious.shape}")
            # Use the minimum length to ensure compatibility
            min_len = min(len(pred_classes), len(target_classes), len(confidences), len(ious))
            pred_classes = pred_classes[:min_len]
            target_classes = target_classes[:min_len]
            confidences = confidences[:min_len]
            ious = ious[:min_len]

        # Update ground truth counts
        unique_classes, class_counts = torch.unique(target_classes, return_counts=True)
        for cls, count in zip(unique_classes, class_counts):
            # Ignore background class
            if cls != self.num_classes:
                self.gt_counts[cls] += count

        # Store predictions as tensors
        mask = pred_classes != self.num_classes
        self.predictions['class_ids'].append(pred_classes[mask])
        self.predictions['confidences'].append(confidences[mask])
        self.predictions['is_correct'].append(pred_classes[mask] == target_classes[mask])
        self.predictions['ious'].append(ious[mask])

        # Track IoUs per class and globally
        self.global_ious.extend(ious[mask].tolist())

        # Update per-class IoUs
        for cls in range(self.num_classes):
            class_mask = (pred_classes == cls) & (target_classes == cls)
            if class_mask.any():
                self.class_ious[cls].extend(ious[class_mask].tolist())

    def compute_ap(self, class_id, threshold):
        """Compute AP for a specific class and IoU threshold"""
        if not self.predictions['class_ids']:
            return torch.tensor(0.0, device=self.device)

        try:
            # Concatenate all predictions - with error handling
            class_ids = torch.cat(self.predictions['class_ids'])
            confidences = torch.cat(self.predictions['confidences'])
            is_correct = torch.cat(self.predictions['is_correct'])
            ious = torch.cat(self.predictions['ious'])

            # Safety check for tensor shapes
            if not (len(class_ids) == len(confidences) == len(is_correct) == len(ious)):
                print(f"Warning: Mismatched tensor sizes after concatenation - class_ids: {class_ids.shape}, "
                      f"confidences: {confidences.shape}, is_correct: {is_correct.shape}, ious: {ious.shape}")
                return torch.tensor(0.0, device=self.device)

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

        except Exception as e:
            print(f"Error in compute_ap for class {class_id} at threshold {threshold}: {str(e)}")
            return torch.tensor(0.0, device=self.device)

    def compute_metrics(self):
        """Compute AP metrics for all classes and thresholds"""
        metrics = {}

        try:
            # Initialize tensors for storing APs
            class_aps = torch.zeros((self.num_classes, 3), device=self.device)  # 3 thresholds: 0.5, 0.75, 0.95

            # Compute AP for each class and threshold
            for class_id in range(self.num_classes):
                # Compute AP for standard thresholds
                class_aps[class_id, 0] = self.compute_ap(class_id, 0.5)  # AP@50
                class_aps[class_id, 1] = self.compute_ap(class_id, 0.75)  # AP@75
                class_aps[class_id, 2] = self.compute_ap(class_id, 0.95)  # AP@95

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

                # Compute and add IoU for this class
                if self.class_ious[class_id]:
                    metrics[f'IoU_class_{class_id}'] = torch.tensor(
                        sum(self.class_ious[class_id]) / len(self.class_ious[class_id]),
                        device=self.device
                    )
                else:
                    metrics[f'IoU_class_{class_id}'] = torch.tensor(0.0, device=self.device)

            # Compute mAP metrics
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

            # Compute global (mean) IoU across all predictions
            if self.global_ious:
                metrics['mIoU'] = torch.tensor(
                    sum(self.global_ious) / len(self.global_ious),
                    device=self.device
                )
            else:
                metrics['mIoU'] = torch.tensor(0.0, device=self.device)

        except Exception as e:
            print(f"Error in compute_metrics: {str(e)}")
            metrics = self._create_empty_metrics()

        return metrics

    def _create_empty_metrics(self):
        """Create zero-filled metrics when there are no predictions"""
        metrics = {
            'mAP': torch.tensor(0.0, device=self.device),
            'mIoU': torch.tensor(0.0, device=self.device)
        }
        for threshold in [50, 75, 95]:
            metrics[f'mAP_{threshold}'] = torch.tensor(0.0, device=self.device)
            for c in range(self.num_classes):
                metrics[f'AP_class_{c}_{threshold}'] = torch.tensor(0.0, device=self.device)

        for c in range(self.num_classes):
            metrics[f'AP_class_{c}'] = torch.tensor(0.0, device=self.device)
            metrics[f'IoU_class_{c}'] = torch.tensor(0.0, device=self.device)

        return metrics


class ObjectDetectionMetrics(nn.Module):
    def __init__(self, args):
        super(ObjectDetectionMetrics, self).__init__()
        self.matcher = HungarianMatcher(args.classCost, args.bboxCost, args.giouCost)
        self.numClass = args.numClass
        self.ap_calculator = APCalculator(self.numClass, args.device)
        self.step_counter = 0
        self.confusion_matrix = None
        self.device = args.device

    def reset_confusion_matrix(self):
        """Initialize or reset the confusion matrix"""
        # Include background class in confusion matrix (+1)
        self.confusion_matrix = torch.zeros((self.numClass + 1, self.numClass + 1), dtype=torch.long,
                                            device=self.device)

    def update_confusion_matrix(self, pred_classes, target_classes):
        """Update confusion matrix with batch predictions"""
        if self.confusion_matrix is None:
            self.reset_confusion_matrix()

        for p, t in zip(pred_classes, target_classes):
            self.confusion_matrix[p, t] += 1

    def get_confusion_matrix(self):
        """Return the current confusion matrix"""
        if self.confusion_matrix is None:
            self.reset_confusion_matrix()
        return self.confusion_matrix.clone()

    def forward(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        try:
            metrics = self.compute_metrics(x, y)

            # Handle auxiliary predictions if present
            for i, aux in enumerate(x['aux']):
                try:
                    aux_metrics = self.compute_metrics(aux, y)
                    metrics.update({f'{k}_aux{i}': v for k, v in aux_metrics.items()})
                except Exception as e:
                    print(f"Error computing auxiliary metrics {i}: {str(e)}")

            return metrics

        except Exception as e:
            print(f"Error in metrics forward pass: {str(e)}")
            # Return empty metrics
            return {
                'mAP': torch.tensor(0.0, device=x['class'].device),
                'mIoU': torch.tensor(0.0, device=x['class'].device)
            }

    def compute_metrics(self, x: Dict[str, Tensor], y: List[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        try:
            if not hasattr(self, 'step_counter'):
                self.step_counter = 0

            ids = self.matcher(x, y)
            idx = self.get_permutation_idx(ids)

            # Get matched predictions and targets
            logits = x['class']
            targetClassO = torch.cat([t['labels'] for t, (_, J) in zip(y, ids)])

            # Get boxes for matched predictions
            mask = targetClassO != self.numClass
            boxes = x['bbox'][idx][mask]
            targetBoxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(y, ids)], 0)[mask]

            metrics = {}

            with torch.no_grad():
                matched_logits = logits[idx]
                softmaxed = nn.functional.softmax(matched_logits, -1)
                confidences, predClass = softmaxed.max(-1)

                if len(boxes) > 0:
                    ious = torch.diag(boxIoU(boxCxcywh2Xyxy(boxes), boxCxcywh2Xyxy(targetBoxes))[0])

                    # Debug print every 100 steps
                    if self.step_counter % 1000 == 0:
                        print(f"[Step {self.step_counter}] Metrics calculation")
                        print(f"Pred: {predClass[:10].tolist()}... | Target: {targetClassO[:10].tolist()}...")
                        print(f"Confidences: {confidences[:10].tolist()}... | IoUs: {ious[:10].tolist()}...")

                    # Update confusion matrix for matched classes
                    if mask.any():
                        self.update_confusion_matrix(predClass[mask].detach(), targetClassO[mask].detach())

                    self.step_counter += 1

                    try:
                        # Make sure all tensors have the same size before updating AP calculator
                        if len(predClass[mask]) == len(targetClassO[mask]) == len(confidences[mask]) == len(ious):
                            # Update AP calculator with batch statistics
                            self.ap_calculator.update(
                                predClass[mask],
                                targetClassO[mask],
                                confidences[mask],
                                ious
                            )

                            # Calculate all metrics at once
                            metrics = self.ap_calculator.compute_metrics()
                        else:
                            print(f"Warning: Skipping AP calculation due to size mismatch - "
                                  f"pred: {predClass[mask].shape}, target: {targetClassO[mask].shape}, "
                                  f"conf: {confidences[mask].shape}, ious: {ious.shape}")
                            # Add empty metrics
                            metrics = self.ap_calculator._create_empty_metrics()
                    except Exception as e:
                        print(f"Error in AP calculation: {str(e)}")
                        # Add empty metrics
                        metrics = self.ap_calculator._create_empty_metrics()
                else:
                    # Create empty metrics when there are no valid boxes
                    metrics = self.ap_calculator._create_empty_metrics()

            return metrics

        except Exception as e:
            print(f"Error in compute_metrics: {str(e)}")
            # Return empty metrics
            device = x['class'].device if 'class' in x else 'cpu'
            return {
                'mAP': torch.tensor(0.0, device=device),
                'mIoU': torch.tensor(0.0, device=device)
            }

    @staticmethod
    def get_permutation_idx(indices: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor]:
        batchIdx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        srcIdx = torch.cat([src for (src, _) in indices])
        return batchIdx, srcIdx