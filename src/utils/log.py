import matplotlib.pyplot as plt
import seaborn as sns
import torch
import wandb

def log_confusion_matrix(conf_matrix, class_labels, step, prefix="train"):
    """Helper function to create and log confusion matrix visualizations."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{prefix.capitalize()} Confusion Matrix')

    # Log the figure as an image
    wandb.log({f"{prefix}/confusion_matrix_heatmap": wandb.Image(plt)}, step=step)
    plt.close()


# Updated log_iou_metrics function to handle forgetting and retaining metrics
def log_iou_metrics(metrics_dict, step, prefix, num_classes, has_forgetting=False):
    """
    Log IoU and AP metrics to wandb with improved organization

    Args:
        metrics_dict: Dictionary containing metrics
        step: Current step for logging
        prefix: Prefix for the metric name (train/val/test)
        num_classes: Number of classes in the dataset
        has_forgetting: Whether forgetting/retaining metrics are available
    """
    # Log overall metrics
    global_metrics = {
        'mAP': metrics_dict.get('mAP', 0.0),
        'mAP_50': metrics_dict.get('mAP_50', 0.0),
        'mAP_75': metrics_dict.get('mAP_75', 0.0),
        'mAP_95': metrics_dict.get('mAP_95', 0.0),
        'mIoU': metrics_dict.get('mIoU', 0.0)
    }

    for k, v in global_metrics.items():
        wandb.log({f"{prefix}/{k}": v}, step=step)

    # Log per-class metrics
    for c in range(num_classes):
        per_class_metrics = {
            f'AP_class_{c}': metrics_dict.get(f'AP_class_{c}', 0.0),
            f'AP_class_{c}_50': metrics_dict.get(f'AP_class_{c}_50', 0.0),
            f'AP_class_{c}_75': metrics_dict.get(f'AP_class_{c}_75', 0.0),
            f'AP_class_{c}_95': metrics_dict.get(f'AP_class_{c}_95', 0.0),
            f'IoU_class_{c}': metrics_dict.get(f'IoU_class_{c}', 0.0)
        }

        for k, v in per_class_metrics.items():
            wandb.log({f"{prefix}/classes/{k}": v}, step=step)

    # Log forgetting and retaining metrics if available
    if has_forgetting:
        forgetting_metrics = {
            'mAP_forgetting': metrics_dict.get('mAP_forgetting', 0.0),
            'mAP_forgetting_50': metrics_dict.get('mAP_forgetting_50', 0.0),
            'mAP_forgetting_75': metrics_dict.get('mAP_forgetting_75', 0.0),
            'mAP_forgetting_95': metrics_dict.get('mAP_forgetting_95', 0.0),
            'mAP_retaining': metrics_dict.get('mAP_retaining', 0.0),
            'mAP_retaining_50': metrics_dict.get('mAP_retaining_50', 0.0),
            'mAP_retaining_75': metrics_dict.get('mAP_retaining_75', 0.0),
            'mAP_retaining_95': metrics_dict.get('mAP_retaining_95', 0.0)
        }

        for k, v in forgetting_metrics.items():
            wandb.log({f"{prefix}/forgetting/{k}": v}, step=step)

