import matplotlib.pyplot as plt
import seaborn as sns
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

def log_iou_metrics(metrics, step, prefix, num_classes):
    """Logs per-class IoU metrics and plots a bar chart."""
    class_ious = [metrics.get(f'IoU_class_{i}', 0).cpu().numpy() for i in range(num_classes)]

    plt.figure(figsize=(10, 5))
    plt.bar(range(num_classes), class_ious)
    plt.xlabel("Class")
    plt.ylabel("IoU")
    plt.title(f"{prefix} IoU per Class")
    plt.xticks(range(num_classes), [f"Class {i}" for i in range(num_classes)], rotation=45)
    plt.tight_layout()

    # Log the image to wandb
    wandb.log({f"{prefix}/IoU_per_class": wandb.Image(plt)}, step=step)
    plt.close()

