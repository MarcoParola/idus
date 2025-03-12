import os
import torch
from torch.utils.data import DataLoader
import wandb
import hydra
from tqdm import tqdm
from collections import defaultdict

from src.datasets.dataset import load_datasets, collateFunction
from src.models import SetCriterion
from src.utils.log import log_confusion_matrix, log_iou_metrics
from src.utils.utils import load_model
from src.utils import cast2Float


def accumulate_metrics(metrics_list):
    """Accumulate metrics across batches properly."""
    accumulated = defaultdict(list)

    for metrics in metrics_list:
        for k, v in metrics.items():
            accumulated[k].append(v)

    # Average the accumulated metrics
    return {k: torch.stack(v).mean() for k, v in accumulated.items()}


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(args):
    args.wandbProject = args.wandbProject + '_eval'
    wandb.init(entity=args.wandbEntity, project=args.wandbProject, config=dict(args))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.outputDir, exist_ok=True)

    # Load datasets
    _, _, test_dataset, actual_num_classes = load_datasets(args)

    # Update class count
    args.numClass = actual_num_classes

    test_dataloader = DataLoader(test_dataset,
                                 batch_size=32,
                                 shuffle=False,
                                 collate_fn=collateFunction,
                                 num_workers=args.numWorkers)

    # Load model and criterion
    criterion = SetCriterion(args).to(device)
    model = load_model(args).to(device)

    if args.multi:
        model = torch.nn.DataParallel(model)

    model.eval()
    criterion.eval()
    criterion.reset_confusion_matrix()  # Reset confusion matrix before testing

    with torch.no_grad():
        all_metrics = []
        total_predictions = 0
        class_counts = torch.zeros(args.numClass + 1, device=device)

        # Iterate over test dataset
        for batch, (imgs, targets) in enumerate(tqdm(test_dataloader)):
            imgs = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(imgs)
            outputs = cast2Float(outputs)  # Ensure consistency in precision

            # Compute metrics
            metrics = criterion(outputs, targets)
            all_metrics.append(metrics)

            # Update class prediction counts
            pred_classes = outputs['class'].argmax(-1)
            for c in range(args.numClass + 1):
                class_counts[c] += (pred_classes == c).sum().item()

            total_predictions += pred_classes.numel()

            # Clear cache periodically
            if batch % 10 == 0:
                torch.cuda.empty_cache()

        # Aggregate final metrics
        final_metrics = accumulate_metrics(all_metrics)
        test_conf_matrix = criterion.get_confusion_matrix().cpu().numpy()

        # Prepare class labels
        class_labels = [f"class_{i}" for i in range(args.numClass)] + ["background"]

        # Log confusion matrix
        log_confusion_matrix(test_conf_matrix, class_labels, step=0, prefix="test")

        # Log IoU metrics
        log_iou_metrics(final_metrics, step=0, prefix="test", num_classes=args.numClass)

        # Debugging information
        print("\n=== Debugging Information ===")
        print(f"Total batches processed: {len(test_dataloader)}")
        print(f"Total predictions: {total_predictions}")
        print("\nClass distribution in predictions:")
        for c in range(args.numClass + 1):
            count = class_counts[c].item()
            percentage = (count / total_predictions) * 100
            print(f"Class {c}: {count} ({percentage:.2f}%)")

        print("\nFinal Metrics:")
        for k, v in final_metrics.items():
            if k.startswith('mAP') or k.startswith('AP_class_'):
                print(f"{k}: {v.item():.4f}")

        # Log metrics to wandb
        for k, v in final_metrics.items():
            wandb.log({f"test/{k}": v.item()}, step=0)

    wandb.finish()


if __name__ == '__main__':
    main()
