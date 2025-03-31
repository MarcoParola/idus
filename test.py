import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
import hydra
from tqdm import tqdm
import gc

from src.datasets.wrappers.only_forgetting_set import OnlyForgettingSet
from src.datasets.wrappers.only_retaining_set import OnlyRetainingSet
from src.datasets.dataset import load_datasets, collateFunction
from src.loss.neggrad_criterion import NegGradCriterion, NegGradPlusCriterion
from src.models.RandomRelabellingCriterion import RandomRelabellingCriterion
from src.models.object_detection_metrics import ObjectDetectionMetrics
from src.loss.base_criterion import BaseCriterion
from src.utils.log import log_iou_metrics
from src.utils.utils import load_model
from src.utils import cast2Float


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(args):
    args.wandbProject = args.wandbProject + '_eval'
    wandb.init(entity=args.wandbEntity, project=args.wandbProject, config=dict(args))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.outputDir, exist_ok=True)

    # Generate filename based on unlearningMethod (same as training script)
    if args.unlearningMethod == "none":
        filename = f"original_{args.dataset}_{args.model}.pt"
    elif args.unlearningMethod == "golden":
        filename = f"golden_{args.dataset}_{args.model}_{args.unlearningType}_{args.excludeClasses}.pt"
    else:
        # Default filename if unlearningMethod is not specified
        filename = f"model_{args.model}_{args.dataset}.pt"

    # Full path to load model
    model_path = os.path.join(args.outputDir, filename)
    print(f"[+] Loading model from: {model_path}")

    # Load datasets
    _, _, test_dataset = load_datasets(args)

    # Handle forgetting set (will be null in case of original model evaluation)
    forgetting_set = args.excludeClasses
    unlearning_method = args.unlearningMethod

    # Configure datasets based on unlearning method (same logic as training script)
    if unlearning_method == 'golden' or unlearning_method == 'finetuning':
        test_dataset = OnlyRetainingSet(test_dataset, forgetting_set, removal=args.unlearningType)
        args.numClass = len(test_dataset.classes) - (1 if test_dataset.classes[0].lower() == 'background' else 0)

    # Initialize model and criterion
    model = load_model(args).to(device)

    # Initialize criterion based on unlearning method
    if unlearning_method == 'randomrelabelling':
        criterion = RandomRelabellingCriterion(args).to(device)
    elif unlearning_method == 'neggrad':
        test_dataset = OnlyForgettingSet(test_dataset, forgetting_set, removal=args.unlearningType)
        criterion = NegGradCriterion(args).to(device)
    elif unlearning_method == 'neggrad+':
        criterion = NegGradPlusCriterion(args).to(device)
    elif unlearning_method == 'ours':
        # TODO: Implement custom unlearning method
        pass
    else:  # 'none', 'golden', or default
        criterion = BaseCriterion(args).to(device)

    # Initialize metrics calculator
    metrics_calculator = ObjectDetectionMetrics(args).to(device)

    # Load the trained model
    model.load_state_dict(torch.load(model_path))
    print(f"[+] Successfully loaded model from {model_path}")

    if args.multi:
        model = torch.nn.DataParallel(model)

    # Create test dataloader
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=args.batchSize,
                                 shuffle=False,
                                 collate_fn=collateFunction,
                                 num_workers=args.numWorkers)

    # Set model and criterion to evaluation mode
    model.eval()
    criterion.eval()
    metrics_calculator.eval()
    metrics_calculator.reset_confusion_matrix()  # Reset confusion matrix before testing

    with torch.no_grad():
        test_losses = []
        test_metrics = []
        test_eval_metrics = []
        total_predictions = 0
        class_counts = torch.zeros(args.numClass + 1, device=device)

        # Iterate over test dataset
        for batch, (imgs, targets) in enumerate(tqdm(test_dataloader)):
            imgs = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Perform garbage collection periodically
            if batch % 100 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            # Forward pass
            out = model(imgs)
            out = cast2Float(out)

            # Compute loss
            loss_dict = criterion(out, targets)
            test_metrics.append(loss_dict)
            loss = sum(v for k, v in loss_dict.items() if 'loss' in k)
            test_losses.append(loss.cpu().item())

            # Compute evaluation metrics
            eval_metrics = metrics_calculator(out, targets)
            test_eval_metrics.append(eval_metrics)

            # Update class prediction counts
            pred_classes = out['class'].argmax(-1)
            for c in range(args.numClass + 1):
                class_counts[c] += (pred_classes == c).sum().item()

            total_predictions += pred_classes.numel()

        # Compute average test loss
        test_metrics_dict = {k: torch.stack([m[k] for m in test_metrics]).mean().item() for k in test_metrics[0]}
        avg_test_loss = np.mean(test_losses)

        # Compute average test evaluation metrics
        test_eval_metrics_dict = {k: torch.stack([m[k] for m in test_eval_metrics]).mean().item() for k in
                                  test_eval_metrics[0]}

        # Log test losses
        wandb.log({"test/loss": avg_test_loss})
        for k, v in test_metrics_dict.items():
            wandb.log({f"test/{k}": v})

        # Log test evaluation metrics
        log_iou_metrics(test_eval_metrics_dict, 0, "test", args.numClass)
        for k, v in test_eval_metrics_dict.items():
            wandb.log({f"test_metrics/{k}": v})

        # Get and log confusion matrix
        confusion_matrix = metrics_calculator.get_confusion_matrix()

        # Create class labels for confusion matrix
        class_labels = [f"class_{i}" for i in range(args.numClass)] + ["background"]

        # Get indices for true and predicted classes
        true_classes = np.arange(len(class_labels))
        pred_classes = np.arange(len(class_labels))

        # Use wandb's built-in confusion matrix function if possible
        try:
            wandb.log({
                "test/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=true_classes,
                    preds=pred_classes,
                    class_names=class_labels,
                    title="Test Confusion Matrix"
                )
            })
        except Exception as e:
            print(f"Error logging test confusion matrix with wandb.plot: {e}")

        # Print debugging information
        print("\n=== Evaluation Results ===")
        print(f"Total batches processed: {len(test_dataloader)}")
        print(f"Total predictions: {total_predictions}")
        print("\nClass distribution in predictions:")
        for c in range(args.numClass + 1):
            count = class_counts[c].item()
            percentage = (count / total_predictions) * 100
            print(f"Class {c}: {count} ({percentage:.2f}%)")

        print("\nFinal Metrics:")
        for k, v in test_eval_metrics_dict.items():
            if k.startswith('mAP') or k.startswith('AP_class_'):
                print(f"{k}: {v:.4f}")

        print(f"\n[+] Evaluation complete for model: {model_path}")

    wandb.finish()


if __name__ == '__main__':
    main()