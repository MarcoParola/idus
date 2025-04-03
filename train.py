import os
import numpy as np
import pandas as pd
import torch
from torch.cuda import amp
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb
import hydra
import gc
from tqdm import tqdm

from src.datasets.wrappers.only_forgetting_set import OnlyForgettingSet
from src.datasets.wrappers.only_retaining_set import OnlyRetainingSet
from src.datasets.dataset import collateFunction, load_datasets
from src.datasets.wrappers.random_relabelling_set import RandomRelabellingSet
from src.loss.neggrad_criterion import NegGradCriterion, NegGradPlusCriterion
from src.models.object_detection_metrics import ObjectDetectionMetrics
from src.utils.log import log_iou_metrics
from src.loss.base_criterion import BaseCriterion

from src.utils.utils import load_model
from src.utils import cast2Float
from src.utils import EarlyStopping


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(args):
    print("Starting training...")

    wandb.init(entity=args.wandbEntity, project=args.wandbProject, config=dict(args))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Generate filename based on unlearningMethod
    if args.unlearningMethod == "none":
        filename = f"original_{args.model}_{args.dataset}.pt"
    else:
        filename = f"{args.unlearningMethod}_{args.model}_{args.dataset}_{args.unlearningType}_{args.excludeClasses}.pt"

    # Full path to save model
    model_path = os.path.join(args.outputDir, filename)
    print(f"[+] Model will be saved to: {model_path}")

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets(args)

    # Initialize model
    model = load_model(args).to(device)

    # Handle forgetting set (will be null in case of original model training)
    forgetting_set = args.excludeClasses
    unlearning_method = args.unlearningMethod

    # Flag to check if we have forgetting/retaining metrics
    has_forgetting_metrics = forgetting_set is not None and forgetting_set != []

    if unlearning_method != 'none':
        if unlearning_method != 'golden':
            # Load original model for unlearning from ./checkpoints
            model.load_state_dict(torch.load(f"{args.outputDir}/original_{args.model}_{args.dataset}.pt"))

    # Initialize standard validation criterion for always using positive gradients during validation
    validation_criterion = BaseCriterion(args).to(device)
    metrics_calculator = ObjectDetectionMetrics(args).to(device)

    # Configure datasets and criterion based on unlearning method
    if unlearning_method == 'neggrad':
        # Only use forgetting classes for training
        train_dataset = OnlyForgettingSet(train_dataset, forgetting_set, removal=args.unlearningType)
        # Use negative gradients during training
        train_criterion = NegGradCriterion(args).to(device)
    elif unlearning_method == 'neggrad+':
        # Keep all classes during training
        train_criterion = NegGradPlusCriterion(args).to(device)
    else:
        # For other methods, use their respective configurations
        train_criterion = BaseCriterion(args).to(device)
        if unlearning_method == 'golden' or unlearning_method == 'finetuning':
            train_dataset = OnlyRetainingSet(train_dataset, forgetting_set, removal=args.unlearningType)
            args.numClass = len(train_dataset.classes) - (1 if train_dataset.classes[0].lower() == 'background' else 0)
            # Reinitialize model and criteria with updated number of classes
            model = load_model(args).to(device)
            train_criterion = BaseCriterion(args).to(device)
            validation_criterion = BaseCriterion(args).to(device)
            metrics_calculator = ObjectDetectionMetrics(args).to(device)
        elif unlearning_method == 'randomrelabelling':
            train_dataset = RandomRelabellingSet(train_dataset, forgetting_set, removal=args.unlearningType)

    # Create data loaders
    train_dataloader = DataLoader(train_dataset,
                                 batch_size=args.batchSize,
                                 shuffle=True,
                                 collate_fn=collateFunction,
                                 num_workers=args.numWorkers)

    val_dataloader = DataLoader(val_dataset,
                               batch_size=args.batchSize,
                               shuffle=False,
                               collate_fn=collateFunction,
                               num_workers=args.numWorkers)

    test_dataloader = DataLoader(test_dataset,
                                batch_size=1,
                                shuffle=False,
                                collate_fn=collateFunction,
                                num_workers=args.numWorkers)

    with torch.no_grad():
        # Set the background class bias to match DETR's initialization
        bias_value = -torch.log(torch.tensor((1 - 0.01) / 0.01))
        model.class_embed.layers[-1].bias.data[-1] = bias_value

    # Configure optimizer with separate learning rates
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lrBackbone, },
    ]

    early_stopping = EarlyStopping(patience=args.patience)
    optimizer = AdamW(param_dicts, args.lr, weight_decay=args.weightDecay)
    prev_best_loss = np.inf
    batches = len(train_dataloader)
    scaler = amp.GradScaler()

    # Training loop
    for epoch in range(args.epochs):
        # Set model and criterion to training mode
        model.train()
        train_criterion.train()
        validation_criterion.eval()  # Keep validation criterion in eval mode
        metrics_calculator.eval()  # Keep metrics calculator in eval mode

        wandb.log({"epoch": epoch}, step=epoch * batches)
        total_loss = 0.0
        total_metrics = None
        total_eval_metrics = None

        # MARK: - Training
        for batch, (imgs, targets) in enumerate(tqdm(train_dataloader)):
            imgs = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Garbage collection every 700 batches
            if batch % 700 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            # Forward pass
            if args.amp:
                with amp.autocast():
                    out = model(imgs)
                out = cast2Float(out)
            else:
                out = model(imgs)

            # Compute loss using train_criterion (negative gradients for forgetting classes)
            loss_dict = train_criterion(out, targets)

            # Initialize total_metrics on the first batch
            if total_metrics is None:
                total_metrics = {k: 0.0 for k in loss_dict}

            # Accumulate loss values
            for k, v in loss_dict.items():
                total_metrics[k] += v.item()

            # Calculate total loss
            loss = sum(v for k, v in loss_dict.items() if 'loss' in k)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            if args.amp:
                scaler.scale(loss).backward()
                if args.clipMaxNorm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipMaxNorm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.clipMaxNorm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clipMaxNorm)
                optimizer.step()

            # Compute evaluation metrics during training (without affecting gradients)
            with torch.no_grad():
                eval_metrics = metrics_calculator(out, targets)

                # Initialize total_eval_metrics on the first batch
                if total_eval_metrics is None:
                    total_eval_metrics = {k: 0.0 for k in eval_metrics}

                # Accumulate evaluation metrics
                for k, v in eval_metrics.items():
                    total_eval_metrics[k] += v.item()

        # Calculate average loss and metrics
        avg_loss = total_loss / len(train_dataloader)
        avg_metrics = {k: v / len(train_dataloader) for k, v in total_metrics.items()}
        avg_eval_metrics = {k: v / len(train_dataloader) for k, v in total_eval_metrics.items()}

        # Log training losses
        wandb.log({"train/loss": avg_loss}, step=epoch * batches)
        print(f'Epoch {epoch}, loss: {avg_loss:.8f}')

        for k, v in avg_metrics.items():
            wandb.log({f"train/{k}": v}, step=epoch * batches)

        # Log training evaluation metrics
        log_iou_metrics(avg_eval_metrics, epoch * batches, "train", args.numClass, False)

        # MARK: - Validation
        model.eval()
        validation_criterion.eval()  # Use standard positive criterion for validation
        metrics_calculator.eval()

        with torch.no_grad():
            val_losses = []
            val_metrics = []
            val_eval_metrics = []

            for imgs, targets in tqdm(val_dataloader):
                imgs = imgs.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                out = model(imgs)

                # Compute loss using standard validation criterion (positive gradients)
                loss_dict = validation_criterion(out, targets)
                val_metrics.append(loss_dict)
                loss = sum(v for k, v in loss_dict.items() if 'loss' in k)
                val_losses.append(loss.cpu().item())

                # Compute evaluation metrics
                eval_metrics = metrics_calculator(out, targets)
                val_eval_metrics.append(eval_metrics)

            # Compute average validation loss
            val_metrics_dict = {k: torch.stack([m[k] for m in val_metrics]).mean().item() for k in val_metrics[0]}
            avg_val_loss = np.mean(val_losses)

            # Compute average validation evaluation metrics
            val_eval_metrics_dict = {k: torch.stack([m[k] for m in val_eval_metrics]).mean().item() for k in
                                    val_eval_metrics[0]}

            # Log validation losses
            wandb.log({"val/loss": avg_val_loss}, step=epoch * batches)
            for k, v in val_metrics_dict.items():
                wandb.log({f"val/{k}": v}, step=epoch * batches)

            # Log validation evaluation metrics
            log_iou_metrics(val_eval_metrics_dict, epoch * batches, "val", args.numClass, has_forgetting_metrics)

        # MARK: - Save model
        if avg_val_loss < prev_best_loss:
            print('[+] Loss improved from {:.8f} to {:.8f}, saving model...'.format(prev_best_loss, avg_val_loss))

            # Save to local path
            torch.save(model.state_dict(), model_path)
            print(f'[+] Model saved locally to {model_path}')

            # Save to wandb with the same filename
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, filename))
            wandb.save(os.path.join(wandb.run.dir, filename))

            prev_best_loss = avg_val_loss

        # MARK: - Early stopping
        if early_stopping(avg_val_loss):
            print('[+] Early stopping at epoch {}'.format(epoch))
            break

    # MARK: - Testing
    model.eval()
    validation_criterion.eval()  # Use standard positive criterion for testing too
    metrics_calculator.eval()

    # Reset confusion matrix for testing
    metrics_calculator.reset_confusion_matrix()

    with torch.no_grad():
        test_losses = []
        test_metrics = []
        test_eval_metrics = []

        for imgs, targets in tqdm(test_dataloader):
            imgs = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            out = model(imgs)

            # Compute loss using standard validation criterion (positive gradients)
            loss_dict = validation_criterion(out, targets)
            test_metrics.append(loss_dict)
            loss = sum(v for k, v in loss_dict.items() if 'loss' in k)
            test_losses.append(loss.cpu().item())

            # Compute evaluation metrics
            eval_metrics = metrics_calculator(out, targets)
            test_eval_metrics.append(eval_metrics)

        # Compute average test loss
        test_metrics_dict = {k: torch.stack([m[k] for m in test_metrics]).mean().item() for k in test_metrics[0]}
        avg_test_loss = np.mean(test_losses)

        # Compute average test evaluation metrics
        test_eval_metrics_dict = {k: torch.stack([m[k] for m in test_eval_metrics]).mean().item() for k in
                                 test_eval_metrics[0]}

        # Log test losses
        wandb.log({"test/loss": avg_test_loss}, step=epoch * batches)
        for k, v in test_metrics_dict.items():
            wandb.log({f"test/{k}": v}, step=epoch * batches)

        # Log test evaluation metrics
        log_iou_metrics(test_eval_metrics_dict, epoch * batches, "test", args.numClass, has_forgetting_metrics)

        # Create an additional dedicated panel for forgetting/retaining metrics comparison
        if has_forgetting_metrics:
            # Prepare data for a bar chart comparing retaining vs forgetting performance
            comparison_data = {
                "Class Group": ["Retaining", "Forgetting"],
                "mAP": [test_eval_metrics_dict.get('mAP_retaining', 0.0),
                       test_eval_metrics_dict.get('mAP_forgetting', 0.0)],
                "mAP_50": [test_eval_metrics_dict.get('mAP_retaining_50', 0.0),
                          test_eval_metrics_dict.get('mAP_forgetting_50', 0.0)],
                "mAP_75": [test_eval_metrics_dict.get('mAP_retaining_75', 0.0),
                          test_eval_metrics_dict.get('mAP_forgetting_75', 0.0)]
            }

            try:
                # Log a custom bar chart for comparing retaining vs forgetting performance
                table = wandb.Table(dataframe=pd.DataFrame(comparison_data))
                wandb.log({
                    "test/retaining_vs_forgetting": wandb.plot.bar(
                        table, "Class Group",
                        ["mAP", "mAP_50", "mAP_75"],
                        title="Retaining vs Forgetting Classes Performance"
                    )
                }, step=epoch * batches)
            except Exception as e:
                print(f"Error creating retaining vs forgetting comparison chart: {e}")

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
            }, step=epoch * batches)
        except Exception as e:
            print(f"Error logging test confusion matrix with wandb.plot: {e}")
            # Fallback to logging raw confusion matrix
            wandb.log({"test/raw_confusion_matrix": confusion_matrix.cpu().numpy()}, step=epoch * batches)

    # Make sure the final best model is saved again at the end of training
    print(f'[+] Training complete. Best model saved at {model_path}')
    wandb.finish()


if __name__ == '__main__':
    main()