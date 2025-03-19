import os
import numpy as np
import torch
from torch.cuda import amp
from torch.optim import AdamW
from torch.utils.data import DataLoader
import wandb
import hydra
import gc
from tqdm import tqdm

from src.datasets.OnlyForgettingSet import OnlyForgettingSet
from src.datasets.OnlyRetainingSet import OnlyRetainingSet
from src.datasets.dataset import collateFunction, load_datasets
from src.models.ObjectDetectionMetrics import ObjectDetectionMetrics
from src.utils.log import log_iou_metrics
from src.models.BaseCriterion import BaseCriterion

from src.utils.utils import load_model
from src.utils import cast2Float
from src.utils import EarlyStopping


@hydra.main(config_path="config", config_name="config", version_base="1.3")
def main(args):
    print("Starting training...")

    wandb.init(entity=args.wandbEntity, project=args.wandbProject, config=dict(args))
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Generate filename based on runType
    if args.runType == "original":
        filename = f"original_{args.dataset}_{args.model}.pt"
    elif args.runType == "golden":
        filename = f"golden_{args.dataset}_{args.model}_{args.unlearningType}_{args.excludeClasses}.pt"
    else:
        # Default filename if runType is not specified
        filename = f"model_{args.model}_{args.dataset}.pt"

    # Full path to save model
    model_path = os.path.join(args.outputDir, filename)
    print(f"[+] Model will be saved to: {model_path}")

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets(args)

    # Initialize model
    model = load_model(args).to(device)

    # Initialize criterion and metrics
    criterion = BaseCriterion(args).to(device)
    metrics_calculator = ObjectDetectionMetrics(args).to(device)

    # Handle forgetting set (will be null in case of original model training)
    forgetting_set = args.excludeClasses
    unlearning_method = args.unlearningMethod

    if unlearning_method != 'golden':
        # Load original model for unlearning from ./checkpoints (set in config outputDir)
        model.load_state_dict(torch.load(f"{args.outputDir}/original_{args.dataset}_{args.model}.pt"))

    # Configure datasets based on unlearning method
    if unlearning_method == 'golden' or unlearning_method == 'finetuning':
        train_dataset = OnlyRetainingSet(train_dataset, forgetting_set, removal=args.unlearningType)
        val_dataset = OnlyRetainingSet(val_dataset, forgetting_set, removal=args.unlearningType)
        test_dataset = OnlyRetainingSet(test_dataset, forgetting_set, removal=args.unlearningType)
        args.numClass = len(train_dataset.classes) - (1 if train_dataset.classes[0].lower() == 'background' else 0)
        # Reinitialize model and criterion with updated number of classes
        model = load_model(args).to(device)
        criterion = BaseCriterion(args).to(device)
        metrics_calculator = ObjectDetectionMetrics(args).to(device)

    elif unlearning_method == 'randomrelabelling':
        # Use default train_dataset
        # Modify loss function to generate random labels for forgetting set
        pass

    elif unlearning_method == 'neggrad':
        train_dataset = OnlyForgettingSet(train_dataset, forgetting_set, removal=args.unlearningType)
        val_dataset = OnlyForgettingSet(val_dataset, forgetting_set, removal=args.unlearningType)
        test_dataset = OnlyForgettingSet(test_dataset, forgetting_set, removal=args.unlearningType)
        # Use negative loss for forgetting set

    elif unlearning_method == 'neggrad+':
        # Use default train_dataset
        # Use positive loss for retaining set and negative loss for forgetting set
        pass

    elif unlearning_method == 'ours':
        # TODO: Implement custom unlearning method
        pass

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
        criterion.train()
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

            # Compute loss
            loss_dict = criterion(out, targets)

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
        log_iou_metrics(avg_eval_metrics, epoch * batches, "train", args.numClass)
        for k, v in avg_eval_metrics.items():
            wandb.log({f"train_metrics/{k}": v}, step=epoch * batches)

        # MARK: - Validation
        model.eval()
        criterion.eval()
        metrics_calculator.eval()

        with torch.no_grad():
            val_losses = []
            val_metrics = []
            val_eval_metrics = []

            for imgs, targets in tqdm(val_dataloader):
                imgs = imgs.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                out = model(imgs)

                # Compute loss
                loss_dict = criterion(out, targets)
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
            log_iou_metrics(val_eval_metrics_dict, epoch * batches, "val", args.numClass)
            for k, v in val_eval_metrics_dict.items():
                wandb.log({f"val_metrics/{k}": v}, step=epoch * batches)

        # Check if the model is estrnn-yolos, if so, predict and save the first 20 images
        if args.model == 'estrnn-yolos':
            for _i in range(20):
                img, target = val_dataset.__getitem__(_i)
                print(img.shape)

                pred = model.estrnn_enhancer(img.unsqueeze(0))

                print(img.shape, pred.shape)

                # Get first image among the frames and sum it to the prediction
                img = img[0].squeeze().cpu().numpy()
                pred = pred.squeeze().detach().cpu().numpy()
                enhanced_img = img + pred

                # Save both original and predicted images
                from skimage.io import imsave

                # Scale the image to 0-1
                enhanced_img = (enhanced_img - enhanced_img.min()) / (enhanced_img.max() - enhanced_img.min())
                img = (img - img.min()) / (img.max() - img.min())
                # Convert to 0-255
                enhanced_img = (enhanced_img * 255).astype(np.uint8)
                img = (img * 255).astype(np.uint8)
                imsave(f"{wandb.run.dir}/val_epoch{epoch}_img_{_i}.png", enhanced_img)
                imsave(f"{wandb.run.dir}/val_img_{_i}_original.png", img)

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
    criterion.eval()
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

            # Compute loss
            loss_dict = criterion(out, targets)
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
        log_iou_metrics(test_eval_metrics_dict, epoch * batches, "test", args.numClass)
        for k, v in test_eval_metrics_dict.items():
            wandb.log({f"test_metrics/{k}": v}, step=epoch * batches)

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

    # Make sure the final best model is saved again at the end of training
    print(f'[+] Training complete. Best model saved at {model_path}')
    wandb.finish()


if __name__ == '__main__':
    main()