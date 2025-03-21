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

from src.datasets.OnlyRetainingSet import OnlyRetainingSet
from src.datasets.dataset import collateFunction, load_datasets
from src.utils.log import log_iou_metrics
from src.models import SetCriterion
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

    # load data
    train_dataset, val_dataset, test_dataset= load_datasets(args)

    # set model and criterion, load weights if available
    model = load_model(args).to(device)
    criterion = SetCriterion(args).to(device)

    # recupero il forgetting set (che sarà nullo nel caso di original model training)
    forgetting_set = args.excludeClasses
    unlearning_method = args.unlearningMethod

    if unlearning_method != 'golden':
        # se non è golden, carico il modello originale per fare unlearning da ./checkpoints che è già in config outputDir
        model.load_state_dict(torch.load(f"{args.outputDir}/original_{args.dataset}.pt"))

    if unlearning_method == 'golden' or unlearning_method == 'finetuning':
        train_dataset = OnlyRetainingSet(train_dataset, forgetting_set, removal=args.unlearningType)
        val_dataset = OnlyRetainingSet(val_dataset, forgetting_set, removal=args.unlearningType)
        test_dataset = OnlyRetainingSet(test_dataset, forgetting_set, removal=args.unlearningType)
        args.numClass = len(train_dataset.classes) - (1 if train_dataset.classes[0].lower() == 'background' else 0)
        model = load_model(args).to(device)
        criterion = SetCriterion(args).to(device)

    elif unlearning_method == 'randomrelabelling':
        # il train_dataset non viene modificato
        # modifico la loss che è una crossentropy che prende come argomento il forgetting set, così ogni volta che calcola la loss per i sample del forgetting set, genera una label random
        # criterion = ....
        pass

    elif unlearning_method == 'neggrad':
        train_dataset = OnlyRetainingSet(train_dataset, forgetting_set, removal=args.unlearningType)
        val_dataset = OnlyRetainingSet(val_dataset, forgetting_set, removal=args.unlearningType)
        test_dataset = OnlyRetainingSet(test_dataset, forgetting_set, removal=args.unlearningType)
        args.numClass = len(train_dataset.classes) - (1 if train_dataset.classes[0].lower() == 'background' else 0)
        model = load_model(args).to(device)
        # criterion = .... -> loss negativa

    elif unlearning_method == 'neggrad+':
        # in questo caso il dataset rimane uguale (tutto)
        # criterion = neggradplusloss -> calcola loss positiva su retaining set e negativa su forgetting set
        pass
    
    elif unlearning_method == 'ours':
        # TODO
        pass

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

    # separate learning rate
    paramDicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
         "lr": args.lrBackbone, },
    ]

    early_stopping = EarlyStopping(patience=args.patience)
    optimizer = AdamW(paramDicts, args.lr, weight_decay=args.weightDecay)
    prevBestLoss = np.inf
    batches = len(train_dataloader)
    scaler = amp.GradScaler()
    model.train()
    criterion.train()

    for epoch in range(args.epochs):
        wandb.log({"epoch": epoch}, step=epoch * batches)
        total_loss = 0.0
        total_metrics = None  # Initialize total_metrics

        # MARK: - training
        for batch, (imgs, targets) in enumerate(tqdm(train_dataloader)):
            imgs = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            # gc every 50 batches
            if batch % 700 == 0:
                torch.cuda.empty_cache()
                gc.collect()

            if args.amp:
                with amp.autocast():
                    out = model(imgs)
                out = cast2Float(out)
            else:
                out = model(imgs)

            metrics = criterion(out, targets)
            # Initialize total_metrics on the first batch
            if total_metrics is None:
                total_metrics = {k: 0.0 for k in metrics}

            # Calculate mean values progressively
            for k, v in metrics.items():
                total_metrics[k] += v.item()

            loss = sum(v for k, v in metrics.items() if 'loss' in k)
            total_loss += loss.item()

            # MARK: - backpropagation
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

        # Calculate average loss and metrics
        avg_loss = total_loss / len(train_dataloader)
        avg_metrics = {k: v / len(train_dataloader) for k, v in total_metrics.items()}

        # Log IoU metrics
        log_iou_metrics(avg_metrics, epoch * batches, "train", args.numClass)

        wandb.log({"train/loss": avg_loss}, step=epoch * batches)
        print(f'Epoch {epoch}, loss: {avg_loss:.8f}')

        for k, v in avg_metrics.items():
            wandb.log({f"train/{k}": v}, step=epoch * batches)

        # MARK: - validation
        model.eval()
        criterion.eval()

        with torch.no_grad():
            valMetrics = []
            losses = []
            for imgs, targets in tqdm(val_dataloader):
                imgs = imgs.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                out = model(imgs)

                metrics = criterion(out, targets)
                valMetrics.append(metrics)
                loss = sum(v for k, v in metrics.items() if 'loss' in k)
                losses.append(loss.cpu().item())

            # Compute average validation metrics
            valMetricsDict = {k: torch.stack([m[k] for m in valMetrics]).mean().item() for k in valMetrics[0]}

            # Log validation IoU metrics
            log_iou_metrics(valMetricsDict, epoch * batches, "val", args.numClass)

            valMetrics = {k: torch.stack([m[k] for m in valMetrics]).mean() for k in valMetrics[0]}
            avgLoss = np.mean(losses)
            wandb.log({"val/loss": avgLoss}, step=epoch * batches)
            for k, v in valMetrics.items():
                wandb.log({f"val/{k}": v.item()}, step=epoch * batches)

        # check if the model is estrnn-yolos, if so, predict the first 10 images of the val set
        if args.model == 'estrnn-yolos':
            for _i in range(20):
                img, target = val_dataset.__getitem__(_i)
                print(img.shape)

                pred = model.estrnn_enhancer(img.unsqueeze(0))

                print(img.shape, pred.shape)

                # get first image among the frames and sum it to the prediction
                img = img[0].squeeze().cpu().numpy()
                pred = pred.squeeze().detach().cpu().numpy()
                enhanced_img = img + pred

                # save both original and predicted images
                from skimage.io import imsave

                # scale the image to 0-1
                enhanced_img = (enhanced_img - enhanced_img.min()) / (enhanced_img.max() - enhanced_img.min())
                img = (img - img.min()) / (img.max() - img.min())
                # convert to 0-255
                enhanced_img = (enhanced_img * 255).astype(np.uint8)
                img = (img * 255).astype(np.uint8)
                imsave(f"{wandb.run.dir}/val_epoch{epoch}_img_{_i}.png", enhanced_img)
                imsave(f"{wandb.run.dir}/val_img_{_i}_original.png", img)

        model.train()
        criterion.train()

        # MARK: - save model
        if avgLoss < prevBestLoss:
            print('[+] Loss improved from {:.8f} to {:.8f}, saving model...'.format(prevBestLoss, avgLoss))

            # Save to local path
            torch.save(model.state_dict(), model_path)
            print(f'[+] Model saved locally to {model_path}')

            # Save to wandb with the same filename
            torch.save(model.state_dict(), os.path.join(wandb.run.dir, filename))
            wandb.save(os.path.join(wandb.run.dir, filename))

            prevBestLoss = avgLoss

        # MARK: - early stopping
        if early_stopping(avgLoss):
            print('[+] Early stopping at epoch {}'.format(epoch))
            break

    # MARK: - testing
    model.eval()
    criterion.eval()
    # Reset confusion matrix for testing
    criterion.reset_confusion_matrix()

    with torch.no_grad():
        valMetrics = []
        losses = []
        for imgs, targets in tqdm(test_dataloader):
            imgs = imgs.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            out = model(imgs)

            metrics = criterion(out, targets)
            valMetrics.append(metrics)
            loss = sum(v for k, v in metrics.items() if 'loss' in k)
            losses.append(loss.cpu().item())

        # Compute average test metrics
        testMetricsDict = {k: torch.stack([m[k] for m in valMetrics]).mean().item() for k in valMetrics[0]}

        # Log test IoU metrics
        log_iou_metrics(testMetricsDict, epoch * batches, "test", args.numClass)

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

        valMetrics = {k: torch.stack([m[k] for m in valMetrics]).mean() for k in valMetrics[0]}
        avgLoss = np.mean(losses)
        wandb.log({"test/loss": avgLoss}, step=epoch * batches)
        for k, v in valMetrics.items():
            wandb.log({f"test/{k}": v.item()}, step=epoch * batches)

    # Make sure the final best model is saved again at the end of training
    print(f'[+] Training complete. Best model saved at {model_path}')
    wandb.finish()


if __name__ == '__main__':
    main()
