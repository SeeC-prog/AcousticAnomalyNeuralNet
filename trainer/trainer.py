import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    f1_score,
)
from collections import defaultdict
import itertools


# Utility: Aggregate segment level reconstruction errors per file - after segmentation
def _aggregate_file_metrics(errors, labels, paths):
    """
    Aggregates segment-level errors to file level by averaging.
    Uses 90th percentile instead of mean to reduce small section influence.
    """
    file_errs = defaultdict(list)
    file_labels = {}
    for e, y, p in zip(errors, labels, paths):
        file_errs[p].append(e)
        file_labels[p] = int(y)
    avg_errs = [np.percentile(v, 97) for v in file_errs.values()]
    labels = [file_labels[p] for p in file_errs.keys()]
    return np.array(avg_errs), np.array(labels)


# Trainer - Unsupervised reconstruction
class Trainer:
    """
    Handles training, validation, and testing of the reconstruction
    
    - Trains via MSE reconstruction loss
    - Uses file level aggregated errors for validation and test metrics
    - Supports early stopping and best-checkpoint saving by AUC
    """
    def __init__(
        self,
        model,
        optimizer,
        train_loader,
        val_loader=None,
        scheduler=None,
        device='cpu',
        checkpoint_dir="checkpoints_recon",
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.scheduler = scheduler
        self.device = device
        self.loss_fn = nn.MSELoss() # Reconstruction loss from MSE
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Track performance and early stopping
        self.best_val_auc = 0.0
        self.best_epoch = 0
        self.early_stop_patience = 12
        self.epochs_no_improve = 0
        self.best_checkpoint = self.checkpoint_dir / "best_model.pt"
        
        # Metric history and TensorBoard logging
        self.history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_auc": [], "val_f1": []}
        self.writer = SummaryWriter(log_dir = self.checkpoint_dir / "runs")
        
        # Automatic mixed precision (AMP) utilities for GPU
        self.scaler = GradScaler(enabled = (device.type == "cuda"))

        
    # Training Loop
    def fit(self, num_epochs):
        best_metric = -float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        patience = self.early_stop_patience

        for epoch in range(1, num_epochs + 1):
            
            # Training Phase
            train_loss = self.train_epoch(epoch)
            self.history["train_loss"].append(train_loss)
            self.writer.add_scalar("Loss/Train", train_loss, epoch)

            # Validation Phase
            if self.val_loader:
                val_loss, val_acc, auc, ap, f1, cm = self.validate_epoch(epoch)
                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(val_acc)
                self.history["val_auc"].append(auc)
                self.history["val_f1"].append(f1)

                # TensorBoard metrics
                self.writer.add_scalar("Loss/Val", val_loss, epoch)
                self.writer.add_scalar("Val/AUC", auc, epoch)
                self.writer.add_scalar("Val/AP", ap, epoch)
                self.writer.add_scalar("Val/Acc", val_acc, epoch)
                self.writer.add_scalar("Val/F1", f1, epoch)

                # Scheduler step and LR logging 
                if self.scheduler:
                    lr = self.optimizer.param_groups[0]["lr"]
                    self.writer.add_scalar("LR", lr, epoch)

                # Log confusion matrix
                if cm is not None:
                    self._log_confusion_matrix(cm, epoch)

                # Model checkpointing
                if np.isfinite(auc) and auc > best_metric:
                    best_metric = auc
                    self.best_epoch = epoch
                    self.epochs_no_improve = 0
                    
                    # Save best checkpoint with all relevant metadata
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_auc": auc,
                        "val_loss": val_loss,
                        "best_thresh": getattr(self, "best_thresh", None),
                    }, self.best_checkpoint)
                    print(f"New best AUC {auc:.4f} at epoch {epoch} (checkpoint saved).")
                else:
                    self.epochs_no_improve += 1
                    print(f"No improvement for {self.epochs_no_improve} epoch(s).")

                # Early stopping
                if self.epochs_no_improve >= patience:
                    print(f"Early stopping at epoch {epoch} (no improvement in {patience} epochs).")
                    break
            else:
                # Sae checkpoint periodically if no validation loader
                if epoch % 5 == 0:
                    torch.save(self.model.state_dict(), self.checkpoint_dir / f"epoch_{epoch}.pt")

        self.writer.close()
        print(f"\n Training complete.Best AUC {best_metric:.4f} at epoch {self.best_epoch}.")


    # Singular training epoch
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0

        for inputs, _, _, _ in self.train_loader:
            inputs = inputs.to(self.device)
            self.optimizer.zero_grad()

            # Automatic mixed precision for speed and memory efficiency
            with autocast(device_type = self.device.type, enabled = (self.device.type == "cuda")):
                recon = self.model(inputs)
                loss = self.loss_fn(recon, inputs)
                #weights = torch.linspace(0.5, 1.5, steps = recon.size(-1), device = recon.device)
                #loss = ((recon - inputs) ** 2 * weights).mean()
                
            # Scaled backpropagation for AMP
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)    # hopefully to prevent gradient explosion. 
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler:
                self.scheduler.step()

            running_loss += loss.item() * inputs.size(0)

        avg_loss = running_loss / len(self.train_loader.dataset)
        print(f"Epoch {epoch:03d} | Train recon loss: {avg_loss:.4f}")
        return avg_loss



 
    # Singlar Validation epoch - Metrics
    def validate_epoch(self, epoch):
        self.model.eval()
        seg_errors, seg_labels, seg_paths = [], [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch_idx, (mels, y_anom, _, paths) in enumerate(self.val_loader):
                x = mels.to(self.device)
                recon = self.model(x)
                
                # Compute per segment MSE error
                mse = torch.mean((recon - x) ** 2, dim = [1, 2])

                # Log first sample reconstruction every 5 epochs
                if epoch % 5 == 0 and batch_idx == 0:
                    orig = x[0].detach().cpu().T
                    rec = recon[0].detach().cpu().T
                    diff = (orig - rec).abs()

                    fig, axes = plt.subplots(1, 3, figsize = (9, 3))
                    for ax, data, title in zip(
                        axes, [orig, rec, diff],
                        ["Original", "Reconstruction", "Abs Error"]
                    ):
                        im = ax.imshow(data, aspect = 'auto', origin = 'lower', cmap = 'magma')
                        ax.set_title(title)
                        fig.colorbar(im, ax = ax, fraction = 0.046, pad = 0.04)
                    plt.tight_layout()
                    self.writer.add_figure(f"Reconstructions/Epoch_{epoch}", fig, epoch)
                    plt.close(fig)

                seg_errors.extend(mse.cpu().numpy().tolist())
                seg_labels.extend(y_anom.numpy().tolist())
                seg_paths.extend(paths)
                total_loss += mse.mean().item() * x.size(0)

        avg_loss = total_loss / len(self.val_loader.dataset)
        
        # Aggregates by file to reduce varience from short segments
        file_errors, file_labels = _aggregate_file_metrics(seg_errors, seg_labels, seg_paths)

        # Metric Computation
        auc = ap = acc = f1 = float("nan")
        best_thresh = None
        cm = None

        # Only compute metrics if both normal/abnormal classes exist
        if len(np.unique(file_labels)) >= 2:
            auc = roc_auc_score(file_labels, file_errors)
            ap = average_precision_score(file_labels, file_errors)

            # Select optimal threshold using f1 maximisation
            prec, rec, th = precision_recall_curve(file_labels, file_errors)
            f1s = 2 * prec * rec / (prec + rec + 1e-12)
            best_idx = int(np.nanargmax(f1s)) if len(f1s) > 0 else 0
            
            # Derive threshold from either f1 curve or 95th percentile of normal errors.
            if len(th) > 0 and best_idx < len(th):
                best_thresh = th[best_idx]
            else:
                best_thresh = np.percentile(np.array(file_errors)[np.array(file_labels) == 0], 95)

            # Apply threshold and compute derived metrics
            preds = (np.array(file_errors) > best_thresh).astype(int)
            acc = (preds == np.array(file_labels)).mean()
            f1 = f1s[best_idx] if len(f1s) > 0 else float("nan")
            cm = confusion_matrix(file_labels, preds)


        print(f"Val AUC = {auc:.3f} | AUPR = {ap:.3f} | F1 = {f1:.3f} | Val loss: {avg_loss:.4f} | Epoch {epoch:03d} | Acc: {acc:.3f}")

        # Save best threshold for testing
        if best_thresh is not None:
            self.best_thresh = best_thresh

        return avg_loss, acc, auc, ap, f1, cm


    # Confusion matrix visualisation
    def _log_confusion_matrix(self, cm, epoch, class_names = ["Normal", "Abnormal"]):
        fig, ax = plt.subplots(figsize = (3, 3))
        im = ax.imshow(cm, interpolation = "nearest", cmap = plt.cm.Blues)
        ax.figure.colorbar(im, ax = ax)
        ax.set(
            xticks = np.arange(len(class_names)),
            yticks = np.arange(len(class_names)),
            xticklabels = class_names,
            yticklabels = class_names,
            ylabel = "True Label",
            xlabel = "Predicted Label"
        )
        plt.title(f"Confusion Matrix (Epoch {epoch})")
        
        # Annotate each cell with count
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            ax.text(
                j, i, format(cm[i, j], "d"),
                 ha = "center", va = "center",
                 color = "white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        self.writer.add_figure("ConfusionMatrix", fig, epoch)
        plt.close(fig)


    # Final test evaluation
    def test(self, test_loader, threshold = None):
        """
        Evaluates model on test set using best saved checkpoint
        """
        
        # Load best model
        if self.best_checkpoint.exists():
            checkpoint = torch.load(self.best_checkpoint, map_location = self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            persisted_thresh = checkpoint.get("best_thresh", None)
            print(f"\n Loaded best checkpoint from epoch {checkpoint['epoch']} (Val AUC = {checkpoint['val_auc']:.3f})")
        else:
            checkpoint = {}
            persisted_thresh = None
            print(f"\n No best checkpoint found at {self.best_checkpoint}, using current weights.")
        if threshold is None:
            threshold = persisted_thresh

        # Inference
        self.model.eval()
        seg_errors, seg_labels, seg_paths = [], [], []

        with torch.no_grad():
            for mels, y_anom, _, paths in test_loader:
                x = mels.to(self.device)
                recon = self.model(x)
                mse = torch.mean((recon - x) ** 2, dim = [1, 2])
                seg_errors.extend(mse.cpu().numpy().tolist())
                seg_labels.extend(y_anom.numpy().tolist())
                seg_paths.extend(paths)

        # Aggregate and compute metrics
        file_errors, file_labels = _aggregate_file_metrics(seg_errors, seg_labels, seg_paths)
        auc = roc_auc_score(file_labels, file_errors)
        ap = average_precision_score(file_labels, file_errors)

        # Choose threshold if not provided
        if threshold is None:
            threshold = getattr(self, "best_thresh", np.percentile(file_errors[file_labels == 0], 95))
        preds = (np.array(file_errors) > threshold).astype(int)
        f1 = f1_score(file_labels, preds)
        acc = (preds == np.array(file_labels)).mean()

        print(f"Test AUC = {auc:.3f} | AUPR = {ap:.3f} | F1 = {f1:.3f} | Acc = {acc:.3f} | Th = {threshold:.5f}")
        return {"auc": auc, "aupr": ap, "f1": f1, "acc": acc, "threshold": threshold}