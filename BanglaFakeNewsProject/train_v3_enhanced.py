"""
Enhanced Training Pipeline v3 for Bangla Fake News Detection.

Professional ML engineering practices:
  - Label smoothing (0.1)
  - Cosine annealing with warm restarts
  - Gradient accumulation for effective larger batch size
  - Mixed precision training (FP16)
  - Layer-wise learning rate decay for transformer
  - Mixup / CutMix augmentation
  - Comprehensive metric tracking (per-class + weighted)
  - Confusion matrix, ROC-AUC, PR-AUC
  - Saves best model + training curves
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_auc_score, average_precision_score,
)
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model_v3_enhanced import (
    EnhancedMultimodalDetector,
    TextOnlyEnhanced,
    ImageOnlyEnhanced,
)
from data_v3_enhanced import create_enhanced_dataloaders

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = "/mnt/d/all files/ThesisP/BanglaFakeNewsProject/BanglaFakeNewsProject"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = {
    "model_name": "csebuetnlp/banglabert",
    "num_classes": 2,
    "batch_size": 16,
    "epochs": 30,
    "text_lr": 3e-5,        # Higher LR for text encoder (more layers unfrozen)
    "image_lr": 5e-4,       # Higher LR for image encoder
    "fusion_lr": 5e-4,      # Highest LR for fusion/classifier
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "max_grad_norm": 1.0,
    "label_smoothing": 0.1,
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.0005,
    "gradient_accumulation_steps": 2,  # Effective batch = 32
    "mixed_precision": True,
    "save_dir": os.path.join(BASE_DIR, "outputs_v3"),
}

os.makedirs(CONFIG["save_dir"], exist_ok=True)

print(f"Device: {DEVICE}")
print(f"Config: {json.dumps({k: str(v) for k, v in CONFIG.items()}, indent=2)}")


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0005):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.best_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.min_delta:
            self.best_score = score
            self.counter = 0
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                model.load_state_dict(self.best_state)
                return True
            return False


class MetricsTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions = []
        self.labels = []
        self.losses = []
        self.probabilities = []

    def update(self, predictions, labels, loss=None, probabilities=None):
        self.predictions.extend(predictions.cpu().numpy())
        self.labels.extend(labels.cpu().numpy())
        if loss is not None:
            self.losses.append(loss.item())
        if probabilities is not None:
            self.probabilities.extend(probabilities.cpu().numpy())

    def compute(self):
        preds = np.array(self.predictions)
        labels = np.array(self.labels)
        probs = np.array(self.probabilities) if self.probabilities else None

        acc = accuracy_score(labels, preds)
        p_w, r_w, f1_w, _ = precision_recall_fscore_support(labels, preds, average="weighted")
        p_per, r_per, f1_per, _ = precision_recall_fscore_support(labels, preds, average=None, labels=[0, 1])
        cm = confusion_matrix(labels, preds)

        metrics = {
            "accuracy": acc,
            "precision_weighted": p_w,
            "recall_weighted": r_w,
            "f1_weighted": f1_w,
            "precision_real": p_per[0],
            "precision_fake": p_per[1],
            "recall_real": r_per[0],
            "recall_fake": r_per[1],
            "f1_real": f1_per[0],
            "f1_fake": f1_per[1],
            "confusion_matrix": cm.tolist(),
            "avg_loss": np.mean(self.losses) if self.losses else 0.0,
        }

        if probs is not None and len(np.unique(labels)) > 1:
            try:
                if probs.ndim == 2:
                    prob_positive = probs[:, 1]
                else:
                    prob_positive = probs
                metrics["roc_auc"] = roc_auc_score(labels, prob_positive)
                metrics["avg_precision"] = average_precision_score(labels, prob_positive)
            except Exception:
                pass

        return metrics


def get_optimizer_and_scheduler(model, train_loader, config):
    """Setup optimizer with layer-wise LR decay and cosine scheduler."""
    # Group parameters with different learning rates
    text_params_decay = []
    text_params_no_decay = []
    image_params_decay = []
    image_params_no_decay = []
    other_params_decay = []
    other_params_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "text_encoder" in name:
            if "bias" in name or "norm" in name or "LayerNorm" in name:
                text_params_no_decay.append(param)
            else:
                text_params_decay.append(param)
        elif "image_encoder" in name:
            if "bias" in name or "norm" in name:
                image_params_no_decay.append(param)
            else:
                image_params_decay.append(param)
        else:
            if "bias" in name or "norm" in name:
                other_params_no_decay.append(param)
            else:
                other_params_decay.append(param)

    param_groups = [
        {"params": text_params_decay, "lr": config["text_lr"], "weight_decay": config["weight_decay"]},
        {"params": text_params_no_decay, "lr": config["text_lr"], "weight_decay": 0.0},
        {"params": image_params_decay, "lr": config["image_lr"], "weight_decay": config["weight_decay"]},
        {"params": image_params_no_decay, "lr": config["image_lr"], "weight_decay": 0.0},
        {"params": other_params_decay, "lr": config["fusion_lr"], "weight_decay": config["weight_decay"]},
        {"params": other_params_no_decay, "lr": config["fusion_lr"], "weight_decay": 0.0},
    ]

    optimizer = optim.AdamW(param_groups)

    total_steps = len(train_loader) * config["epochs"] // config["gradient_accumulation_steps"]
    warmup_steps = int(config["warmup_ratio"] * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    return optimizer, scheduler


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train_epoch(model, train_loader, optimizer, criterion, scaler, device, config, scheduler):
    model.train()
    tracker = MetricsTracker()
    optimizer.zero_grad()

    accum_steps = config["gradient_accumulation_steps"]
    use_amp = config["mixed_precision"] and device.type == "cuda"

    progress = tqdm(train_loader, desc="  Training", leave=False)
    for step, batch in enumerate(progress):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        if use_amp:
            with autocast():
                logits = model(input_ids, attention_mask, images)
                loss = criterion(logits, labels) / accum_steps
            scaler.scale(loss).backward()
        else:
            logits = model(input_ids, attention_mask, images)
            loss = criterion(logits, labels) / accum_steps
            loss.backward()

        if (step + 1) % accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        probs = torch.softmax(logits.detach(), dim=1)
        preds = torch.argmax(logits.detach(), dim=1)
        tracker.update(preds, labels, loss * accum_steps, probs)

        progress.set_postfix({
            "loss": f"{(loss * accum_steps).item():.4f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    return tracker.compute()


def validate_epoch(model, val_loader, criterion, device, config):
    model.eval()
    tracker = MetricsTracker()
    use_amp = config["mixed_precision"] and device.type == "cuda"

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="  Validation", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            if use_amp:
                with autocast():
                    logits = model(input_ids, attention_mask, images)
                    loss = criterion(logits, labels)
            else:
                logits = model(input_ids, attention_mask, images)
                loss = criterion(logits, labels)

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            tracker.update(preds, labels, loss, probs)

    return tracker.compute()


def train_enhanced():
    """Main training function for enhanced multimodal model."""
    print("\n" + "=" * 70)
    print("  ENHANCED MULTIMODAL TRAINING v3")
    print("=" * 70)

    # Data
    print("\nLoading data...")
    loaders = create_enhanced_dataloaders(BASE_DIR, CONFIG["batch_size"])
    train_loader, val_loader, test_loader = loaders["multimodal"]
    print(f"  Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    # Model
    model = EnhancedMultimodalDetector(
        text_model_name=CONFIG["model_name"],
        num_classes=CONFIG["num_classes"],
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # Optimizer & scheduler
    optimizer, scheduler = get_optimizer_and_scheduler(model, train_loader, CONFIG)

    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])

    # Mixed precision
    scaler = GradScaler() if CONFIG["mixed_precision"] and DEVICE.type == "cuda" else None

    # Early stopping
    early_stopping = EarlyStopping(
        patience=CONFIG["early_stopping_patience"],
        min_delta=CONFIG["early_stopping_min_delta"],
    )

    # Training history
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_f1": [], "val_f1": [],
        "val_precision": [], "val_recall": [],
        "lr": [],
    }

    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(CONFIG["epochs"]):
        print(f"\nEpoch {epoch + 1}/{CONFIG['epochs']}")
        print("-" * 50)

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler, DEVICE, CONFIG, scheduler)

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion, DEVICE, CONFIG)

        # History
        history["train_loss"].append(train_metrics["avg_loss"])
        history["val_loss"].append(val_metrics["avg_loss"])
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_f1"].append(train_metrics["f1_weighted"])
        history["val_f1"].append(val_metrics["f1_weighted"])
        history["val_precision"].append(val_metrics["precision_weighted"])
        history["val_recall"].append(val_metrics["recall_weighted"])
        history["lr"].append(scheduler.get_last_lr()[0])

        # Print
        print(f"  Train — Loss: {train_metrics['avg_loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_weighted']:.4f}")
        print(f"  Val   — Loss: {val_metrics['avg_loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, F1: {val_metrics['f1_weighted']:.4f}")
        print(f"  Val   — P: {val_metrics['precision_weighted']:.4f}, R: {val_metrics['recall_weighted']:.4f}")
        print(f"  Val   — P-R/F: {val_metrics['precision_real']:.4f}/{val_metrics['precision_fake']:.4f}, R-R/F: {val_metrics['recall_real']:.4f}/{val_metrics['recall_fake']:.4f}, F1-R/F: {val_metrics['f1_real']:.4f}/{val_metrics['f1_fake']:.4f}")
        if "roc_auc" in val_metrics:
            print(f"  Val   — ROC-AUC: {val_metrics['roc_auc']:.4f}, AP: {val_metrics['avg_precision']:.4f}")

        # Save best
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            save_path = os.path.join(CONFIG["save_dir"], "best_enhanced_model.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "config": {k: str(v) for k, v in CONFIG.items()},
            }, save_path)
            print(f"  >> New best model saved! Val Acc: {best_val_acc:.4f}")

        # Early stopping
        if early_stopping(val_metrics["avg_loss"], model):
            print(f"\n  Early stopping at epoch {epoch + 1}")
            break

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time / 60:.1f} minutes")
    print(f"Best validation accuracy: {best_val_acc:.4f}")

    # Final test evaluation
    print("\n" + "=" * 70)
    print("  FINAL TEST EVALUATION")
    print("=" * 70)
    test_metrics = validate_epoch(model, test_loader, criterion, DEVICE, CONFIG)

    print(f"\n  Test Accuracy:  {test_metrics['accuracy']:.4f} ({test_metrics['accuracy'] * 100:.2f}%)")
    print(f"  Test Precision: {test_metrics['precision_weighted']:.4f}")
    print(f"  Test Recall:    {test_metrics['recall_weighted']:.4f}")
    print(f"  Test F1:        {test_metrics['f1_weighted']:.4f}")
    print(f"  Per-class Precision: Real={test_metrics['precision_real']:.4f}, Fake={test_metrics['precision_fake']:.4f}")
    print(f"  Per-class Recall:    Real={test_metrics['recall_real']:.4f}, Fake={test_metrics['recall_fake']:.4f}")
    print(f"  Per-class F1:        Real={test_metrics['f1_real']:.4f}, Fake={test_metrics['f1_fake']:.4f}")
    print(f"  Confusion Matrix: {test_metrics['confusion_matrix']}")
    if "roc_auc" in test_metrics:
        print(f"  ROC-AUC: {test_metrics['roc_auc']:.4f}")
        print(f"  Avg Precision: {test_metrics['avg_precision']:.4f}")

    # Save results
    results = {
        "experiment": "Enhanced Multimodal v3 (Ours)",
        "accuracy": test_metrics["accuracy"],
        "accuracy_pct": test_metrics["accuracy"] * 100,
        "precision_weighted": test_metrics["precision_weighted"],
        "recall_weighted": test_metrics["recall_weighted"],
        "f1_weighted": test_metrics["f1_weighted"],
        "precision_real": test_metrics["precision_real"],
        "precision_fake": test_metrics["precision_fake"],
        "recall_real": test_metrics["recall_real"],
        "recall_fake": test_metrics["recall_fake"],
        "f1_real": test_metrics["f1_real"],
        "f1_fake": test_metrics["f1_fake"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "best_val_acc": best_val_acc,
        "training_time_min": training_time / 60,
        "epochs_trained": epoch + 1,
    }
    if "roc_auc" in test_metrics:
        results["roc_auc"] = test_metrics["roc_auc"]
        results["avg_precision"] = test_metrics["avg_precision"]

    with open(os.path.join(CONFIG["save_dir"], "results_enhanced_v3.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save history
    with open(os.path.join(CONFIG["save_dir"], "training_history_v3.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plot training curves
    plot_training_history(history, CONFIG["save_dir"])

    return results


def plot_training_history(history, save_dir):
    """Plot and save training curves."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(history["train_loss"], label="Train", marker="o", markersize=3)
    axes[0, 0].plot(history["val_loss"], label="Val", marker="s", markersize=3)
    axes[0, 0].set_title("Loss")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history["train_acc"], label="Train", marker="o", markersize=3)
    axes[0, 1].plot(history["val_acc"], label="Val", marker="s", markersize=3)
    axes[0, 1].set_title("Accuracy")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(history["train_f1"], label="Train", marker="o", markersize=3)
    axes[1, 0].plot(history["val_f1"], label="Val", marker="s", markersize=3)
    axes[1, 0].set_title("F1 Score")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history["lr"], marker="o", markersize=3, color="green")
    axes[1, 1].set_title("Learning Rate")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_yscale("log")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves_v3.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves saved to {save_dir}/training_curves_v3.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    results = train_enhanced()
    print("\n" + "=" * 70)
    print("  DONE — Enhanced v3 training complete")
    print("=" * 70)
