"""Evaluation and plotting utilities for SER."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from utils import ensure_dir


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    output_dir: str,
) -> Dict[str, object]:
    """Evaluate model and save confusion matrix + report."""
    ensure_dir(output_dir)
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []

    with torch.no_grad():
        for input_values, labels in dataloader:
            input_values = input_values.to(device)
            logits = model(input_values)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")

    target_names = [id2label[i] for i in sorted(id2label.keys())]
    report_text = classification_report(
        all_labels,
        all_preds,
        labels=sorted(id2label.keys()),
        target_names=target_names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(all_labels, all_preds, labels=sorted(id2label.keys()))

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
    )
    plt.title("Confusion Matrix - Test Set")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = str(Path(output_dir) / "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()

    report_path = Path(output_dir) / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    metrics = {
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "classification_report": report_text,
        "confusion_matrix": cm,
        "confusion_matrix_path": cm_path,
    }
    return metrics


def plot_training_curves(history: Dict[str, List[float]], output_dir: str) -> None:
    """Plot train/validation loss and accuracy curves."""
    ensure_dir(output_dir)
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "training_curves.png"))
    plt.close()
