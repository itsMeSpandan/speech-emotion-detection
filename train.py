"""Training routines for Wav2Vec2 SER."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import ensure_dir


def compute_weighted_loss(labels: List[int], device: torch.device) -> nn.CrossEntropyLoss:
    """Build weighted CrossEntropyLoss to mitigate class imbalance."""
    classes = np.unique(labels)
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=np.array(labels),
    )

    max_class = int(np.max(classes))
    full_weights = np.ones(max_class + 1, dtype=np.float32)
    for class_id, w in zip(classes, class_weights):
        full_weights[int(class_id)] = float(w)

    weight_tensor = torch.tensor(full_weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weight_tensor)


def _run_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
) -> Tuple[float, float]:
    """Run one train/eval epoch and return loss and accuracy."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    correct = 0
    total = 0

    iterator = tqdm(dataloader, desc="Train" if is_train else "Val", leave=False)
    for input_values, labels in iterator:
        input_values = input_values.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(input_values)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def train_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels: List[int],
    device: torch.device,
    output_dir: str,
    label2id: Dict[str, int],
    epochs: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 3,
    grad_clip: float = 1.0,
) -> Tuple[Dict[str, List[float]], str]:
    """Train with early stopping and save best checkpoint by val loss."""
    ensure_dir(output_dir)
    checkpoint_path = str(Path(output_dir) / "model.pth")

    criterion = compute_weighted_loss(train_labels, device=device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = _run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            grad_clip=grad_clip,
        )
        val_loss, val_acc = _run_epoch(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
        )
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"train_acc={train_acc:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    return history, checkpoint_path
