"""Colab-ready script to fine-tune Wav2Vec2 on RAVDESS speech emotions.

Usage in Google Colab:
1) !pip install -q torch torchaudio transformers librosa soundfile numpy scikit-learn matplotlib seaborn tqdm
2) !python colab_train_wav2vec2_ravdess.py --download_data

Optional arguments:
- --data_dir /content/data
- --output_dir /content/outputs
- --epochs 15
- --batch_size 16
"""

from __future__ import annotations

import argparse
import os
import random
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2Processor


RAVDESS_EMOTION_MAP: Dict[str, str] = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

RAVDESS_ZIP_URL = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"


class AudioProcessingError(Exception):
    """Raised when an audio file is missing, empty, or corrupted."""


def set_seed(seed: int = 42) -> None:
    """Set seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> None:
    """Create directory recursively when needed."""
    os.makedirs(path, exist_ok=True)


def build_label_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create label2id and id2label mappings from RAVDESS classes."""
    labels = [RAVDESS_EMOTION_MAP[k] for k in sorted(RAVDESS_EMOTION_MAP.keys())]
    label2id = {name: idx for idx, name in enumerate(labels)}
    id2label = {idx: name for name, idx in label2id.items()}
    return label2id, id2label


def download_and_extract_ravdess(data_dir: str) -> None:
    """Download and extract RAVDESS speech subset to data_dir."""
    ensure_dir(data_dir)
    zip_path = Path(data_dir) / "Audio_Speech_Actors_01-24.zip"

    actor_1 = Path(data_dir) / "Actor_01"
    if actor_1.exists():
        print("RAVDESS dataset already present. Skipping download.")
        return

    print(f"Downloading RAVDESS from: {RAVDESS_ZIP_URL}")
    urllib.request.urlretrieve(RAVDESS_ZIP_URL, str(zip_path))
    print(f"Saved archive to: {zip_path}")

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(data_dir)
    print("Extraction complete.")


def extract_emotion_from_filename(file_path: str | Path) -> Optional[str]:
    """Extract emotion label from RAVDESS filename fields."""
    stem = Path(file_path).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None
    return RAVDESS_EMOTION_MAP.get(parts[2])


def gather_ravdess_files(data_dir: str | Path) -> List[str]:
    """Collect RAVDESS .wav files from Actor_* subfolders."""
    data_path = Path(data_dir)
    files = sorted([str(p) for p in data_path.glob("Actor_*/*.wav")])
    if not files:
        raise FileNotFoundError(
            f"No RAVDESS WAV files found in '{data_dir}'. Expected pattern: {data_path / 'Actor_*/*.wav'}"
        )
    return files


def load_audio_mono_16k(audio_path: str | Path, target_sr: int = 16_000) -> np.ndarray:
    """Load audio as mono with target sample rate and validation checks."""
    audio_path = Path(audio_path)
    if not audio_path.exists() or not audio_path.is_file():
        raise AudioProcessingError(f"Missing file: {audio_path}")

    try:
        info = sf.info(str(audio_path))
    except Exception as exc:
        raise AudioProcessingError(f"Cannot read metadata: {audio_path}") from exc

    if info.frames == 0:
        raise AudioProcessingError(f"Empty file: {audio_path}")

    try:
        waveform, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
    except Exception as exc:
        raise AudioProcessingError(f"Decode failed: {audio_path}") from exc

    if waveform is None or waveform.size == 0:
        raise AudioProcessingError(f"Decoded waveform is empty: {audio_path}")

    return waveform.astype(np.float32)


def standardize_length(waveform: np.ndarray, target_length: int = 48_000) -> np.ndarray:
    """Pad or truncate waveform to a fixed sample count."""
    if waveform.shape[0] < target_length:
        pad = target_length - waveform.shape[0]
        waveform = np.pad(waveform, (0, pad), mode="constant")
    elif waveform.shape[0] > target_length:
        waveform = waveform[:target_length]
    return waveform


def normalize_waveform(waveform: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Scale waveform into stable range."""
    if waveform.size == 0:
        return waveform.astype(np.float32)
    max_abs = float(np.max(np.abs(waveform)))
    if max_abs < eps:
        return waveform.astype(np.float32)
    return (waveform / max_abs).astype(np.float32)


def stratified_split_indices(
    labels: Sequence[int],
    seed: int = 42,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test split indices using stratification when possible."""
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("Split ratios must sum to 1.0")

    y = np.array(labels)
    idx = np.arange(len(y))

    try:
        train_idx, temp_idx, _, y_temp = train_test_split(
            idx,
            y,
            test_size=(1.0 - train_size),
            random_state=seed,
            stratify=y,
        )
        val_ratio_in_temp = val_size / (val_size + test_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1.0 - val_ratio_in_temp),
            random_state=seed,
            stratify=y_temp,
        )
    except ValueError:
        train_idx, temp_idx = train_test_split(
            idx,
            test_size=(1.0 - train_size),
            random_state=seed,
            shuffle=True,
        )
        val_ratio_in_temp = val_size / (val_size + test_size)
        val_idx, test_idx = train_test_split(
            temp_idx,
            test_size=(1.0 - val_ratio_in_temp),
            random_state=seed,
            shuffle=True,
        )

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


class RavdessDataset(Dataset):
    """RAVDESS Dataset for Wav2Vec2 input generation."""

    def __init__(
        self,
        data_dir: str,
        processor: Wav2Vec2Processor,
        label2id: Dict[str, int],
        target_sr: int = 16_000,
        fixed_seconds: int = 3,
        drop_invalid: bool = True,
    ) -> None:
        """Build sample list and labels from RAVDESS directory."""
        self.processor = processor
        self.label2id = label2id
        self.target_sr = target_sr
        self.target_length = target_sr * fixed_seconds

        all_files = gather_ravdess_files(data_dir)
        self.file_paths: List[str] = []
        self.labels: List[int] = []

        for fp in all_files:
            emotion = extract_emotion_from_filename(fp)
            if emotion is None or emotion not in self.label2id:
                continue

            if drop_invalid:
                try:
                    _ = load_audio_mono_16k(fp, target_sr=self.target_sr)
                except AudioProcessingError:
                    continue

            self.file_paths.append(fp)
            self.labels.append(self.label2id[emotion])

        if not self.file_paths:
            raise RuntimeError("No valid RAVDESS audio files after filtering.")

    def __len__(self) -> int:
        """Return number of valid examples."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Return processed input values and class id."""
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            waveform = load_audio_mono_16k(audio_path, target_sr=self.target_sr)
        except AudioProcessingError:
            waveform = np.zeros(self.target_length, dtype=np.float32)

        waveform = standardize_length(waveform, target_length=self.target_length)
        waveform = normalize_waveform(waveform)

        out = self.processor(
            waveform,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=False,
        )
        input_values = out.input_values.squeeze(0)
        return input_values, label


class Wav2Vec2EmotionClassifier(nn.Module):
    """Wav2Vec2 backbone + simple MLP classifier head."""

    def __init__(
        self,
        num_classes: int = 8,
        model_name: str = "facebook/wav2vec2-base",
        dropout: float = 0.3,
        freeze_feature_extractor: bool = True,
    ) -> None:
        """Initialize backbone and classification head."""
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_feature_extractor:
            for p in self.wav2vec2.feature_extractor.parameters():
                p.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Return logits without softmax."""
        outputs = self.wav2vec2(input_values=input_values)
        hidden = outputs.last_hidden_state
        pooled = hidden.mean(dim=1)
        logits = self.classifier(pooled)
        return logits


def create_dataloaders(
    dataset: RavdessDataset,
    batch_size: int,
    seed: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """Create stratified train/val/test DataLoaders with 70/15/15 split."""
    train_idx, val_idx, test_idx = stratified_split_indices(dataset.labels, seed=seed)

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }


def compute_weighted_loss(labels: List[int], device: torch.device) -> nn.Module:
    """Build class-weighted cross-entropy loss."""
    classes = np.unique(labels)
    weights = compute_class_weight("balanced", classes=classes, y=np.array(labels))
    max_class = int(np.max(classes))
    full = np.ones(max_class + 1, dtype=np.float32)
    for c, w in zip(classes, weights):
        full[int(c)] = float(w)
    return nn.CrossEntropyLoss(weight=torch.tensor(full, dtype=torch.float32, device=device))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: Optional[float] = None,
) -> Tuple[float, float]:
    """Run one training or validation epoch."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    bar = tqdm(loader, desc="Train" if is_train else "Val", leave=False)
    for input_values, labels in bar:
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

        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_samples += labels.size(0)
        total_loss += loss.item() * labels.size(0)

    return total_loss / max(total_samples, 1), total_correct / max(total_samples, 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_labels: List[int],
    device: torch.device,
    label2id: Dict[str, int],
    output_dir: str,
    epochs: int = 15,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    patience: int = 3,
    grad_clip: float = 1.0,
) -> Tuple[Dict[str, List[float]], str]:
    """Fine-tune model with early stopping and save best checkpoint."""
    ensure_dir(output_dir)
    ckpt_path = str(Path(output_dir) / "model.pth")

    criterion = compute_weighted_loss(train_labels, device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=1)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    no_improve = 0

    model.to(device)

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer, grad_clip=grad_clip
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device)
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
            no_improve = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "label2id": label2id,
                    "best_val_loss": best_val_loss,
                },
                ckpt_path,
            )
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}.")
                break

    return history, ckpt_path


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
    output_dir: str,
) -> Dict[str, object]:
    """Evaluate model and save report and confusion matrix."""
    ensure_dir(output_dir)
    model.eval()

    y_true: List[int] = []
    y_pred: List[int] = []

    with torch.no_grad():
        for input_values, labels in test_loader:
            input_values = input_values.to(device)
            logits = model(input_values)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.numpy().tolist())

    acc = accuracy_score(y_true, y_pred)
    f1w = f1_score(y_true, y_pred, average="weighted")

    class_ids = sorted(id2label.keys())
    names = [id2label[i] for i in class_ids]

    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=names,
        digits=4,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=names, yticklabels=names)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    cm_path = str(Path(output_dir) / "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    report_path = Path(output_dir) / "classification_report.txt"
    report_path.write_text(report, encoding="utf-8")

    return {
        "accuracy": float(acc),
        "weighted_f1": float(f1w),
        "classification_report": report,
        "confusion_matrix_path": cm_path,
    }


def plot_training_curves(history: Dict[str, List[float]], output_dir: str) -> None:
    """Save train vs val loss and accuracy plots."""
    ensure_dir(output_dir)
    x = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, history["train_loss"], label="Train Loss")
    plt.plot(x, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, history["train_acc"], label="Train Acc")
    plt.plot(x, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(str(Path(output_dir) / "training_curves.png"))
    plt.close()


def load_trained_model(
    checkpoint_path: str,
    device: torch.device,
    model_name: str = "facebook/wav2vec2-base",
) -> Tuple[nn.Module, Dict[int, str], Wav2Vec2Processor]:
    """Load checkpoint and return model, id2label, and processor."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    label2id = checkpoint["label2id"]
    id2label = {idx: label for label, idx in label2id.items()}

    model = Wav2Vec2EmotionClassifier(num_classes=len(label2id), model_name=model_name)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    processor = Wav2Vec2Processor.from_pretrained(model_name)
    return model, id2label, processor


def predict_emotion(
    audio_path: str,
    model: nn.Module,
    processor: Wav2Vec2Processor,
    id2label: Dict[int, str],
    device: torch.device,
    target_sr: int = 16_000,
    fixed_seconds: int = 3,
) -> Dict[str, float | str]:
    """Predict emotion label and confidence for one audio file."""
    target_len = target_sr * fixed_seconds

    waveform = load_audio_mono_16k(audio_path, target_sr=target_sr)
    waveform = standardize_length(waveform, target_length=target_len)
    waveform = normalize_waveform(waveform)

    proc = processor(waveform, sampling_rate=target_sr, return_tensors="pt", padding=False)
    input_values = proc.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values)
        probs = F.softmax(logits, dim=-1)
        conf, pred = torch.max(probs, dim=-1)

    pred_id = int(pred.item())
    return {"label": id2label[pred_id], "confidence": float(conf.item())}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Fine-tune Wav2Vec2 for SER on RAVDESS (Colab)")
    p.add_argument("--data_dir", type=str, default="/content/data")
    p.add_argument("--output_dir", type=str, default="/content/outputs")
    p.add_argument("--model_name", type=str, default="facebook/wav2vec2-base")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--download_data", action="store_true")
    # In notebook kernels, extra args like '-f <kernel.json>' are appended.
    # parse_known_args keeps this script runnable both as CLI and inside Colab cells.
    args, _ = p.parse_known_args()
    return args


def main() -> None:
    """Run full training/evaluation/inference pipeline."""
    args = parse_args()

    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    print(f"Using device: {device}")

    if args.download_data:
        download_and_extract_ravdess(args.data_dir)

    processor = Wav2Vec2Processor.from_pretrained(args.model_name)

    label2id, id2label = build_label_mappings()
    print(f"label2id: {label2id}")

    dataset = RavdessDataset(
        data_dir=args.data_dir,
        processor=processor,
        label2id=label2id,
        target_sr=16_000,
        fixed_seconds=3,
    )
    print(f"Total valid samples: {len(dataset)}")
    print(f"Class distribution: {dict(Counter(dataset.labels))}")

    train_loader, val_loader, test_loader, splits = create_dataloaders(
        dataset=dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(
        f"Split sizes => train: {len(splits['train'])}, "
        f"val: {len(splits['val'])}, test: {len(splits['test'])}"
    )

    model = Wav2Vec2EmotionClassifier(
        num_classes=len(label2id),
        model_name=args.model_name,
        dropout=0.3,
        freeze_feature_extractor=True,
    )

    train_labels = [dataset.labels[i] for i in splits["train"].tolist()]
    history, ckpt_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_labels=train_labels,
        device=device,
        label2id=label2id,
        output_dir=args.output_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
        grad_clip=1.0,
    )

    print(f"Best model saved at: {ckpt_path}")

    best = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best["model_state_dict"])
    model.to(device)
    model.eval()

    metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        id2label=id2label,
        output_dir=args.output_dir,
    )
    plot_training_curves(history, args.output_dir)

    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Weighted F1: {metrics['weighted_f1']:.4f}")
    print(metrics["classification_report"])

    sample_paths = [dataset.file_paths[i] for i in splits["test"][:3].tolist()]
    print("Sample inference on at least 3 test files:")
    model_inf, id2label_inf, processor_inf = load_trained_model(
        checkpoint_path=ckpt_path,
        device=device,
        model_name=args.model_name,
    )

    for p in sample_paths:
        try:
            result = predict_emotion(p, model_inf, processor_inf, id2label_inf, device)
            print(f"{p} -> {result}")
        except Exception as exc:
            print(f"Failed on {p}: {exc}")


if __name__ == "__main__":
    main()
