"""Entry point for training and evaluating a Wav2Vec2 SER system on RAVDESS."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import Wav2Vec2Processor

from dataset import RavdessDataset, create_dataloaders, summarize_class_distribution
from evaluate import evaluate_model, plot_training_curves
from inference import run_sample_inference
from model import Wav2Vec2EmotionClassifier
from train import train_model
from utils import build_label_mappings, ensure_dir, get_device, set_seed


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Speech Emotion Recognition with Wav2Vec2")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to RAVDESS root")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for artifacts")
    parser.add_argument("--model_name", type=str, default="facebook/wav2vec2-base")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Run full training, evaluation, and sample inference pipeline."""
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    ensure_dir(args.output_dir)

    print(f"Using device: {device}")
    print("Loading processor...")
    processor = Wav2Vec2Processor.from_pretrained(args.model_name)

    label2id, id2label = build_label_mappings()
    print(f"Labels: {label2id}")

    print("Building dataset...")
    dataset = RavdessDataset(
        data_dir=args.data_dir,
        processor=processor,
        label2id=label2id,
        target_sr=16_000,
        fixed_seconds=3,
    )

    dist = summarize_class_distribution(dataset.labels)
    print(f"Dataset size: {len(dataset)}")
    print(f"Class distribution (label_id: count): {dist}")

    train_loader, val_loader, test_loader, split_indices = create_dataloaders(
        dataset,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    print(
        f"Split sizes -> train: {len(split_indices['train'])}, "
        f"val: {len(split_indices['val'])}, test: {len(split_indices['test'])}"
    )

    model = Wav2Vec2EmotionClassifier(
        num_classes=len(label2id),
        model_name=args.model_name,
        dropout=0.3,
        freeze_feature_extractor=True,
    )

    print("Starting training...")
    train_labels = [dataset.labels[i] for i in split_indices["train"].tolist()]
    history, checkpoint_path = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        train_labels=train_labels,
        device=device,
        output_dir=args.output_dir,
        label2id=label2id,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patience=args.patience,
    )

    print(f"Best model saved to: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Evaluating on test set...")
    metrics = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        id2label=id2label,
        output_dir=args.output_dir,
    )
    plot_training_curves(history, args.output_dir)

    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
    print("Classification report:")
    print(metrics["classification_report"])

    test_paths = [dataset.file_paths[i] for i in split_indices["test"][:3].tolist()]
    print("Running inference on at least 3 test samples...")
    run_sample_inference(
        sample_paths=test_paths,
        checkpoint_path=checkpoint_path,
        device=device,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
