"""Dataset and preprocessing utilities for RAVDESS SER."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import Wav2Vec2Processor

from utils import (
    AudioProcessingError,
    extract_emotion_from_filename,
    gather_ravdess_files,
    load_audio_mono_16k,
    normalize_waveform,
    standardize_length,
    stratified_split_indices,
)


class RavdessDataset(Dataset):
    """PyTorch Dataset for RAVDESS speech emotion recognition.

    Each item returns:
    - input_values (Tensor): shape [fixed_length]
    - label (int)
    """

    def __init__(
        self,
        data_dir: str,
        processor: Wav2Vec2Processor,
        label2id: Dict[str, int],
        target_sr: int = 16_000,
        fixed_seconds: int = 3,
        drop_invalid: bool = True,
    ) -> None:
        """Initialize dataset and index valid files.

        Args:
            data_dir: Root directory containing Actor_* folders.
            processor: HuggingFace Wav2Vec2Processor.
            label2id: Emotion-to-index mapping.
            target_sr: Target sampling rate for audio loading.
            fixed_seconds: Fixed clip length in seconds.
            drop_invalid: If True, invalid/corrupted files are skipped.
        """
        self.processor = processor
        self.label2id = label2id
        self.target_sr = target_sr
        self.target_length = target_sr * fixed_seconds

        all_files = gather_ravdess_files(data_dir)
        self.file_paths: List[str] = []
        self.labels: List[int] = []

        for path in all_files:
            emotion = extract_emotion_from_filename(path)
            if emotion is None or emotion not in self.label2id:
                continue

            if drop_invalid:
                try:
                    waveform = load_audio_mono_16k(path, target_sr=self.target_sr)
                    if waveform.size == 0:
                        continue
                except AudioProcessingError:
                    continue

            self.file_paths.append(path)
            self.labels.append(self.label2id[emotion])

        if not self.file_paths:
            raise RuntimeError("No valid audio files found after filtering.")

    def __len__(self) -> int:
        """Return number of valid samples."""
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load, preprocess, and return a single sample.

        If a file unexpectedly fails at runtime, returns a zero waveform.
        """
        audio_path = self.file_paths[idx]
        label = self.labels[idx]

        try:
            waveform = load_audio_mono_16k(audio_path, target_sr=self.target_sr)
        except AudioProcessingError:
            waveform = np.zeros(self.target_length, dtype=np.float32)

        waveform = standardize_length(waveform, target_length=self.target_length)
        waveform = normalize_waveform(waveform)

        processed = self.processor(
            waveform,
            sampling_rate=self.target_sr,
            return_tensors="pt",
            padding=False,
        )
        input_values = processed.input_values.squeeze(0)

        return input_values, label


def create_dataloaders(
    dataset: RavdessDataset,
    batch_size: int = 16,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, np.ndarray]]:
    """Create train/val/test DataLoaders with 70/15/15 split."""
    train_idx, val_idx, test_idx = stratified_split_indices(dataset.labels, seed=seed)

    train_ds = Subset(dataset, train_idx.tolist())
    val_ds = Subset(dataset, val_idx.tolist())
    test_ds = Subset(dataset, test_idx.tolist())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    split_indices = {"train": train_idx, "val": val_idx, "test": test_idx}
    return train_loader, val_loader, test_loader, split_indices


def summarize_class_distribution(labels: List[int]) -> Dict[int, int]:
    """Return class-count summary for diagnostics."""
    return dict(Counter(labels))
