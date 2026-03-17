"""Utility helpers for speech emotion recognition with RAVDESS."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch
from sklearn.model_selection import train_test_split

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


class AudioProcessingError(Exception):
    """Raised when an audio file cannot be decoded or is invalid."""


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_label_mappings() -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create label2id and id2label mappings from RAVDESS emotions."""
    labels = [RAVDESS_EMOTION_MAP[key] for key in sorted(RAVDESS_EMOTION_MAP.keys())]
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def extract_emotion_from_filename(file_path: str | Path) -> Optional[str]:
    """Extract emotion label name from a RAVDESS filename.

    Expected example: 03-01-05-01-02-01-12.wav where the 3rd field is emotion id.
    """
    name = Path(file_path).name
    stem = Path(name).stem
    parts = stem.split("-")
    if len(parts) < 3:
        return None

    emotion_id = parts[2]
    return RAVDESS_EMOTION_MAP.get(emotion_id)


def standardize_length(waveform: np.ndarray, target_length: int = 48_000) -> np.ndarray:
    """Pad or trim waveform to a fixed number of samples."""
    if waveform.shape[0] < target_length:
        pad_width = target_length - waveform.shape[0]
        waveform = np.pad(waveform, (0, pad_width), mode="constant")
    elif waveform.shape[0] > target_length:
        waveform = waveform[:target_length]
    return waveform


def normalize_waveform(waveform: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Normalize waveform amplitude to [-1, 1] with numerical stability."""
    max_abs = np.max(np.abs(waveform)) if waveform.size else 0.0
    if max_abs < eps:
        return waveform.astype(np.float32)
    return (waveform / max_abs).astype(np.float32)


def load_audio_mono_16k(audio_path: str | Path, target_sr: int = 16_000) -> np.ndarray:
    """Load audio robustly as mono waveform at target sampling rate.

    Raises AudioProcessingError for corrupted, unreadable, or empty audio.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists() or not audio_path.is_file():
        raise AudioProcessingError(f"Missing audio file: {audio_path}")

    try:
        info = sf.info(str(audio_path))
    except Exception as exc:
        raise AudioProcessingError(f"Cannot read audio metadata: {audio_path}") from exc

    if info.frames == 0:
        raise AudioProcessingError(f"Empty audio file: {audio_path}")

    try:
        waveform, _ = librosa.load(str(audio_path), sr=target_sr, mono=True)
    except Exception as exc:
        raise AudioProcessingError(f"Failed to decode audio: {audio_path}") from exc

    if waveform is None or waveform.size == 0:
        raise AudioProcessingError(f"Decoded empty waveform: {audio_path}")

    return waveform.astype(np.float32)


def gather_ravdess_files(data_dir: str | Path) -> List[str]:
    """Collect all WAV files from Actor_* subdirectories."""
    data_path = Path(data_dir)
    pattern = str(data_path / "Actor_*" / "*.wav")
    files = sorted([str(path) for path in data_path.glob("Actor_*/*.wav")])

    if not files:
        raise FileNotFoundError(
            f"No RAVDESS WAV files found in '{data_dir}'. Expected pattern: {pattern}"
        )

    return files


def stratified_split_indices(
    labels: Sequence[int],
    seed: int = 42,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split indices into train/val/test with stratification and fallback.

    The split is performed in two stages:
    1) train vs temp
    2) val vs test from temp
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    labels_np = np.array(labels)
    indices = np.arange(len(labels_np))

    try:
        train_idx, temp_idx, y_train, y_temp = train_test_split(
            indices,
            labels_np,
            test_size=(1.0 - train_size),
            random_state=seed,
            stratify=labels_np,
        )

        val_ratio_in_temp = val_size / (val_size + test_size)
        val_idx, test_idx, _, _ = train_test_split(
            temp_idx,
            y_temp,
            test_size=(1.0 - val_ratio_in_temp),
            random_state=seed,
            stratify=y_temp,
        )
    except ValueError:
        # Fallback for tiny or heavily imbalanced subsets where stratification can fail.
        train_idx, temp_idx = train_test_split(
            indices,
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


def ensure_dir(path: str | Path) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)
