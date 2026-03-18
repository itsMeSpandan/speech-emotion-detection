"""Audio loading and waveform plotting helpers for EmoSense AI."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def load_audio_from_upload(file_name: str, file_bytes: bytes) -> Tuple[np.ndarray, int]:
    """Decode uploaded audio bytes into mono waveform and sample rate."""
    suffix = Path(file_name).suffix.lower()
    # Browser microphone capture may provide non-standard extensions.
    # We allow common uploads directly and still attempt decode for unknown suffixes.
    known_suffixes = {".wav", ".mp3", ".ogg", ".webm", ".m4a", ".aac"}
    if suffix and suffix not in known_suffixes:
        raise ValueError("Unsupported format. Upload .wav, .mp3, .ogg, .webm, or .m4a.")

    buffer = io.BytesIO(file_bytes)

    try:
        waveform, sample_rate = sf.read(buffer)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if waveform.size == 0:
            raise ValueError("Uploaded audio is empty.")
        return waveform.astype(np.float32), int(sample_rate)
    except Exception:
        # Fallback path for encoded formats unsupported by soundfile in some setups.
        waveform, sample_rate = librosa.load(io.BytesIO(file_bytes), sr=None, mono=True)
        if waveform is None or waveform.size == 0:
            raise ValueError("Could not decode uploaded audio.")
        return waveform.astype(np.float32), int(sample_rate)


def get_audio_metadata(file_bytes: bytes, waveform: np.ndarray, sample_rate: int) -> dict:
    """Return metadata about the loaded audio file."""
    duration = len(waveform) / sample_rate
    file_size_kb = len(file_bytes) / 1024.0
    
    # We load as mono, though the original could be stereo.
    # Returning 'Mono' since that's what the model uses.
    return {
        "duration": f"{duration:.1f}s",
        "sample_rate": f"{sample_rate:,} Hz",
        "channels": "Mono",
        "file_size_kb": f"{file_size_kb:.0f} KB"
    }


def waveform_figure(waveform: np.ndarray, sample_rate: int):
    """Build a matplotlib figure for waveform display."""
    fig, ax = plt.subplots(figsize=(8, 2.8))
    librosa.display.waveshow(waveform, sr=sample_rate, ax=ax, color="#62b0ff")
    ax.set_title("Waveform of uploaded audio")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    fig.tight_layout()
    return fig
