"""Inference utilities for SER model."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from transformers import Wav2Vec2Processor

from model import Wav2Vec2EmotionClassifier
from utils import load_audio_mono_16k, normalize_waveform, standardize_length

_MODEL: Optional[Wav2Vec2EmotionClassifier] = None
_PROCESSOR: Optional[Wav2Vec2Processor] = None
_ID2LABEL: Optional[Dict[int, str]] = None
_DEVICE: Optional[torch.device] = None


def load_ser_model(
    checkpoint_path: str,
    device: torch.device,
    model_name: str = "facebook/wav2vec2-base",
) -> Tuple[Wav2Vec2EmotionClassifier, Dict[str, int], Dict[int, str]]:
    """Load trained SER model checkpoint and label mappings."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    label2id = checkpoint["label2id"]
    id2label = {idx: label for label, idx in label2id.items()}

    model = Wav2Vec2EmotionClassifier(
        num_classes=len(label2id),
        model_name=model_name,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, label2id, id2label


def initialize_inference(
    checkpoint_path: str,
    device: Optional[torch.device] = None,
    model_name: str = "facebook/wav2vec2-base",
) -> None:
    """Initialize global inference objects for predict_emotion(audio_path)."""
    global _MODEL, _PROCESSOR, _ID2LABEL, _DEVICE

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _PROCESSOR = Wav2Vec2Processor.from_pretrained(model_name)
    _MODEL, _, _ID2LABEL = load_ser_model(
        checkpoint_path=checkpoint_path,
        device=device,
        model_name=model_name,
    )
    _DEVICE = device


def _predict_with_components(
    audio_path: str,
    model: Wav2Vec2EmotionClassifier,
    processor: Wav2Vec2Processor,
    id2label: Dict[int, str],
    device: torch.device,
    target_sr: int = 16_000,
    fixed_seconds: int = 3,
) -> Dict[str, float | str]:
    """Predict emotion label and confidence for an audio file."""
    target_length = target_sr * fixed_seconds

    waveform = load_audio_mono_16k(audio_path, target_sr=target_sr)
    waveform = standardize_length(waveform, target_length=target_length)
    waveform = normalize_waveform(waveform)

    processed = processor(
        waveform,
        sampling_rate=target_sr,
        return_tensors="pt",
        padding=False,
    )
    input_values = processed.input_values.to(device)

    with torch.no_grad():
        logits = model(input_values)
        probs = F.softmax(logits, dim=-1)
        conf, pred_idx = torch.max(probs, dim=-1)

    pred_class = int(pred_idx.item())
    return {
        "label": id2label[pred_class],
        "confidence": float(conf.item()),
    }


def predict_emotion(audio_path: str) -> Dict[str, float | str]:
    """Predict emotion for one audio file using initialized global inference state.

    Call initialize_inference(checkpoint_path, ...) once before using this function.
    """
    if _MODEL is None or _PROCESSOR is None or _ID2LABEL is None or _DEVICE is None:
        raise RuntimeError(
            "Inference is not initialized. Call initialize_inference(checkpoint_path, ...) first."
        )

    return _predict_with_components(
        audio_path=audio_path,
        model=_MODEL,
        processor=_PROCESSOR,
        id2label=_ID2LABEL,
        device=_DEVICE,
    )


def predict_emotion_from_checkpoint(
    audio_path: str,
    checkpoint_path: str,
    device: torch.device,
    model_name: str = "facebook/wav2vec2-base",
) -> Dict[str, float | str]:
    """Convenience wrapper that loads model and predicts in one call."""
    initialize_inference(
        checkpoint_path=checkpoint_path,
        device=device,
        model_name=model_name,
    )
    return predict_emotion(audio_path)


def run_sample_inference(
    sample_paths: list[str],
    checkpoint_path: str,
    device: torch.device,
    model_name: str = "facebook/wav2vec2-base",
) -> None:
    """Run inference on a list of sample files and print predictions."""
    if not sample_paths:
        print("No sample paths supplied for inference.")
        return

    initialize_inference(
        checkpoint_path=checkpoint_path,
        device=device,
        model_name=model_name,
    )

    for path in sample_paths:
        if not Path(path).exists():
            print(f"Skipping missing file: {path}")
            continue
        try:
            result = predict_emotion(path)
            print(f"{path} -> {result}")
        except Exception as exc:
            print(f"Failed inference for {path}: {exc}")
