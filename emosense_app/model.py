"""Model loading and inference helpers for EmoSense AI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor

EMOTION_EMOJI = {
    "angry": "😠",
    "happy": "😊",
    "sad": "😢",
    "fearful": "😨",
    "neutral": "😐",
    "disgust": "🤢",
    "surprised": "😲",
    "calm": "😌",
}

MODEL_NAME = "facebook/wav2vec2-base"
TARGET_SR = 16_000
FIXED_SECONDS = 3
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "outputs" / "model.pth"


class Wav2Vec2EmotionClassifier(nn.Module):
    """Wav2Vec2 encoder with a lightweight classifier head."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = MODEL_NAME,
        dropout: float = 0.3,
    ) -> None:
        """Initialize backbone and classifier layers."""
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """Run forward pass and return logits."""
        outputs = self.wav2vec2(input_values=input_values)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        return self.classifier(pooled)


@dataclass
class PredictorBundle:
    """Container for predictor dependencies and status flags."""

    model: Wav2Vec2EmotionClassifier | None
    processor: Wav2Vec2Processor | None
    id2label: Dict[int, str]
    device: torch.device
    is_mock: bool
    error_message: str | None = None


def _build_default_label_map() -> Dict[int, str]:
    ordered = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
    return {idx: label for idx, label in enumerate(ordered)}


def _prepare_waveform(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Resample, normalize, and pad/trim waveform to fixed input length."""
    if sample_rate != TARGET_SR:
        waveform = librosa.resample(waveform.astype(np.float32), orig_sr=sample_rate, target_sr=TARGET_SR)

    max_abs = np.max(np.abs(waveform)) if waveform.size else 0.0
    if max_abs > 1e-9:
        waveform = waveform / max_abs

    target_len = TARGET_SR * FIXED_SECONDS
    if waveform.shape[0] < target_len:
        waveform = np.pad(waveform, (0, target_len - waveform.shape[0]), mode="constant")
    elif waveform.shape[0] > target_len:
        waveform = waveform[:target_len]

    return waveform.astype(np.float32)


def load_predictor(model_path: str | Path = MODEL_PATH) -> PredictorBundle:
    """Load model checkpoint from outputs/model.pth, with a safe mock fallback."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(model_path)

    try:
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device)
        label2id = checkpoint.get("label2id", {v: k for k, v in _build_default_label_map().items()})
        id2label = {idx: label for label, idx in label2id.items()}

        processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
        model = Wav2Vec2EmotionClassifier(num_classes=len(label2id), model_name=MODEL_NAME)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        return PredictorBundle(
            model=model,
            processor=processor,
            id2label=id2label,
            device=device,
            is_mock=False,
            error_message=None,
        )
    except Exception as exc:
        return PredictorBundle(
            model=None,
            processor=None,
            id2label=_build_default_label_map(),
            device=device,
            is_mock=True,
            error_message=str(exc),
        )


def predict_distribution(
    waveform: np.ndarray,
    sample_rate: int,
    predictor: PredictorBundle,
) -> Dict[str, float]:
    """Predict confidence distribution for all emotions."""
    prepared = _prepare_waveform(waveform, sample_rate)

    labels = [predictor.id2label[i] for i in sorted(predictor.id2label.keys())]

    if predictor.is_mock or predictor.model is None or predictor.processor is None:
        # Stable pseudo-probabilities for demo mode.
        seed = int(np.abs(prepared[:1024]).sum() * 1_000_000) % (2**32 - 1)
        rng = np.random.default_rng(seed)
        logits = rng.normal(size=len(labels)).astype(np.float32)
        exps = np.exp(logits - logits.max())
        probs = exps / exps.sum()
        return {label: float(prob) for label, prob in zip(labels, probs)}

    proc = predictor.processor(
        prepared,
        sampling_rate=TARGET_SR,
        return_tensors="pt",
        padding=False,
    )
    input_values = proc.input_values.to(predictor.device)

    with torch.no_grad():
        logits = predictor.model(input_values)
        probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    return {label: float(prob) for label, prob in zip(labels, probs)}


def top_prediction(conf_dist: Dict[str, float]) -> Tuple[str, float]:
    """Return highest-confidence emotion and score."""
    label = max(conf_dist, key=conf_dist.get)
    return label, float(conf_dist[label])
