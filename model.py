"""Model definition for Wav2Vec2-based speech emotion recognition."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class Wav2Vec2EmotionClassifier(nn.Module):
    """Wav2Vec2 encoder with a lightweight classification head."""

    def __init__(
        self,
        num_classes: int = 8,
        model_name: str = "facebook/wav2vec2-base",
        dropout: float = 0.3,
        freeze_feature_extractor: bool = True,
    ) -> None:
        """Initialize pretrained backbone and classifier head."""
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)

        if freeze_feature_extractor:
            for param in self.wav2vec2.feature_extractor.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return class logits without softmax."""
        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        pooled = hidden_states.mean(dim=1)
        logits = self.classifier(pooled)
        return logits
