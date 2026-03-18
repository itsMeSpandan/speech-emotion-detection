"""Mood report generation utilities."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict


def build_mood_report(
    predicted_emotion: str,
    confidence: float,
    filename: str,
    distribution: Dict[str, float],
) -> Dict[str, object]:
    """Create mood report payload for UI and downloads."""
    return {
        "predicted_emotion": predicted_emotion,
        "confidence_percent": round(confidence * 100, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "audio_filename": filename,
        "status": "Looks Good" if confidence > 0.80 else "Low Confidence",
        "distribution_percent": {k: round(v * 100, 2) for k, v in distribution.items()},
    }


def report_as_json(report_payload: Dict[str, object]) -> str:
    """Serialize report to formatted JSON string."""
    return json.dumps(report_payload, indent=2)


def report_as_text(report_payload: Dict[str, object]) -> str:
    """Serialize report to plain text string."""
    lines = [
        "EmoSense AI Mood Report",
        f"Predicted Emotion: {report_payload['predicted_emotion']}",
        f"Confidence: {report_payload['confidence_percent']}%",
        f"Timestamp: {report_payload['timestamp']}",
        f"Audio Filename: {report_payload['audio_filename']}",
        f"Status: {report_payload['status']}",
        "",
        "Confidence Distribution (%):",
    ]

    dist = report_payload.get("distribution_percent", {})
    for emotion, pct in dist.items():
        lines.append(f"- {emotion}: {pct}%")

    return "\n".join(lines)
