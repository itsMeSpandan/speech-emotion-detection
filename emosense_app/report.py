"""Mood report generation utilities."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Dict, List


def build_mood_report(
    predicted_emotion: str,
    confidence: float,
    filename: str,
    distribution: Dict[str, float],
    chat_log: List[Dict[str, object]] | None = None,
    session_summary: Dict[str, object] | None = None,
) -> Dict[str, object]:
    """Create mood report payload for UI and downloads."""
    return {
        "predicted_emotion": predicted_emotion,
        "confidence_percent": round(confidence * 100, 2),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "audio_filename": filename,
        "status": "Looks Good" if confidence > 0.80 else "Low Confidence",
        "distribution_percent": {k: round(v * 100, 2) for k, v in distribution.items()},
        "chat_log": chat_log or [],
        "session_summary": session_summary
        or {
            "total_turns": 0,
            "emotion_drift": [],
            "dominant_intent": "none",
            "contradiction_detected": False,
        },
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

    chat_log = report_payload.get("chat_log", [])
    if chat_log:
        lines.append("")
        lines.append("CONVERSATION LOG")
        for item in chat_log:
            role = str(item.get("role", "unknown")).capitalize()
            content = str(item.get("content", "")).strip()
            timestamp = str(item.get("timestamp", ""))
            intent = str(item.get("intent", ""))
            sentiment = str(item.get("sentiment", ""))
            resources_shown = item.get("resources_shown", [])

            lines.append(f"- [{timestamp}] {role}: {content}")
            if intent:
                lines.append(f"  Intent: {intent}")
            if sentiment:
                lines.append(f"  Sentiment: {sentiment}")
            if resources_shown:
                lines.append(f"  Resources: {', '.join(map(str, resources_shown))}")

    summary = report_payload.get("session_summary", {})
    if summary:
        lines.append("")
        lines.append("SESSION SUMMARY")
        lines.append(f"- Total Turns: {summary.get('total_turns', 0)}")
        lines.append(f"- Emotion Drift: {summary.get('emotion_drift', [])}")
        lines.append(f"- Dominant Intent: {summary.get('dominant_intent', 'none')}")
        lines.append(f"- Contradiction Detected: {summary.get('contradiction_detected', False)}")

    return "\n".join(lines)
