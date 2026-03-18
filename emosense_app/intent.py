"""Intent and crisis detection helpers for EmoSense conversations."""

from __future__ import annotations


CRISIS_PHRASES = [
    "end my life",
    "kill myself",
    "can't go on",
    "want to die",
    "no reason to live",
    "disappear forever",
    "hurt myself",
    "not worth living",
    "give up on everything",
    "can't do this anymore",
]


def detect_crisis(text: str) -> bool:
    """Hard crisis phrase check that runs before any LLM call."""
    lowered = (text or "").lower()
    return any(phrase in lowered for phrase in CRISIS_PHRASES)


def classify_intent(text: str, emotion: str, sentiment: str) -> str:
    """Rule-based intent classifier with strict priority order."""
    lowered = (text or "").lower()
    sentiment_label = (sentiment or "NEUTRAL").upper()
    emotion_label = (emotion or "neutral").lower()

    if detect_crisis(lowered):
        return "crisis"

    advice_keywords = [
        "what should i",
        "help me",
        "advice",
        "what do i do",
        "how should i",
    ]
    if any(keyword in lowered for keyword in advice_keywords):
        return "seeking_advice"

    resource_keywords = [
        "exercises",
        "resources",
        "recommend",
        "suggest",
        "music",
        "article",
    ]
    if any(keyword in lowered for keyword in resource_keywords):
        return "resource_request"

    gratitude_keywords = ["thanks", "thank you", "that helped", "appreciate it"]
    if any(keyword in lowered for keyword in gratitude_keywords):
        return "gratitude"

    negative_emotions = {"sad", "angry", "fearful", "disgust"}
    if emotion_label in negative_emotions and sentiment_label == "NEGATIVE" and "?" not in lowered:
        return "venting"

    return "casual"
