"""Sentiment analysis and signal-combination helpers for EmoSense."""

from __future__ import annotations

import streamlit as st
from transformers import pipeline


@st.cache_resource
def get_sentiment_pipeline():
    """Load sentiment model once per process."""
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")


def analyze_sentiment(text: str) -> dict:
    """Analyze text sentiment and return normalized output."""
    cleaned = (text or "").strip()
    if not cleaned:
        return {"label": "NEUTRAL", "score": 0.0, "readable": "Neutral"}

    model = get_sentiment_pipeline()
    result = model(cleaned[:4000])[0]
    label = str(result.get("label", "NEUTRAL")).upper()
    score = float(result.get("score", 0.0))

    if label not in {"NEGATIVE", "NEUTRAL", "POSITIVE"}:
        label = "NEUTRAL"

    readable = label.capitalize().lower().capitalize()
    return {"label": label, "score": score, "readable": readable}


def combine_signals(
    voice_emotion: str,
    voice_confidence: float,
    text_sentiment: dict,
) -> dict:
    """Combine voice and text cues into a single interpreted profile."""
    emotion = (voice_emotion or "neutral").lower()
    voice_conf = float(voice_confidence or 0.0)
    sentiment_label = str(text_sentiment.get("label", "NEUTRAL")).upper()
    sentiment_score = float(text_sentiment.get("score", 0.0))

    positive_voice = {"happy", "calm", "surprised", "neutral"}
    negative_voice = {"sad", "angry", "fearful", "disgust"}

    contradiction = False
    dominant = "voice" if voice_conf >= sentiment_score else "text"

    if emotion in positive_voice and sentiment_label == "NEGATIVE":
        contradiction = True
        dominant = "text" if sentiment_score > voice_conf else "voice"
        summary = "Voice sounds positive, but text reads negative"
    elif emotion in negative_voice and sentiment_label == "POSITIVE":
        contradiction = True
        dominant = "voice"
        summary = "Sounds distressed, but typing positively"
    else:
        contradiction = False
        dominant = "voice" if voice_conf >= sentiment_score else "text"
        summary = f"Voice and text broadly align toward {emotion} / {sentiment_label.lower()}"

    return {
        "voice_emotion": emotion,
        "voice_confidence": voice_conf,
        "text_sentiment": sentiment_label,
        "text_score": sentiment_score,
        "contradiction": contradiction,
        "dominant": dominant,
        "summary": summary,
    }
