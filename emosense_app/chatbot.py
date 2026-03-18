"""Emotion-aware chatbot helpers powered by Qwen on Hugging Face Inference API."""

from __future__ import annotations

import os
import importlib
from pathlib import Path
from typing import Dict, List

from huggingface_hub import InferenceClient

try:
    from .intent import classify_intent, detect_crisis
    from .resources import get_resources
    from .sentiment import analyze_sentiment, combine_signals
except ImportError:
    from intent import classify_intent, detect_crisis
    from resources import get_resources
    from sentiment import analyze_sentiment, combine_signals

QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"

EMOTION_TONE_MAP: Dict[str, str] = {
    "neutral": "friendly and conversational",
    "calm": "gentle, thoughtful, peaceful",
    "happy": "warm, upbeat, celebratory",
    "sad": "compassionate, gentle, supportive",
    "angry": "calm, non-confrontational, validating",
    "fearful": "reassuring, grounding, safe",
    "disgust": "non-judgmental, calm, understanding",
    "surprised": "curious, engaged, energetic",
}


def _build_system_prompt(
    voice_emotion: str,
    voice_confidence: float,
    sentiment_label: str,
    sentiment_score: float,
    contradiction: bool,
    summary: str,
    intent: str,
) -> str:
    """Create behavior-constrained prompt template for the full pipeline."""
    contradiction_text = "Yes" if contradiction else "No"
    return (
        "You are EmoSense, an emotionally intelligent AI companion.\n\n"
        "EMOTIONAL CONTEXT:\n"
        f"- Voice emotion: {voice_emotion} ({voice_confidence:.0%} confidence)\n"
        f"- Text sentiment: {sentiment_label} ({sentiment_score:.0%})\n"
        f"- Signal contradiction: {contradiction_text} — {summary}\n"
        f"- Conversation intent: {intent}\n\n"
        "BEHAVIOUR RULES:\n"
        "- If intent is venting: reflect feelings, do NOT give advice unless asked\n"
        "- If intent is seeking_advice: give 1-2 concrete, gentle suggestions\n"
        "- If intent is casual: be warm and playful\n"
        "- If intent is resource_request: mention you are showing resources below\n"
        "- If contradiction is True: gently acknowledge the user may be masking their feelings, do not push\n"
        "- Never mention 'emotion detection', 'model', 'analysis', or 'AI pipeline'\n"
        "- Respond in 2-4 sentences unless user asks for more\n"
        "- Do not repeat the emotion label back robotically"
    )


def _resolve_hf_token() -> str:
    """Resolve Hugging Face token in strict priority order."""
    # 1) Environment variable
    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token

    # 2) Streamlit secrets
    try:
        import streamlit as st

        secret_token = str(st.secrets.get("HF_TOKEN", "")).strip()
        if secret_token:
            return secret_token
    except Exception:
        pass

    # 3) .env fallback via python-dotenv
    project_root = Path(__file__).resolve().parent.parent
    dotenv_path = project_root / ".env"

    try:
        dotenv_module = importlib.import_module("dotenv")
        load_dotenv = getattr(dotenv_module, "load_dotenv")

        load_dotenv(dotenv_path=dotenv_path, override=False)
        dotenv_token = os.environ.get("HF_TOKEN", "").strip()
        if dotenv_token:
            return dotenv_token
    except Exception:
        # Minimal fallback parser when python-dotenv is unavailable.
        if dotenv_path.exists():
            try:
                for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
                    line = raw_line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    if key.strip() != "HF_TOKEN":
                        continue
                    cleaned = value.strip().strip('"').strip("'")
                    if cleaned:
                        return cleaned
            except Exception:
                pass

    return ""


def _build_client() -> InferenceClient:
    """Build a Hugging Face inference client using the environment token."""
    token = _resolve_hf_token()
    if not token:
        raise RuntimeError(
            "Hugging Face token not found. Configure one of these:\n"
            "1) Environment variable: HF_TOKEN\n"
            "2) Streamlit secrets: .streamlit/secrets.toml with HF_TOKEN\n"
            "3) Project .env file: HF_TOKEN=your_huggingface_token_here"
        )
    return InferenceClient(model=QWEN_MODEL, token=token)


def _safe_content(raw_content: object) -> str:
    """Normalize message content into plain text for API payloads."""
    if isinstance(raw_content, str):
        return raw_content
    return str(raw_content)


def full_pipeline(
    voice_emotion: str,
    voice_confidence: float,
    user_message: str,
    history: list,
) -> dict:
    """Run crisis, sentiment, intent, response, and resource recommendation pipeline."""
    if detect_crisis(user_message):
        return {
            "reply": "",
            "intent": "crisis",
            "sentiment": {},
            "signal_profile": {
                "voice_emotion": voice_emotion,
                "voice_confidence": float(voice_confidence),
                "text_sentiment": "UNKNOWN",
                "text_score": 0.0,
                "contradiction": False,
                "dominant": "voice",
                "summary": "Crisis language detected",
            },
            "resources": get_resources(voice_emotion, "crisis", False),
            "is_crisis": True,
            "safe_response": (
                "I hear that you're going through something really painful right now. "
                "You don't have to face this alone. Please reach out to a crisis helpline "
                "- they are free, confidential, and available right now. "
                "iCall: 9152987821 | Vandrevala: 1860-2662-345 (24x7)"
            ),
        }

    sentiment = analyze_sentiment(user_message)
    intent = classify_intent(user_message, voice_emotion, sentiment.get("label", "NEUTRAL"))
    signal_profile = combine_signals(voice_emotion, voice_confidence, sentiment)

    system_prompt = _build_system_prompt(
        voice_emotion=voice_emotion,
        voice_confidence=float(voice_confidence),
        sentiment_label=str(sentiment.get("label", "NEUTRAL")),
        sentiment_score=float(sentiment.get("score", 0.0)),
        contradiction=bool(signal_profile.get("contradiction", False)),
        summary=str(signal_profile.get("summary", "")),
        intent=intent,
    )

    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for msg in history:
        role = msg.get("role", "user")
        if role not in {"user", "assistant", "system"}:
            continue
        messages.append({"role": role, "content": _safe_content(msg.get("content", ""))})
    messages.append({"role": "user", "content": user_message})

    reply = ""
    try:
        client = _build_client()
        response = client.chat.completions.create(
            model=QWEN_MODEL,
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
        )
        content = response.choices[0].message.content if response.choices else ""
        reply = _safe_content(content).strip()
    except Exception as exc:
        reply = f"I could not reach the chatbot service right now. Please try again in a moment. ({exc})"

    if not reply:
        reply = "I am here with you. Tell me a little more so I can better support you."

    resources = get_resources(
        emotion=voice_emotion,
        intent=intent,
        contradiction=bool(signal_profile.get("contradiction", False)),
    )

    return {
        "reply": reply,
        "intent": intent,
        "sentiment": sentiment,
        "signal_profile": signal_profile,
        "resources": resources,
        "is_crisis": False,
        "safe_response": "",
    }


def get_opening_message(emotion: str, confidence: float) -> str:
    """Create a short opening message immediately after audio analysis."""
    normalized = (emotion or "neutral").strip().lower()
    confidence_pct = round(float(confidence) * 100, 1)

    opening_map = {
        "neutral": "Hi, it is great to connect with you. What is on your mind today?",
        "calm": "Hi, welcome in. We can take this one step at a time together.",
        "happy": "Hi, I am glad to be here with you. Want to share what is making today feel good?",
        "sad": "Hi, I am here with you. You can share as much or as little as you want.",
        "angry": "Hi, thank you for checking in. We can sort through what is feeling intense right now.",
        "fearful": "Hi, you are safe here. We can slow things down and focus on what helps right now.",
        "disgust": "Hi, thank you for sharing this moment. I am here to listen without judgment.",
        "surprised": "Hi, that sounds like a lot happened quickly. Want to walk me through it?",
    }

    opening = opening_map.get(normalized, opening_map["neutral"])
    return f"{opening} I will keep things concise and helpful. (Context confidence: {confidence_pct}%)"
