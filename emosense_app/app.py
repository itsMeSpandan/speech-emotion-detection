"""Production UI for EmoSense with clean layout and backend placeholders."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import streamlit as st

from audio_utils import load_audio_from_upload
from model import EMOTION_EMOJI, MODEL_PATH, load_predictor, predict_distribution, top_prediction

MODEL_PATH_DISPLAY = MODEL_PATH.name


st.set_page_config(
    page_title="EmoSense",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    [data-testid="stToolbar"] {display: none;}

    :root {
      --bg-0: #0b1220;
      --bg-1: #111827;
      --bg-2: #1f2937;
      --text-0: #f8fafc;
      --text-1: #cbd5e1;
      --accent: #f97316;
      --accent-soft: rgba(249, 115, 22, 0.18);
      --ok: #22c55e;
      --warn: #f59e0b;
      --border: rgba(148, 163, 184, 0.22);
    }

    html, body, [class*="css"] {
      background: radial-gradient(circle at 12% 12%, #172036 0%, var(--bg-0) 58%, #060a13 100%);
      color: var(--text-0);
    }

    .app-title {
      margin-bottom: 0.65rem;
      padding: 1rem 1.1rem;
      border-radius: 14px;
      border: 1px solid var(--border);
      background: linear-gradient(120deg, rgba(15,23,42,0.82), rgba(17,24,39,0.92));
      box-shadow: 0 14px 34px rgba(0,0,0,0.32);
    }

    .app-title h1 {
      margin: 0;
      font-size: 1.45rem;
      font-weight: 700;
      letter-spacing: 0.2px;
    }

    .app-title p {
      margin: 0.35rem 0 0;
      color: var(--text-1);
      font-size: 0.93rem;
    }

    .panel {
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1rem;
      background: linear-gradient(180deg, rgba(15,23,42,0.85), rgba(17,24,39,0.9));
      box-shadow: 0 12px 28px rgba(0,0,0,0.28);
      height: 100%;
    }

    .section-title {
      font-size: 1rem;
      font-weight: 600;
      margin-bottom: 0.55rem;
      color: var(--text-0);
    }

    .context-chip {
      display: inline-block;
      padding: 0.35rem 0.68rem;
      border-radius: 999px;
      border: 1px solid rgba(249,115,22,0.55);
      background: var(--accent-soft);
      color: #ffedd5;
      font-size: 0.82rem;
      font-weight: 600;
      margin-bottom: 0.75rem;
    }

    .chat-wrap {
      margin-top: 1rem;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1rem;
      background: linear-gradient(180deg, rgba(17,24,39,0.94), rgba(10,16,30,0.96));
      box-shadow: 0 10px 30px rgba(0,0,0,0.26);
    }

    [data-testid="stMetricValue"] {
      color: #fb923c;
      font-weight: 800;
      letter-spacing: 0.2px;
    }

    [data-testid="stMetricLabel"] {
      color: #cbd5e1;
      font-weight: 600;
    }

    [data-testid="stProgressBar"] > div > div {
      background: linear-gradient(90deg, #fb923c, #ea580c);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_predictor_bundle():
    """Load SER model resources once per process."""
    return load_predictor(MODEL_PATH)


def init_state() -> None:
    """Initialize all production session keys."""
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("current_emotion", "neutral")
    st.session_state.setdefault("current_confidence", 0.0)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("raw_model_output", {})
    st.session_state.setdefault("raw_llm_output", {})
    st.session_state.setdefault("chat_opened", False)


def append_chat(role: str, content: str) -> None:
    """Append a chat message for rendering."""
    st.session_state["chat_history"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })


def get_recorded_audio(label: str, key: str) -> tuple[Optional[bytes], str]:
    """Capture microphone audio with Streamlit native input and fallback."""
    recorded_name = f"recorded_{datetime.now().strftime('%H%M%S')}.wav"

    if hasattr(st, "audio_input"):
        clip = st.audio_input(label, key=key)
        if clip is not None:
            audio_bytes = clip.getvalue()
            st.audio(audio_bytes, format="audio/wav")
            return audio_bytes, recorded_name
        return None, recorded_name

    try:
        from audiorecorder import audiorecorder

        recorded_audio = audiorecorder("Start recording", "Stop recording")
        if len(recorded_audio) > 0:
            audio_bytes = recorded_audio.export(format="wav").read()
            st.audio(audio_bytes, format="audio/wav")
            return audio_bytes, recorded_name
    except Exception:
        st.caption("Microphone input unavailable. Install fallback: pip install streamlit-audiorecorder")

    return None, recorded_name


def run_emotion_inference(file_name: str, audio_bytes: bytes, predictor) -> Dict[str, object]:
    """Run model inference and return a compact UI-ready result object."""
    # [INSERT WAV2VEC INFERENCE HERE]
    # Replace this block with your own backend pipeline if needed.
    waveform, sample_rate = load_audio_from_upload(file_name, audio_bytes)
    conf_dist = predict_distribution(waveform, sample_rate, predictor)
    pred_label, pred_conf = top_prediction(conf_dist)

    raw_payload = {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "distribution": conf_dist,
        "predicted_label": pred_label,
        "predicted_confidence": pred_conf,
    }
    st.session_state["raw_model_output"] = raw_payload

    return {
        "filename": file_name,
        "audio_bytes": audio_bytes,
        "predicted_label": pred_label,
        "predicted_confidence": float(pred_conf),
    }


def run_chat_inference(user_message: str) -> str:
    """Run chatbot inference using current emotional context from session state."""
    emotion = st.session_state.get("current_emotion", "neutral")
    confidence = float(st.session_state.get("current_confidence", 0.0))
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.get("chat_history", [])]

    # [INSERT LLM INFERENCE HERE]
    # Replace this block with your own conversational backend if needed.
    try:
        from chatbot import full_pipeline

        payload = full_pipeline(
            voice_emotion=emotion,
            voice_confidence=confidence,
            user_message=user_message,
            history=history,
        )
        st.session_state["raw_llm_output"] = payload

        if payload.get("is_crisis"):
            return str(payload.get("safe_response", "I am here with you."))
        return str(payload.get("reply", "I am here with you."))
    except Exception:
        fallback = (
            f"I am here with you. I am keeping your current context in mind and can continue from here. "
            f"(Context: {emotion}, {confidence * 100:.0f}% confidence)"
        )
        st.session_state["raw_llm_output"] = {"fallback": True}
        return fallback


init_state()
predictor = get_predictor_bundle()


st.markdown(
    """
    <div class="app-title">
      <h1>EmoSense</h1>
      <p>Voice-driven emotional context with conversational support in a clean production dashboard.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


if predictor.is_mock:
    st.error(
        f"Model load failed from {MODEL_PATH_DISPLAY}. Running in demo mode. "
        f"Details are stored internally and hidden from UI."
    )
else:
    st.success(f"Model checkpoint loaded from {MODEL_PATH_DISPLAY}")


left_col, right_col = st.columns([1.02, 1.0], gap="large")

with left_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎛️ Input</div>', unsafe_allow_html=True)

    input_mode = st.radio(
        "Input Mode",
        options=["Upload File", "Record Voice"],
        horizontal=True,
        label_visibility="collapsed",
    )

    uploaded = None
    recorded_bytes: Optional[bytes] = None
    recorded_name = f"recorded_{datetime.now().strftime('%H%M%S')}.wav"

    if input_mode == "Upload File":
        uploaded = st.file_uploader(
            "Upload Audio",
            type=["wav", "mp3", "ogg"],
            help="Supported: .wav, .mp3, .ogg",
        )
    else:
        recorded_bytes, recorded_name = get_recorded_audio(
            label="Record Voice",
            key="dashboard_record_voice",
        )

    analyze_clicked = st.button("Analyze Emotion", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📈 Analysis</div>', unsafe_allow_html=True)

    result = st.session_state.get("analysis_result")
    if result is None:
        st.metric("Primary Emotion", "--")
        st.metric("Confidence", "--")
        st.progress(0)
        st.caption("Run analysis to populate emotion context.")
    else:
        emotion = str(result["predicted_label"])
        confidence = float(result["predicted_confidence"])
        emoji = EMOTION_EMOJI.get(emotion, "🎧")

        st.metric("Primary Emotion", f"{emoji} {emotion.capitalize()}")
        st.metric("Confidence", f"{confidence * 100:.1f}%")
        st.progress(min(max(confidence, 0.0), 1.0))
        st.caption(f"Audio source: {result['filename']}")

    st.markdown("</div>", unsafe_allow_html=True)


if analyze_clicked:
    file_name = ""
    audio_bytes: Optional[bytes] = None

    if uploaded is not None:
        file_name = uploaded.name
        audio_bytes = uploaded.getvalue()
    elif recorded_bytes is not None:
        file_name = recorded_name
        audio_bytes = recorded_bytes

    if audio_bytes is None:
        st.error("Please upload or record audio before analyzing.")
    else:
        with st.spinner("Processing..."):
            try:
                analyzed = run_emotion_inference(file_name, audio_bytes, predictor)
                st.session_state["analysis_result"] = analyzed
                st.session_state["current_emotion"] = analyzed["predicted_label"]
                st.session_state["current_confidence"] = float(analyzed["predicted_confidence"])
                st.session_state["chat_opened"] = False
                st.rerun()
            except Exception:
                st.error("Audio processing failed. Please try a different clip.")


st.markdown('<div class="chat-wrap">', unsafe_allow_html=True)
st.markdown("### 💬 Chat")

ctx_emotion = st.session_state.get("current_emotion", "neutral")
ctx_conf = float(st.session_state.get("current_confidence", 0.0))
ctx_emoji = EMOTION_EMOJI.get(str(ctx_emotion), "🎧")
st.markdown(
    f'<div class="context-chip">Context: {ctx_emoji} {str(ctx_emotion).capitalize()} ({ctx_conf * 100:.0f}%)</div>',
    unsafe_allow_html=True,
)

if st.session_state.get("analysis_result") is not None and not st.session_state.get("chat_opened", False):
    # [INSERT OPENING MESSAGE LOGIC HERE]
    # Keep this block if you want to auto-greet after each fresh analysis.
    opening = "Hi, I am here with you. Share what is on your mind, and we can take it step by step."
    append_chat("assistant", opening)
    st.session_state["chat_opened"] = True

for msg in st.session_state["chat_history"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_prompt = st.chat_input("Type your message...")
if user_prompt:
    append_chat("user", user_prompt)
    with st.spinner("Processing..."):
        reply = run_chat_inference(user_prompt)
    append_chat("assistant", reply)
    st.rerun()

st.markdown("</div>", unsafe_allow_html=True)
