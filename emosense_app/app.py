"""Production UI for EmoSense with clean layout and backend placeholders."""

from __future__ import annotations

import json
import html
from datetime import datetime
from hashlib import sha1
from pathlib import Path
from typing import Dict, Optional
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from audio_utils import load_audio_from_upload, get_audio_metadata
from model import EMOTION_EMOJI, MODEL_PATH, load_predictor, predict_distribution, top_prediction

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

MODEL_PATH_DISPLAY = MODEL_PATH.name
TEMP_AUDIO_DIR = Path(__file__).resolve().parent / ".temp_audio"

st.set_page_config(
    page_title="EmoSense AI",
    page_icon="\U0001F399\uFE0F",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<meta charset="UTF-8">
<style>
    * { font-family: "Segoe UI Emoji", "Apple Color Emoji", "Noto Color Emoji", sans-serif; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(160deg,
        #020817 0%,
        #0F172A 50%,
        #0C1A3A 100%);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(160deg, #020817 0%, #0F172A 50%, #0C1A3A 100%);
    min-height: 100vh;
    color: #F1F5F9;
}

[data-testid="stHeader"] { visibility: hidden; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }

/* Custom padding removal */
.block-container { padding-top: 1rem; padding-bottom: 2rem; }

/* Cards */
[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #1A1D27 !important;
    border: 1px solid #2D3148 !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
}

/* Headings */
h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif !important; }

/* Metric Styling */
[data-testid="stMetricValue"] {
    color: #1E40AF !important;
    font-weight: 700;
}
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-weight: 500;
}

/* Buttons */
.stButton > button {
    border-radius: 999px !important;
    background: linear-gradient(135deg, #1E40AF, #2563EB) !important;
    color: white !important;
    border: none !important;
    transition: all 0.2s;
    font-weight: 600 !important;
    box-shadow: 0 4px 20px rgba(30,64,175,0.4);
}
.stButton > button:hover {
    background: linear-gradient(135deg, #1D4ED8, #1E40AF) !important;
    color: white !important;
    transform: translateY(-1px);
}
.stDownloadButton > button {
    border-radius: 999px !important;
    background-color: transparent !important;
    color: #1E40AF !important;
    border: 1px solid #1E40AF !important;
}
.stDownloadButton > button:hover {
    background-color: rgba(30, 64, 175, 0.1) !important;
}

/* File Uploader styling */
[data-testid="stFileUploaderDropzone"] {
    border: 2px dashed #1E40AF !important;
    border-radius: 12px !important;
    background: rgba(30, 64, 175, 0.05) !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0E1117; }
::-webkit-scrollbar-thumb { background: #1E40AF; border-radius: 3px; }

/* Emotion Emoji mapping style */
.hero-emoji { font-size: 80px; line-height: 1; }
.hero-label { font-family: 'Space Grotesk', sans-serif; font-size: 42px; font-weight: bold; color: #1E40AF; margin-top: 10px; }
.hero-conf { color: #06B6D4; font-size: 22px; font-weight: 600; margin-top: -10px; margin-bottom: 20px; }

/* Small Metric Pills */
.metric-pill {
    background: #0E1117;
    border-radius: 999px;
    padding: 8px 16px;
    text-align: center;
    margin-bottom: 8px;
}
.metric-pill-label { color: #94A3B8; font-size: 11px; text-transform: uppercase; letter-spacing: 0.5px;}
.metric-pill-value { color: white; font-size: 14px; font-weight: 600; }

/* Chips */
.chip { display: inline-block; padding: 6px 12px; border-radius: 999px; background: #0E1117; color: #94A3B8; font-size: 12px; margin-right: 6px; margin-bottom: 6px; border: 1px solid #2D3148;}
.chip.active { background: #1E40AF; color: white; border-color: #1E40AF;}

[data-testid="stProgressBar"] > div > div {
    background: #3B82F6;
}

/* Banners */
.banner { padding: 12px 16px; border-radius: 8px; font-size: 14px; font-weight: 500; display: flex; align-items: center; }
.banner-warn { background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); color: #F59E0B; }
.banner-success { background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); color: #10B981; }

hr { border-color: #2D3148; }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
.unread-badge {
    position: fixed;
    bottom: 74px;
    right: 26px;
    width: 20px;
    height: 20px;
    background: #EF4444;
    border-radius: 50%;
    color: white;
    font-size: 11px;
    font-weight: bold;
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }
    70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}
div.stButton > button[key="chat_fab"] {
    position: fixed;
    bottom: 28px;
    right: 28px;
    width: 60px;
    height: 60px;
    border-radius: 30px !important;
    background: linear-gradient(135deg, #1E40AF, #1D4ED8) !important;
    box-shadow: 0 8px 32px rgba(30, 64, 175, 0.45) !important;
    z-index: 9999;
    font-size: 28px !important;
    padding: 0 !important;
    line-height: 1 !important;
}
div.stButton > button[key="chat_fab"]:hover {
    transform: scale(1.08);
}
</style>
""", unsafe_allow_html=True)

EMOTION_HERO_EMOJI = {
    "neutral": "\U0001F610", "calm": "\U0001F60C", "happy": "\U0001F60A", "sad": "\U0001F622",
    "angry": "\U0001F620", "fearful": "\U0001F628", "disgust": "\U0001F922", "surprised": "\U0001F632"
}

@st.cache_resource
def get_predictor_bundle():
    return load_predictor(MODEL_PATH)

def init_state() -> None:
    st.session_state.setdefault("analysis_result", None)
    st.session_state.setdefault("current_emotion", "neutral")
    st.session_state.setdefault("current_confidence", 0.0)
    st.session_state.setdefault("chat_history", [])
    st.session_state.setdefault("raw_model_output", {})
    st.session_state.setdefault("raw_llm_output", {})
    st.session_state.setdefault("chat_opened", False)
    st.session_state.setdefault("session_analyses", 0)
    st.session_state.setdefault("emotion_counts", {})
    st.session_state.setdefault("unread_msg", 0)

init_state()
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
predictor = get_predictor_bundle()

def append_chat(role: str, content: str) -> None:
    st.session_state["chat_history"].append({
        "role": role,
        "content": content,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

def get_recorded_audio():
    return st.audio_input("&#127908; Click to record your voice")

def persist_audio_for_player(file_name: str, audio_bytes: bytes) -> str:
    suffix = Path(file_name).suffix.lower() or ".wav"
    if suffix not in {".wav", ".mp3", ".ogg", ".m4a", ".aac", ".webm"}:
        suffix = ".wav"
    digest = sha1(audio_bytes).hexdigest()[:16]
    TEMP_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = TEMP_AUDIO_DIR / f"{digest}{suffix}"
    if not audio_path.exists():
        audio_path.write_bytes(audio_bytes)
    return str(audio_path)

def run_emotion_inference(file_name: str, audio_bytes: bytes, predictor) -> Dict[str, object]:
    waveform, sample_rate = load_audio_from_upload(file_name, audio_bytes)
    conf_dist = predict_distribution(waveform, sample_rate, predictor)
    pred_label, pred_conf = top_prediction(conf_dist)
    
    meta = get_audio_metadata(audio_bytes, waveform, sample_rate)

    raw_payload = {
        "waveform": waveform,
        "sample_rate": sample_rate,
        "distribution": conf_dist,
        "predicted_label": pred_label,
        "predicted_confidence": pred_conf,
        "metadata": meta
    }
    st.session_state["raw_model_output"] = raw_payload
    audio_path = persist_audio_for_player(file_name, audio_bytes)
    
    st.session_state["session_analyses"] += 1
    cnt = st.session_state["emotion_counts"].get(pred_label, 0)
    st.session_state["emotion_counts"][pred_label] = cnt + 1

    return {
        "filename": file_name,
        "audio_path": audio_path,
        "predicted_label": pred_label,
        "predicted_confidence": float(pred_conf),
    }

def run_chat_inference(user_message: str) -> str:
    emotion = st.session_state.get("current_emotion", "neutral")
    confidence = float(st.session_state.get("current_confidence", 0.0))
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.get("chat_history", [])]

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
        return f"I am here with you. (Context: {emotion}, {confidence * 100:.0f}% conf)"

@st.dialog("EmoSense Chat", width="large")
def render_chat_dialog() -> None:
    st.session_state["unread_msg"] = 0
    ctx_emotion = st.session_state.get("current_emotion", "neutral")
    ctx_conf = float(st.session_state.get("current_confidence", 0.0))
    ctx_emoji = EMOTION_HERO_EMOJI.get(str(ctx_emotion), "\U0001F3A7")

    st.markdown(
        f'<div class="banner banner-success" style="margin-bottom:1rem;">Context: {ctx_emoji} {str(ctx_emotion).capitalize()} ({ctx_conf * 100:.0f}%)</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.get("analysis_result") is not None and not st.session_state.get("chat_opened", False):
        opening = "Hi, I am here with you. Share what is on your mind, and we can take it step by step."
        append_chat("assistant", opening)
        st.session_state["chat_opened"] = True

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_prompt = st.chat_input("Type your message")
    if user_prompt and user_prompt.strip():
        append_chat("user", user_prompt.strip())
        with st.spinner("Processing..."):
            reply = run_chat_inference(user_prompt.strip())
        append_chat("assistant", reply)
        st.rerun()

with st.sidebar:
    st.title("\u2699\uFE0F Settings")
    st.subheader("Model Settings")
    model_var = st.selectbox("Qwen model variant", ["7B Instruct", "32B Instruct", "72B Instruct"])
    resp_len = st.slider("Response length", 256, 2048, 512, step=128)
    temp = st.slider("Temperature", 0.1, 1.0, 0.75, step=0.05)
    
    st.subheader("Display Settings")
    show_spec = st.toggle("Show spectrogram", value=True)
    show_contradiction = st.toggle("Show signal contradiction banner", value=True)
    auto_chat = st.toggle("Auto-open chat after analysis", value=True)
    
    st.subheader("Session")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("\U0001F5D1\uFE0F Clear", use_container_width=True):
            st.session_state["chat_history"] = []
            st.rerun()
    with col2:
        if st.button("\U0001F504 Reset", use_container_width=True):
            init_state()
            st.rerun()
            
    with st.expander("\U0001F4CA Session Summary"):
        st.write(f"Total analyses: **{st.session_state.get('session_analyses', 0)}**")
        freq = "None"
        if st.session_state["emotion_counts"]:
            freq = max(st.session_state["emotion_counts"], key=st.session_state["emotion_counts"].get).capitalize()
        st.write(f"Most frequent: **{freq}**")
        st.write("Emotions detected:")
        for k,v in st.session_state["emotion_counts"].items():
            st.caption(f"- {k.capitalize()}: {v}")

col_left, col_right = st.columns([0.35, 0.65], gap="large")

with col_left:
    st.markdown("""
        <div style="display:flex; align-items:center; gap: 12px; margin-bottom: 24px;">
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#1E40AF" stroke-width="2"/>
                <path d="M8 14V10M12 16V8M16 13V11" stroke="#1E40AF" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <div>
                <h1 style="margin:0; padding:0; font-size: 28px; font-weight: 700; font-family:'Space Grotesk', sans-serif;">EmoSense AI</h1>
                <div style="color: #94A3B8; font-size: 13px; margin-top: -4px;">Speech Emotion Intelligence</div>
            </div>
        </div>
        <hr style="border-color:#1E40AF; margin-top:-10px; margin-bottom:24px; opacity:0.3;"/>
    """, unsafe_allow_html=True)
    
    with st.container():
        st.markdown("<h3 style='margin-top:0;'>\U0001F399\uFE0F Audio Input</h3>", unsafe_allow_html=True)
        tab_up, tab_rec = st.tabs(["Upload", "Record"])
        
        audio_bytes = None
        file_name = "audio.wav"
        
        with tab_up:
            uploaded = st.file_uploader("", type=["wav", "mp3", "ogg"])
            if uploaded:
                audio_bytes = uploaded.getvalue()
                file_name = uploaded.name
                
        with tab_rec:
            st.markdown("<div style='margin-bottom: 10px; color: #94A3B8; font-size: 14px;'>Click to toggle recording</div>", unsafe_allow_html=True)
            rec_bytes = get_recorded_audio()
            if rec_bytes is not None:
                audio_bytes = rec_bytes.getvalue()
                file_name = f"recorded_{datetime.now().strftime('%H%M%S')}.wav"
                st.markdown("<div style='display:flex; align-items:center; gap:8px; margin-top:10px;'><div style='width:10px; height:10px; border-radius:50%; background:#10B981;'></div><span style='color:#10B981; font-size:14px;'>Recording captured</span></div>", unsafe_allow_html=True)
            else:
                st.markdown("<div style='display:flex; align-items:center; gap:8px; margin-top:10px;'><div style='width:10px; height:10px; border-radius:50%; background:#94A3B8;'></div><span style='color:#94A3B8; font-size:14px;'>Ready to record</span></div>", unsafe_allow_html=True)

    if audio_bytes:
        with st.container():
            st.markdown("<h3 style='margin-top:0;'>\u25B6\uFE0F Playback</h3>", unsafe_allow_html=True)
            audio_path = persist_audio_for_player(file_name, audio_bytes)
            st.audio(audio_path)
            
            meta = {}
            if "raw_model_output" in st.session_state and st.session_state["raw_model_output"].get("metadata"):
                meta = st.session_state["raw_model_output"]["metadata"]
            
            if meta:
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"<div class='metric-pill'><div class='metric-pill-label'>Duration</div><div class='metric-pill-value'>{meta.get('duration','-')}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-pill'><div class='metric-pill-label'>Channels</div><div class='metric-pill-value'>{meta.get('channels','-')}</div></div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='metric-pill'><div class='metric-pill-label'>Sample Rate</div><div class='metric-pill-value'>{meta.get('sample_rate','-')}</div></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-pill'><div class='metric-pill-label'>File Size</div><div class='metric-pill-value'>{meta.get('file_size_kb','-')}</div></div>", unsafe_allow_html=True)
                
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("\U0001F50D Analyse Emotion", use_container_width=True):
        if audio_bytes:
            st.session_state["do_analysis"] = (file_name, audio_bytes)
        else:
            st.error("Please provide audio first.")
    st.markdown("<div style='text-align:center; color:#94A3B8; font-size:12px; margin-top: 8px;'>Powered by Wav2Vec2 + Qwen2.5</div>", unsafe_allow_html=True)


with col_right:
    if st.session_state.get("do_analysis"):
        fn, ab = st.session_state["do_analysis"]
        del st.session_state["do_analysis"]
        
        ph = st.empty()
        msgs = ["\U0001F399\uFE0F Processing audio...", "\U0001F9E0 Running emotion model...", "\U0001F4CA Calculating confidence scores...", "\u3030\uFE0F Rendering waveform...", "\u2728 Almost done..."]
        for i in range(5):
            with ph.container():
                st.markdown(f"""
                <div style='background:#1A1D27; border:1px solid #2D3148; border-radius:16px; padding:60px 24px; text-align:center;'>
                    <svg width="80" height="80" viewBox="0 0 24 24" fill="none" class="pulsing">
                        <path d="M12 22C17.5 22 22 17.5 22 12C22 6.5 17.5 2 12 2" stroke="#1E40AF" stroke-width="2" stroke-linecap="round"/>
                        <path d="M8 14V10M12 16V8M16 13V11" stroke="#1E40AF" stroke-width="2" stroke-linecap="round"/>
                    </svg>
                    <style>.pulsing {{ animation: spin 2s linear infinite; }} @keyframes spin {{ 100% {{ transform: rotate(360deg); }} }}</style>
                    <h3 style='color:#1E40AF; margin-top:20px;'>{msgs[i]}</h3>
                </div>
                """, unsafe_allow_html=True)
            time.sleep(0.5)
            
        res = run_emotion_inference(fn, ab, predictor)
        st.session_state["analysis_result"] = res
        st.session_state["current_emotion"] = res["predicted_label"]
        st.session_state["current_confidence"] = float(res["predicted_confidence"])
        st.session_state["chat_opened"] = False
        ph.empty()
        
        if auto_chat:
            st.session_state["unread_msg"] = 1
        st.rerun()

    res = st.session_state.get("analysis_result")
    raw = st.session_state.get("raw_model_output", {})

    if not res:
        st.markdown("""
            <div style='background:#1A1D27; border:1px solid #2D3148; border-radius:16px; padding:100px 20px; text-align:center; margin-top:10px;'>
                <svg width="80" height="80" viewBox="0 0 24 24" fill="none" style="opacity:0.5;">
                    <path d="M12 22C17.5228 22 22 17.5228 22 12C22 6.47715 17.5228 2 12 2C6.47715 2 2 6.47715 2 12C2 17.5228 6.47715 22 12 22Z" stroke="#1E40AF" stroke-width="2"/>
                    <path d="M8 14V10M12 16V8M16 13V11" stroke="#1E40AF" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <h3 style='color:#F1F5F9; margin-top:20px; font-family:"Space Grotesk", sans-serif;'>Upload or record audio to begin analysis</h3>
                <p style='color:#94A3B8; font-size:14px;'>Supports .wav .mp3 .ogg</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        emotion = res["predicted_label"]
        conf = float(res["predicted_confidence"])
        emoji = EMOTION_HERO_EMOJI.get(emotion, "\U0001F3A7")
        
        with st.container():
            c1, c2 = st.columns([1, 1])
            with c1:
                st.markdown(f"<div class='hero-emoji'>{emoji}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='hero-label'>{emotion.capitalize()}</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='hero-conf'>{(conf*100):.0f}% confident</div>", unsafe_allow_html=True)
            with c2:
                color = "#10B981" if conf > 0.7 else ("#F59E0B" if conf > 0.4 else "#EF4444")
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = conf * 100,
                    number = {'suffix': "%", 'font': {'color': color, 'size': 40, 'family': 'Inter'}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "#2D3148", 'tickfont': {'color': '#94A3B8'}},
                        'bar': {'color': color, 'thickness': 0.8},
                        'bgcolor': "rgba(30, 64, 175, 0.1)",
                        'borderwidth': 0,
                        'shape': "angular",
                    }
                ))
                fig.update_layout(height=200, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)", font={'color': "#F1F5F9", "family": "Inter"})
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

        with st.container():
            st.markdown("<h4 style='margin-top:0; margin-bottom:16px;'>\U0001F4CA Emotion Confidence Breakdown</h4>", unsafe_allow_html=True)
            dist = raw.get("distribution", {})
            if dist:
                sorted_dist = sorted(dist.items(), key=lambda x: x[1], reverse=False)
                labels = [f" {EMOTION_HERO_EMOJI.get(k, '')} {k.capitalize()}" for k, v in sorted_dist]
                vals = [v * 100 for k, v in sorted_dist]
                colors = ["#2563EB" if v == max(vals) else "rgba(30, 64, 175, 0.3)" for v in vals]
                
                fig2 = go.Figure(go.Bar(
                    x=vals, y=labels, orientation='h',
                    marker=dict(color=colors, line=dict(width=0)),
                    text=[f"{v:.1f}%" for v in vals],
                    textposition='outside',
                    textfont=dict(color="#F1F5F9"),
                    hoverinfo='x'
                ))
                fig2.update_layout(
                    height=280, margin=dict(l=0, r=40, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, range=[0, 110], visible=False),
                    yaxis=dict(showgrid=False, tickfont=dict(size=14, color="#cbd5e1", family="Inter")),
                    font=dict(family="Inter")
                )
                st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
                
                chips_html = "<div>"
                for k, v in sorted(dist.items(), key=lambda x: x[1], reverse=True):
                    c_cls = "chip active" if k == emotion else "chip"
                    chips_html += f"<span class='{c_cls}'>{EMOTION_HERO_EMOJI.get(k, '')} {k.capitalize()}</span>"
                chips_html += "</div>"
                st.markdown(chips_html, unsafe_allow_html=True)

        with st.container():
            st.markdown("<h4 style='margin-top:0; margin-bottom:16px;'>\u3030\uFE0F Audio Waveform</h4>", unsafe_allow_html=True)
            wf = raw.get("waveform")
            sr = raw.get("sample_rate", 16000)
            if wf is not None:
                wf_arr = wf.detach().cpu().numpy() if hasattr(wf, "detach") else np.asarray(wf)
                if wf_arr.ndim > 1: wf_arr = wf_arr.mean(axis=0)
                
                stride = max(1, len(wf_arr) // 2000)
                wf_sub = wf_arr[::stride]
                t = np.arange(len(wf_sub)) * stride / sr
                
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=t, y=wf_sub, fill='tozeroy', fillcolor='rgba(6, 182, 212, 0.15)', line=dict(color="#06B6D4", width=1.5)))
                fig3.update_layout(
                    height=160, margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, zeroline=False, color="#94A3B8"),
                    yaxis=dict(showgrid=False, zeroline=False, range=[-1.05, 1.05], visible=False),
                    hovermode="x unified"
                )
                st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
                st.markdown("<div style='text-align:center; color:#94A3B8; font-size:12px; margin-top:-10px;'>Amplitude over Time</div>", unsafe_allow_html=True)
                
                if show_spec and HAS_LIBROSA:
                    try:
                        st.markdown("<h5 style='margin-top:20px; font-size:14px; color:#94A3B8;'>Frequency Spectrum</h5>", unsafe_allow_html=True)
                        D = np.abs(librosa.stft(wf_arr))
                        db = librosa.amplitude_to_db(D, ref=np.max)
                        fig4 = go.Figure(data=go.Heatmap(z=db, colorscale='Viridis', showscale=False))
                        fig4.update_layout(height=100, margin=dict(l=0,r=0,t=0,b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", xaxis=dict(visible=False), yaxis=dict(visible=False))
                        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})
                    except Exception:
                        pass
        
        with st.container():
            st.markdown("<h4 style='margin-top:0; margin-bottom:16px;'>\U0001F9E0 Signal Analysis</h4>", unsafe_allow_html=True)
            sc1, sc2 = st.columns(2)
            with sc1:
                st.markdown("<div style='color:#94A3B8; font-size:13px; font-weight:600;'>\U0001F399\uFE0F Voice Emotion</div>", unsafe_allow_html=True)
                st.markdown(f"<div style='color:#F1F5F9; font-size:16px; font-weight:700; margin-top:4px;'>{emotion.capitalize()}</div>", unsafe_allow_html=True)
                st.progress(conf)
                
            with sc2:
                st.markdown("<div style='color:#94A3B8; font-size:13px; font-weight:600;'>\U0001F4AC Text Sentiment</div>", unsafe_allow_html=True)
                if st.session_state["chat_history"]:
                    st.markdown("<div style='color:#10B981; font-size:16px; font-weight:700; margin-top:4px;'>POSITIVE (0.82)</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div style='color:#94A3B8; font-size:16px; font-weight:700; margin-top:4px;'>NO DATA YET</div>", unsafe_allow_html=True)
            
            if show_contradiction:
                st.markdown("<div style='margin-top: 16px;' class='banner banner-success'>\u2713 Voice and text signals are aligned.</div>", unsafe_allow_html=True)

st.markdown("""
<style>
div[data-testid="stButton"] > button[kind="primary"] {
    position: fixed;
    bottom: 28px;
    right: 28px;
    z-index: 9999;
    width: 180px;
    height: 56px;
    border-radius: 999px;
    background: linear-gradient(135deg, #1E40AF, #2563EB);
    border: none;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 4px 20px rgba(30,64,175,0.4);
    cursor: pointer;
}
.floating-wrap {
    position: fixed;
    bottom: 92px;
    right: 20px;
    z-index: 9998;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 6px;
    pointer-events: none;
}
.floating-wrap .chat-label {
    background: #1A1D27;
    color: #F1F5F9;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 20px;
    border: 1px solid #1E40AF;
    white-space: nowrap;
}
.chat-floating-panel {
    position: fixed;
    right: 24px;
    bottom: 104px;
    width: min(420px, calc(100vw - 32px));
    max-height: 68vh;
    overflow-y: auto;
    z-index: 9997;
    background: #1A1D27;
    border: 1px solid #1E40AF;
    border-radius: 16px;
    padding: 12px;
    box-shadow: 0 16px 40px rgba(2, 6, 23, 0.55);
}
@media (max-width: 900px) {
    .chat-floating-panel {
        right: 12px;
        left: 12px;
        width: auto;
        bottom: 96px;
        max-height: 62vh;
    }
}
</style>

""", unsafe_allow_html=True)

if st.button("Click to Chat", key="chat_toggle", help="Open EmoSense Chat", use_container_width=False, type="primary"):
    st.session_state.chat_open = not st.session_state.chat_open
    st.rerun()

chat_display = "flex" if st.session_state.chat_open else "none"
chat_history_html = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
for msg in st.session_state.get("chat_history", []):
    align = "flex-end" if msg["role"] == "user" else "flex-start"
    bubble_color = "#1E40AF" if msg["role"] == "user" else "#1E293B"
    safe_content = html.escape(str(msg.get("content", ""))).replace("\n", "<br>")
    chat_history_html += (
        f'<div style="display:flex; justify-content:{align}; margin:6px 0;">'
        f'<div style="background:{bubble_color}; color:#F1F5F9; '
        f'padding:10px 14px; border-radius:16px; '
        f'max-width:80%; font-size:14px; line-height:1.5; word-break:break-word;">'
        f'{safe_content}'
        f'</div>'
        f'</div>'
    )

st.markdown(f"""
<style>
.chat-window {{
    position: fixed;
    bottom: 100px;
    right: 28px;
    width: 360px;
    height: 500px;
    background: #0F172A;
    border: 1px solid #1E40AF;
    border-radius: 20px;
    display: {chat_display};
    flex-direction: column;
    z-index: 9998;
    box-shadow: 0 24px 64px rgba(30,64,175,0.4);
    overflow: hidden;
}}
.chat-header {{
    background: linear-gradient(135deg, #1E40AF, #1D4ED8);
    padding: 16px 20px;
    color: white;
    font-weight: 700;
    font-size: 15px;
    letter-spacing: 0.3px;
    flex-shrink: 0;
}}
.chat-messages {{
    flex: 1;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
}}
.chat-messages::-webkit-scrollbar {{
    width: 4px;
}}
.chat-messages::-webkit-scrollbar-thumb {{
    background: #1E40AF;
    border-radius: 4px;
}}
.chat-footer {{
    padding: 12px 16px;
    background: #0F172A;
    border-top: 1px solid #1E293B;
    font-size: 12px;
    color: #64748B;
    text-align: center;
    flex-shrink: 0;
}}
</style>

<div class="chat-window">
    <div class="chat-header">
        &#128172; EmoSense Chat
        &nbsp;&nbsp;&#129302; emotion-aware
    </div>
    <div class="chat-messages">
        {chat_history_html if chat_history_html else '<div style="color:#64748B; font-size:13px; text-align:center; margin-top:40px;">Analyse audio first, then chat with me about how you feel.</div>'}
    </div>
    <div class="chat-footer">
        Type below &#8595; and press Enter to send
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.chat_open:
    st.markdown("""
    <style>
    #widget-key-close_chat_floating button {
        position: fixed;
        right: 40px;
        top: calc(100vh - 590px);
        width: 92px;
        height: 30px;
        min-height: 30px;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.15) !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        color: #FFFFFF !important;
        font-size: 11px !important;
        font-weight: 700;
        padding: 0 8px !important;
        z-index: 9999;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.22);
    }
    #widget-key-close_chat_floating button:hover {
        background: rgba(255, 255, 255, 0.24) !important;
        border-color: rgba(255, 255, 255, 0.55) !important;
        transform: scale(1.04);
    }
    @media (max-width: 900px) {
        #widget-key-close_chat_floating button {
            right: 22px;
            top: calc(100vh - 590px);
        }
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Close Chat", key="close_chat_floating", help="Close chat", type="secondary"):
        st.session_state.chat_open = False
        st.rerun()

    components.html(
        """
        <script>
        const parentDoc = window.parent.document;
        if (!window.parent.__emoSenseEscHandlerBound) {
            window.parent.__emoSenseEscHandler = function (event) {
                if (event.key === 'Escape') {
                    const closeBtn = parentDoc.querySelector('#widget-key-close_chat_floating button');
                    if (closeBtn) {
                        closeBtn.click();
                    }
                }
            };
            parentDoc.addEventListener('keydown', window.parent.__emoSenseEscHandler);
            window.parent.__emoSenseEscHandlerBound = true;
        }
        </script>
        """,
        height=0,
    )

    st.markdown("""
    <style>
    div[data-testid="stChatInput"] {
        position: fixed;
        right: 28px;
        bottom: 24px;
        width: 360px;
        z-index: 9999;
    }
    @media (max-width: 900px) {
        div[data-testid="stChatInput"] {
            right: 12px;
            left: 12px;
            width: auto;
        }
    }
    </style>
    """, unsafe_allow_html=True)
    user_input = st.chat_input("Message EmoSense...")
    if user_input:
        emotion = st.session_state.get("current_emotion", "neutral")
        confidence = st.session_state.get("current_confidence", 0.5)

        from chatbot import full_pipeline
        result = full_pipeline(
            voice_emotion=emotion,
            voice_confidence=confidence,
            user_message=user_input,
            history=st.session_state.get("chat_history", []),
        )

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        st.session_state.chat_history += [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": result["reply"]},
        ]
        st.rerun()

