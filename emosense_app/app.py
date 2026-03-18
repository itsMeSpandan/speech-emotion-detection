"""EmoSense AI Streamlit app for speech emotion detection."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import plotly.express as px
import streamlit as st

from audio_utils import load_audio_from_upload, waveform_figure
from model import EMOTION_EMOJI, MODEL_PATH, load_predictor, predict_distribution, top_prediction
from report import build_mood_report, report_as_json, report_as_text

MODEL_PATH_DISPLAY = MODEL_PATH.name

st.set_page_config(page_title="EmoSense AI", page_icon="🎙️", layout="wide")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
        background: radial-gradient(circle at 10% 10%, #1f2a56 0%, #0f1224 55%, #070a17 100%);
        color: #ecf0ff;
    }

    .app-title {
        display: flex;
        align-items: center;
        gap: 10px;
        font-size: 1.45rem;
        font-weight: 700;
        padding: 10px 14px;
        border-radius: 12px;
        background: linear-gradient(90deg, rgba(102,126,234,0.35), rgba(118,75,162,0.35));
        border: 1px solid rgba(180, 200, 255, 0.2);
        box-shadow: 0 6px 24px rgba(0,0,0,0.25);
    }

    .panel {
        border-radius: 16px;
        padding: 14px;
        background: linear-gradient(180deg, rgba(18,24,48,0.86), rgba(15,18,34,0.9));
        border: 1px solid rgba(120, 148, 255, 0.24);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }

    .result-label {
        font-size: 2rem;
        font-weight: 700;
        text-align: center;
        margin-top: -6px;
    }

    .result-emoji {
        font-size: 5rem;
        text-align: center;
        line-height: 1;
    }

    .badge-ok {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(34, 197, 94, 0.2);
        border: 1px solid rgba(34, 197, 94, 0.7);
        color: #86efac;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .badge-warn {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(245, 158, 11, 0.2);
        border: 1px solid rgba(245, 158, 11, 0.7);
        color: #fcd34d;
        font-weight: 600;
        font-size: 0.85rem;
    }

    .soft-note {
        font-size: 0.9rem;
        opacity: 0.85;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="app-title">🤖 EmoSense AI <span style="opacity:0.8;font-weight:500;">Speech Emotion Detection</span></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Navigation")
    page = st.selectbox(
        "Go to",
        options=["Dashboard", "Home", "About"],
        index=0,
        label_visibility="collapsed",
    )


@st.cache_resource
def get_predictor_bundle():
    """Load model resources once per app process."""
    return load_predictor(MODEL_PATH)


@st.cache_data(show_spinner=False)
def make_report_downloads(payload: Dict[str, object]) -> tuple[str, str]:
    """Cache report serialization to avoid recomputing unchanged payloads."""
    return report_as_json(payload), report_as_text(payload)


def distribution_chart(conf: Dict[str, float], predicted_label: str):
    """Build horizontal confidence distribution chart."""
    display_order = ["angry", "neutral", "happy", "sad", "fearful", "disgust", "surprised", "calm"]
    values = {k: conf.get(k, 0.0) * 100 for k in display_order}

    df = pd.DataFrame(
        {
            "Emotion": [e.capitalize() for e in display_order],
            "Confidence": [values[e] for e in display_order],
            "Highlight": ["Predicted" if e == predicted_label else "Other" for e in display_order],
        }
    )

    fig = px.bar(
        df,
        x="Confidence",
        y="Emotion",
        orientation="h",
        text=df["Confidence"].map(lambda x: f"{x:.1f}%"),
        color="Highlight",
        color_discrete_map={"Predicted": "#7c3aed", "Other": "#4b5563"},
        template="plotly_dark",
        height=360,
    )
    fig.update_layout(
        margin=dict(l=20, r=20, t=10, b=10),
        xaxis_range=[0, 100],
        legend_title_text="",
    )
    fig.update_traces(textposition="outside")
    return fig


def get_recorded_audio() -> tuple[Optional[bytes], str]:
    """Capture microphone audio and return bytes with a generated filename."""
    recorded_bytes: Optional[bytes] = None
    recorded_name = f"recorded_{datetime.now().strftime('%H%M%S')}.wav"

    # Prefer native Streamlit mic capture when available.
    if hasattr(st, "audio_input"):
        mic_clip = st.audio_input("Record Voice")
        if mic_clip is not None:
            recorded_bytes = mic_clip.getvalue()
            st.audio(recorded_bytes, format="audio/wav")
        return recorded_bytes, recorded_name

    # Backward-compatible fallback for older Streamlit versions.
    try:
        from audiorecorder import audiorecorder

        recorded_audio = audiorecorder("Start recording", "Stop recording")
        if len(recorded_audio) > 0:
            recorded_bytes = recorded_audio.export(format="wav").read()
            st.audio(recorded_bytes, format="audio/wav")
        return recorded_bytes, recorded_name
    except Exception:
        st.caption("Microphone input is unavailable in this Streamlit version.")
        st.caption("Install fallback support with: pip install streamlit-audiorecorder")
        return None, recorded_name


if page == "Home":
    st.markdown("### Welcome to EmoSense AI")
    st.markdown(
        "Upload or record speech, run emotion inference, inspect confidence distribution, and export a mood report. "
        f"The app loads the fine-tuned checkpoint from {MODEL_PATH_DISPLAY} by default."
    )
    st.info("Tip: Use the Dashboard tab to analyze audio.")

elif page == "About":
    st.markdown("### About EmoSense AI")
    st.markdown(
        "EmoSense AI is a Streamlit interface for Wav2Vec2-based Speech Emotion Recognition on RAVDESS. "
        f"It supports a strict model path ({MODEL_PATH_DISPLAY}), audio waveform visualization, confidence charts, "
        "and downloadable mood reports."
    )

else:
    st.markdown("### Dashboard")
    st.markdown(
        '<div class="soft-note">Tip: Upload or record a short, clear clip (2-6 seconds) for the best prediction quality.</div>',
        unsafe_allow_html=True,
    )

    predictor = get_predictor_bundle()
    if predictor.is_mock:
        st.error(
            f"Model load failed from {MODEL_PATH_DISPLAY}. "
            f"Running in demo mode with mock predictions. Details: {predictor.error_message}"
        )
    else:
        st.success(f"Loaded model checkpoint from {MODEL_PATH_DISPLAY}")

    left_col, center_col, right_col = st.columns([1.0, 1.05, 1.35], gap="large")

    with left_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Input Section")

        input_mode = st.radio(
            "Choose Input Mode",
            options=["Upload File", "Record Voice"],
            horizontal=True,
        )

        uploaded = None
        recorded_bytes: Optional[bytes] = None
        recorded_name = f"recorded_{datetime.now().strftime('%H%M%S')}.wav"

        if input_mode == "Upload File":
            uploaded = st.file_uploader(
                "Upload Audio File",
                type=["wav", "mp3", "ogg"],
                help="Supported formats: .wav, .mp3, .ogg",
            )
        else:
            st.caption("Use your microphone to record a short audio clip.")
            recorded_bytes, recorded_name = get_recorded_audio()

        analysis_type = st.selectbox(
            "Analysis Type",
            ["Full Analysis", "Emotion Only", "Confidence Only"],
        )

        analyze_clicked = st.button("🎯 Analyze Emotion", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    result = st.session_state.get("analysis_result")

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
            st.error("Please upload or record an audio clip before analysis.")
        else:
            try:
                with st.spinner("Analyzing emotion..."):
                    waveform, sample_rate = load_audio_from_upload(file_name, audio_bytes)
                    conf_dist = predict_distribution(waveform, sample_rate, predictor)
                    pred_label, pred_conf = top_prediction(conf_dist)

                    result = {
                        "filename": file_name,
                        "audio_bytes": audio_bytes,
                        "waveform": waveform,
                        "sample_rate": sample_rate,
                        "distribution": conf_dist,
                        "predicted_label": pred_label,
                        "predicted_confidence": pred_conf,
                        "analysis_type": analysis_type,
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                    }
                    st.session_state["analysis_result"] = result
            except Exception as exc:
                st.error(f"Could not analyze audio. {exc}")

    with center_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Emotion Result")

        if result is None:
            st.info("Run an analysis to see emotion results here.")
        else:
            label = result["predicted_label"]
            conf = float(result["predicted_confidence"])
            emoji = EMOTION_EMOJI.get(label, "🎧")

            st.markdown(f'<div class="result-emoji">{emoji}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-label">{label.capitalize()}</div>',
                unsafe_allow_html=True,
            )

            if result["analysis_type"] != "Emotion Only":
                st.metric("Confidence", f"{conf * 100:.1f}%")

            if result["analysis_type"] == "Confidence Only":
                st.caption("Confidence-only mode selected.")

        st.markdown('</div>', unsafe_allow_html=True)

    with right_col:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Visualizations")

        if result is None:
            st.info("Upload/record audio and click Analyze to see charts.")
        else:
            if result["analysis_type"] != "Emotion Only":
                fig = distribution_chart(result["distribution"], result["predicted_label"])
                st.plotly_chart(fig, use_container_width=True)

            st.audio(result["audio_bytes"], format="audio/wav")
            wf_fig = waveform_figure(result["waveform"], result["sample_rate"])
            st.pyplot(wf_fig, clear_figure=True)
            st.caption("Waveform of uploaded audio")

        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📋 Mood Report", expanded=bool(result)):
        if result is None:
            st.info("Analyze audio to generate a mood report.")
        else:
            report_payload = build_mood_report(
                predicted_emotion=result["predicted_label"],
                confidence=float(result["predicted_confidence"]),
                filename=result["filename"],
                distribution=result["distribution"],
            )
            json_blob, txt_blob = make_report_downloads(report_payload)

            st.markdown("### Summary")
            st.write(
                {
                    "predicted_emotion": report_payload["predicted_emotion"],
                    "confidence_percent": report_payload["confidence_percent"],
                    "timestamp": report_payload["timestamp"],
                    "audio_filename": report_payload["audio_filename"],
                }
            )

            if float(result["predicted_confidence"]) > 0.80:
                st.markdown('<span class="badge-ok">✅ Looks Good!</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span class="badge-warn">⚠️ Low Confidence</span>', unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    label="Download report (.json)",
                    data=json_blob,
                    file_name="mood_report.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with c2:
                st.download_button(
                    label="Download report (.txt)",
                    data=txt_blob,
                    file_name="mood_report.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
