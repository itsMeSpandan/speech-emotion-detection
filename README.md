# Speech Emotion Detection + EmoSense AI

End-to-end speech emotion recognition project built on Wav2Vec2 and the RAVDESS dataset.

## Project Presentation (PPT)

Start here for the project walkthrough and demo storyline:
- [EmoSense AI Pitch Deck](emosenseAI-BYTECLUB.pptx)

## Demo Video

Watch the recorded demo here:
[![Watch the video](https://img.youtube.com/vi/aZuiGAzVEls/maxresdefault.jpg)](https://youtu.be/aZuiGAzVEls)

The repository includes:
- Training and evaluation pipeline (PyTorch)
- Inference utilities for checkpoint-based prediction
- Colab-ready single-file trainer
- EmoSense AI Streamlit app for interactive analysis and report export

## Highlights

- Backbone: `facebook/wav2vec2-base`
- Emotions: `neutral`, `calm`, `happy`, `sad`, `angry`, `fearful`, `disgust`, `surprised`
- Audio standardization: mono, 16 kHz, fixed 3-second clips
- Split strategy: stratified 70/15/15
- Training stability: class-weighted loss, LR scheduling, gradient clipping, early stopping
- Robustness: invalid/corrupt file handling and sample-rate normalization

## Repository Layout

```text
.
|-- main.py                           # Local training + eval entrypoint
|-- dataset.py                        # Dataset and dataloaders
|-- model.py                          # Training-time Wav2Vec2 classifier
|-- train.py                          # Training loop and checkpointing
|-- evaluate.py                       # Metrics and plots
|-- inference.py                      # Inference helpers
|-- colab_train_wav2vec2.py           # Colab script (optional auto download)
|-- requirements.txt                  # Root dependencies
|-- emosense_app/
|   |-- app.py                        # Streamlit UI
|   |-- model.py                      # App-side model loading/predict distribution
|   |-- audio_utils.py                # Upload decoding + waveform rendering
|   |-- report.py                     # Mood report JSON/TXT builders
|   |-- requirements.txt              # App dependencies
|-- data/                             # RAVDESS actor folders
|-- outputs/                          # model.pth + reports + plots
```

## Setup

Recommended Python version: 3.10 or 3.11.

```bash
pip install -r requirements.txt
```

If you only want to run the app environment:

```bash
pip install -r emosense_app/requirements.txt
```

## Setup Secrets

Use a Hugging Face access token for chatbot calls.

Get your token from:
- https://huggingface.co/settings/tokens

For local development:
1. Copy `.env.example` to `.env`.
2. Set your value in `.env`:

```env
HF_TOKEN=your_huggingface_token_here
```

Security rules:
- Never commit `.env`.
- Never commit `.streamlit/secrets.toml` with real credentials.

For Streamlit Cloud deployment:
- Configure `HF_TOKEN` in `.streamlit/secrets.toml` (or Streamlit Cloud Secrets manager).

## Datasets for Fine-Tuning

This project can be fine-tuned using these speech-emotion datasets:

- CREMA-D: Kaggle dataset `ejlok1/cremad`
- RAVDESS: Kaggle dataset `uwrfkaggler/ravdess-emotional-speech-audio`
- SAVEE: Kaggle dataset `barelydedicated/savee-database`

Download method (Kaggle API):

1. Prompt the user to upload `kaggle.json` once at the top of the dataset download section.
2. Use `kaggle.api.dataset_download_files(..., unzip=True)` for each dataset.

Expected download destinations:

- CREMA-D -> `/content/data/CREMA-D/`
- RAVDESS -> `/content/data/RAVDESS/`
- SAVEE -> `/content/data/SAVEE/`

Example Colab snippet:

```python
from google.colab import files
import os
import kaggle

print("Upload kaggle.json")
files.upload()

os.makedirs("/root/.kaggle", exist_ok=True)
os.system("mv kaggle.json /root/.kaggle/kaggle.json")
os.system("chmod 600 /root/.kaggle/kaggle.json")

kaggle.api.dataset_download_files("ejlok1/cremad", path="/content/data/CREMA-D", unzip=True)
kaggle.api.dataset_download_files("uwrfkaggler/ravdess-emotional-speech-audio", path="/content/data/RAVDESS", unzip=True)
kaggle.api.dataset_download_files("barelydedicated/savee-database", path="/content/data/SAVEE", unzip=True)
```

## Train Locally

```bash
python main.py --data_dir data --output_dir outputs
```

Common overrides:

```bash
python main.py --data_dir data --output_dir outputs --batch_size 16 --epochs 15 --lr 1e-4 --weight_decay 1e-5 --patience 3
```

## Artifacts Produced

After training, `outputs/` contains:
- `model.pth`
- `classification_report.txt`
- `confusion_matrix.png`
- `training_curves.png`

## Inference (Python)

```python
import torch
from inference import predict_emotion_from_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
result = predict_emotion_from_checkpoint(
    audio_path="path/to/audio.wav",
    checkpoint_path="outputs/model.pth",
    device=device,
)
print(result)
```

Expected shape:

```python
{"label": "happy", "confidence": 0.92}
```

## Run EmoSense AI (Streamlit)

From repository root:

```bash
streamlit run emosense_app/app.py
```

Features in the app:
- Upload `.wav`, `.mp3`, `.ogg`
- Optional mic recording (`st.audio_input`)
- Emotion label + confidence meter
- Distribution chart across all emotions
- Waveform visualization
- Floating emotion-aware chat window

## Chatbot Integration

The Streamlit app now includes an emotion-aware chatbot powered by Hugging Face Inference API.

- Chat model: `Qwen/Qwen2.5-7B-Instruct`
- Chat helper module: `emosense_app/chatbot.py`
- API client: `huggingface_hub.InferenceClient`

Set your Hugging Face token before launching Streamlit:

PowerShell:

```powershell
$env:HF_TOKEN="your_hf_token_here"
streamlit run emosense_app/app.py
```

Bash:

```bash
export HF_TOKEN="your_hf_token_here"
streamlit run emosense_app/app.py
```

Chat mode in the dashboard:

- `⌨️ Type`: floating chat window using `st.chat_input`

Notes:

- Chat responses are conditioned on current voice emotion + confidence.
- Keep `HF_TOKEN` configured to use Hugging Face inference from the app.

Model behavior:
- Default checkpoint path is `outputs/model.pth`
- If checkpoint load fails, app falls back to demo mode (mock probabilities)

## Methodology, Pitch, and Impact

This project combines multimodal affect understanding (voice + text), intent-aware conversation design, and practical wellbeing support tools in one loop:

- Voice context from Wav2Vec2 emotion recognition
- Text sentiment and intent routing for conversational strategy
- Safety-first crisis interception and helpline-first response path
- Resource recommendation layer tailored by emotion and intent

## Colab Training

```python
!python colab_train_wav2vec2.py --download_data
```

Optional custom paths:

```python
!python colab_train_wav2vec2.py --download_data --data_dir /content/data --output_dir /content/outputs --epochs 15 --batch_size 16
```

## Troubleshooting

1. No audio files found:

```bash
python main.py --data_dir "<ravdess_root_path>"
```

2. Streamlit app says model load failed:
- Make sure `outputs/model.pth` exists and matches the project code.

3. CPU-only execution is slow:
- Reduce batch size and epochs during experimentation.

## License and Dataset Usage

Use this repository according to the licenses of all included dependencies and datasets.
RAVDESS has its own usage terms and citation requirements.
