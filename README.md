# Speech Emotion Detection + EmoSense AI

End-to-end speech emotion recognition project built on Wav2Vec2 and the RAVDESS dataset.

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
|-- colab_train_wav2vec2_ravdess.py   # Colab script (optional auto download)
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

## Dataset

Place RAVDESS in this structure:

```text
data/
  Actor_01/
  Actor_02/
  ...
  Actor_24/
```

Example filename:

```text
03-01-05-01-02-01-12.wav
```

The third field is emotion id (`01` to `08`).

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
- Optional mic recording (`streamlit-audiorecorder`)
- Emotion label + confidence meter
- Distribution chart across all emotions
- Waveform visualization
- Downloadable mood report (`.json` / `.txt`)

Model behavior:
- Default checkpoint path is `outputs/model.pth`
- If checkpoint load fails, app falls back to demo mode (mock probabilities)

## Colab Training

```python
!python colab_train_wav2vec2_ravdess.py --download_data
```

Optional custom paths:

```python
!python colab_train_wav2vec2_ravdess.py --download_data --data_dir /content/data --output_dir /content/outputs --epochs 15 --batch_size 16
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
