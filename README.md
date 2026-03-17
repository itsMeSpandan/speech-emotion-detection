# Speech Emotion Detection using Wav2Vec2 (PyTorch)

A modular Speech Emotion Recognition (SER) system built with PyTorch and a pretrained `facebook/wav2vec2-base` backbone, trained on the RAVDESS speech dataset.

This repository includes:
- Local training and evaluation pipeline
- Colab-ready one-file trainer
- Inference utilities for new audio files
- Plots and metrics (confusion matrix, classification report, training curves)

## Features

- Pretrained Wav2Vec2 encoder + custom classification head
- 8-class RAVDESS emotion mapping:
  - neutral, calm, happy, sad, angry, fearful, disgust, surprised
- Fixed-length 3s audio preprocessing at 16 kHz
- Stratified train/val/test split (70/15/15)
- Early stopping, LR scheduler, gradient clipping
- Class imbalance handling with weighted loss
- Robust handling for:
  - corrupted audio
  - empty files
  - sample-rate mismatch (resampling)

## Project Structure

- `dataset.py` - Dataset and preprocessing integration
- `model.py` - Wav2Vec2 + classifier head
- `train.py` - Training loop, early stopping, checkpointing
- `evaluate.py` - Metrics and plotting utilities
- `inference.py` - Load model and predict emotions
- `utils.py` - Shared helpers (seed, audio utils, splits)
- `main.py` - Local end-to-end training entrypoint
- `colab_train_wav2vec2_ravdess.py` - Single-script Colab trainer
- `requirements.txt` - Python dependencies

## Environment

- Python >= 3.9 (recommended: 3.10/3.11)
- PyTorch + torchaudio

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset (RAVDESS Speech)

Expected layout:

```text
data/
  Actor_01/
  Actor_02/
  ...
  Actor_24/
```

Filename format example:

```text
03-01-05-01-02-01-12.wav
```

The third field is emotion ID:
- `01` neutral
- `02` calm
- `03` happy
- `04` sad
- `05` angry
- `06` fearful
- `07` disgust
- `08` surprised

## Local Training (Windows/Linux)

Run end-to-end pipeline:

```bash
python main.py --data_dir data --output_dir outputs
```

Optional overrides:

```bash
python main.py \
  --data_dir data \
  --output_dir outputs \
  --batch_size 16 \
  --epochs 15 \
  --lr 1e-4 \
  --weight_decay 1e-5 \
  --patience 3
```

## Colab Training (Recommended)

Use `colab_train_wav2vec2_ravdess.py` in Google Colab.

1. Install dependencies:

```python
!pip install -q torch torchaudio transformers librosa soundfile numpy scikit-learn matplotlib seaborn tqdm
```

2. Run training (with automatic dataset download):

```python
!python colab_train_wav2vec2_ravdess.py --download_data
```

3. Optional custom paths:

```python
!python colab_train_wav2vec2_ravdess.py \
  --download_data \
  --data_dir /content/data \
  --output_dir /content/outputs \
  --epochs 15 \
  --batch_size 16
```

Note: The Colab script ignores notebook-injected kernel args (like `-f ...kernel.json`) and works in both CLI and notebook contexts.

## Output Artifacts

After training, the output directory contains:

- `model.pth` (best checkpoint by validation loss)
- `classification_report.txt`
- `confusion_matrix.png`
- `training_curves.png`

Checkpoint format:

```python
torch.save({
  "model_state_dict": model.state_dict(),
  "label2id": label2id,
  "best_val_loss": best_val_loss
}, "model.pth")
```

## Inference

You can run inference using utilities in `inference.py`:

- `load_ser_model(...)`
- `predict_emotion(...)`
- `predict_emotion_from_checkpoint(...)`

Expected prediction output:

```python
{
  "label": "happy",
  "confidence": 0.92
}
```

## Download Trained Model from Colab

After Colab training, download the checkpoint:

```python
from google.colab import files
files.download('/content/outputs/model.pth')
```

Then place it in this repository (for example `outputs/model.pth`) and use local inference.

## Reproducibility

Seed is set to `42` for:
- `random`
- `numpy`
- `torch`

Device auto-detection:
- `cuda` if available
- otherwise `cpu`

## Troubleshooting

### 1) `No RAVDESS WAV files found in 'data'`

Set the right dataset path:

```bash
python main.py --data_dir "<path_to_ravdess_root>"
```

### 2) Colab/Jupyter error: `unrecognized arguments: -f ...kernel.json`

Already handled in the Colab script via `parse_known_args()`.

### 3) Hugging Face rate limits

Optionally set token:

```bash
export HF_TOKEN=your_token
```

or in PowerShell:

```powershell
$env:HF_TOKEN="your_token"
```

## Expected Performance

On RAVDESS, Wav2Vec2 fine-tuning can typically reach around `70% - 85%` accuracy depending on split, compute budget, and hyperparameters.

## License

Use according to the licenses of this repository and all external dependencies/datasets (RAVDESS, Hugging Face models, PyTorch ecosystem).
