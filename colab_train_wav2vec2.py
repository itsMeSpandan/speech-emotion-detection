import os

import re

import inspect

from functools import lru_cache

import json

import random

import shutil

import zipfile

import subprocess

from pathlib import Path

from typing import Dict, List, Optional, Tuple, Any



CONFIG: Dict[str, Any] = {

    "seed": 42,

    "sampling_rate": 16000,

    "max_duration": 4,

    "batch_size": 16,

    "epochs": 15,

    "learning_rate": 3e-5,

    "freeze_epochs": 3,

    "num_labels": 8,

    "data_root": "/content/data",

}

CONFIG.update(

    {

        "downloads_root": f'{CONFIG["data_root"]}/_downloads',

        "model_dir": "/content/wav2vec2-ser-final",

        "confusion_matrix_path": "/content/confusion_matrix.png",

        "training_curves_path": "/content/training_curves.png",

        "kaggle_dir": "/root/.kaggle",

        "backbone_name": "facebook/wav2vec2-base",

        "audio_cache_size": 512,

        "num_workers": 2,

    }

)



os.makedirs(CONFIG["data_root"], exist_ok=True)

os.makedirs(CONFIG["downloads_root"], exist_ok=True)



def install_packages() -> None:

    packages: List[str] = [

        "transformers",

        "datasets",

        "accelerate",

        "librosa",

        "soundfile",

        "tqdm",

        "kaggle",

        "scikit-learn",

        "evaluate",

        "seaborn",

        "matplotlib",

        "pandas",

        "numpy",

    ]

    cmd: List[str] = [os.sys.executable, "-m", "pip", "install", "-q"] + packages

    subprocess.run(cmd, check=False)



install_packages()



import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import seaborn as sns

import matplotlib.pyplot as plt

import librosa



from tqdm.auto import tqdm

from datasets import Dataset

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    accuracy_score,

    f1_score,

    classification_report,

    confusion_matrix,

)

from transformers import (

    Wav2Vec2FeatureExtractor,

    Wav2Vec2Model,

    Trainer,

    TrainingArguments,

    TrainerCallback,

    EarlyStoppingCallback,

    set_seed,

)



def set_all_seeds(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    set_seed(seed)



set_all_seeds(CONFIG["seed"])



has_gpu: bool = torch.cuda.is_available()

gpu_name: str = torch.cuda.get_device_name(0) if has_gpu else "CPU"

use_fp16: bool = bool(has_gpu)



print(f"torch.cuda.is_available(): {has_gpu}")

print(f"GPU: {gpu_name}")

print(f"Mixed precision enabled: {use_fp16}")



KAGGLE_USERNAME: str = "nadnapsuknw"

KAGGLE_KEY: str = "KGAT_dadc303a05dc528854dbdf50b1402283"



def setup_kaggle() -> bool:

    kaggle_dir: str = CONFIG["kaggle_dir"]

    os.makedirs(kaggle_dir, exist_ok=True)

    kaggle_json_path: str = os.path.join(kaggle_dir, "kaggle.json")



    try:

        with open(kaggle_json_path, "w", encoding="utf-8") as f:

            json.dump({"username": KAGGLE_USERNAME, "key": KAGGLE_KEY}, f)

    except (FileNotFoundError, OSError) as e:

        print(f"Failed to create kaggle.json: {e}")

        return False



    try:

        os.chmod(kaggle_json_path, 0o600)

    except OSError:

        pass



    os.environ["KAGGLE_CONFIG_DIR"] = kaggle_dir

    test = subprocess.run(

        ["kaggle", "--version"],

        capture_output=True,

        text=True,

        check=False,

    )

    if test.returncode != 0:

        print("Kaggle CLI not ready. Please verify kaggle.json.")

        return False

    print(f"Kaggle CLI: {test.stdout.strip()}")

    return True



DATASETS: Dict[str, str] = {

    "cremad": "ejlok1/cremad",

    "ravdess": "uwrfkaggler/ravdess-emotional-speech-audio",

    "savee": "barelydedicated/savee-database",

}



def extract_zip_with_progress(zip_path: str, out_dir: str, desc: str) -> None:

    os.makedirs(out_dir, exist_ok=True)

    try:

        with zipfile.ZipFile(zip_path, "r") as zf:

            members: List[str] = zf.namelist()

            for member in tqdm(members, desc=desc, leave=False):

                try:

                    zf.extract(member, out_dir)

                except OSError:

                    continue

    except (FileNotFoundError, OSError, zipfile.BadZipFile) as e:

        print(f"Extraction failed for {zip_path}: {e}")



def count_wavs(root: str) -> int:

    return sum(1 for _ in Path(root).rglob("*.wav"))



def download_and_prepare_dataset(dataset_name: str, kaggle_slug: str) -> int:

    target_dir: str = os.path.join(CONFIG["data_root"], dataset_name)

    download_dir: str = os.path.join(CONFIG["downloads_root"], dataset_name)

    os.makedirs(target_dir, exist_ok=True)

    os.makedirs(download_dir, exist_ok=True)



    existing_count: int = count_wavs(target_dir)

    if existing_count > 0:

        print(f"[{dataset_name}] already exists, wav files: {existing_count}")

        return existing_count



    for step in tqdm(["download", "extract"], desc=f"{dataset_name}"):

        if step == "download":

            cmd: List[str] = [

                "kaggle",

                "datasets",

                "download",

                "-d",

                kaggle_slug,

                "-p",

                download_dir,

                "--force",

            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            if result.returncode != 0:

                print(f"[{dataset_name}] download failed: {result.stderr.strip()}")

                return 0



        if step == "extract":

            zip_files: List[str] = [str(p) for p in Path(download_dir).glob("*.zip")]

            if not zip_files:

                print(f"[{dataset_name}] no zip found in {download_dir}")

                return 0

            for zip_file in zip_files:

                extract_zip_with_progress(zip_file, target_dir, desc=f"extract {dataset_name}")



    final_count: int = count_wavs(target_dir)

    print(f"[{dataset_name}] wav files: {final_count}")

    return final_count



kaggle_ready: bool = setup_kaggle()

dataset_counts: Dict[str, int] = {}



if kaggle_ready:

    for ds_name, ds_slug in tqdm(DATASETS.items(), desc="Datasets"):

        dataset_counts[ds_name] = download_and_prepare_dataset(ds_name, ds_slug)

else:

    print("Skipping downloads because Kaggle setup failed.")

    for ds_name in DATASETS:

        dataset_counts[ds_name] = count_wavs(os.path.join(CONFIG["data_root"], ds_name))



print("\nDataset summary:")

for k, v in dataset_counts.items():

    print(f"{k}: {v} wav files")



LABELS: List[str] = [

    "neutral",

    "calm",

    "happy",

    "sad",

    "angry",

    "fearful",

    "disgust",

    "surprised",

]

label_to_id: Dict[str, int] = {label: i for i, label in enumerate(LABELS)}

id_to_label: Dict[int, str] = {i: label for label, i in label_to_id.items()}



CREMAD_MAP: Dict[str, str] = {

    "NEU": "neutral",

    "HAP": "happy",

    "SAD": "sad",

    "ANG": "angry",

    "FEA": "fearful",

    "DIS": "disgust",

}



RAVDESS_MAP: Dict[str, str] = {

    "01": "neutral",

    "02": "calm",

    "03": "happy",

    "04": "sad",

    "05": "angry",

    "06": "fearful",

    "07": "disgust",

    "08": "surprised",

}



def parse_cremad_label(filename: str) -> Optional[str]:

    stem = Path(filename).stem

    parts = stem.split("_")

    if len(parts) < 3:

        return None

    code = parts[2].upper()

    return CREMAD_MAP.get(code)



def parse_ravdess_label(filename: str) -> Optional[str]:

    stem = Path(filename).stem

    parts = stem.split("-")

    if len(parts) < 3:

        return None

    code = parts[2]

    return RAVDESS_MAP.get(code)



def parse_savee_label(filename: str) -> Optional[str]:

    stem = Path(filename).stem.lower()



    patterns: List[Tuple[str, str]] = [

        (r"_a\d+", "angry"),

        (r"_d\d+", "disgust"),

        (r"_f\d+", "fearful"),

        (r"_h\d+", "happy"),

        (r"_n\d+", "neutral"),

        (r"_sa\d+", "sad"),

        (r"_su\d+", "surprised"),

    ]



    for pattern, label in patterns:

        if re.search(pattern, stem):

            return label



    tokens = re.split(r"[_\-]", stem)

    token_map: Dict[str, str] = {

        "a": "angry",

        "d": "disgust",

        "f": "fearful",

        "h": "happy",

        "n": "neutral",

        "sa": "sad",

        "su": "surprised",

    }

    for token in tokens:

        if token in token_map:

            return token_map[token]

    return None



def parse_tess_label(filename: str) -> Optional[str]:

    stem = Path(filename).stem.lower()

    parent = Path(filename).parent.name.lower()

    candidate = f"{parent}_{stem}"



    if "angry" in candidate:

        return "angry"

    if "disgust" in candidate:

        return "disgust"

    if "fear" in candidate:

        return "fearful"

    if "happy" in candidate:

        return "happy"

    if "neutral" in candidate:

        return "neutral"

    if re.search(r"\bps\b", candidate) or "surprise" in candidate:

        return "surprised"

    if "sad" in candidate:

        return "sad"

    return None



def gather_records() -> pd.DataFrame:

    records: List[Dict[str, str]] = []



    parser_map = {

        "cremad": parse_cremad_label,

        "ravdess": parse_ravdess_label,

        "savee": parse_savee_label,

    }



    for source, parser in parser_map.items():

        source_root = os.path.join(CONFIG["data_root"], source)

        wav_paths: List[Path] = list(Path(source_root).rglob("*.wav"))

        for wav_path in tqdm(wav_paths, desc=f"Parsing labels: {source}"):

            label = parser(str(wav_path))

            if label not in label_to_id:

                continue

            records.append(

                {

                    "path": str(wav_path),

                    "label": label,

                    "source": source,

                }

            )



    df = pd.DataFrame(records, columns=["path", "label", "source"])

    return df



df_raw: pd.DataFrame = gather_records()



print(f"\nTotal samples: {len(df_raw)}")

if len(df_raw) > 0:

    print("Class distribution:")

    print(df_raw["label"].value_counts().reindex(LABELS, fill_value=0))

else:

    print("No valid samples found.")



feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(CONFIG["backbone_name"])

target_len: int = int(CONFIG["sampling_rate"] * CONFIG["max_duration"])



def load_audio_fixed(path: str) -> Optional[np.ndarray]:

    try:

        audio, _ = librosa.load(path, sr=CONFIG["sampling_rate"], mono=True)

    except (FileNotFoundError, OSError):

        return None

    except Exception:

        return None



    if audio is None or len(audio) == 0:

        return None



    if len(audio) < target_len:

        pad_width = target_len - len(audio)

        audio = np.pad(audio, (0, pad_width), mode="constant")

    else:

        audio = audio[:target_len]



    return audio.astype(np.float32)



@lru_cache(maxsize=CONFIG["audio_cache_size"])
def load_audio_fixed_cached(path: str) -> Optional[np.ndarray]:

    return load_audio_fixed(path)



def filter_existing_files(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:

    keep_rows: List[Dict[str, Any]] = []

    missing_count: int = 0

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Checking files"):

        if os.path.isfile(row.path):

            keep_rows.append({"path": row.path, "label": row.label, "source": row.source})

        else:

            missing_count += 1

    return pd.DataFrame(keep_rows), missing_count



df_model, missing_file_count = filter_existing_files(df_raw)

if len(df_model) == 0:

    raise RuntimeError("No readable file paths found after dataset parsing.")



df_model["labels"] = df_model["label"].map(label_to_id).astype(int)



print(f"\nUsable samples: {len(df_model)}")

print(f"Missing files skipped: {missing_file_count}")



if len(df_model) < 20:

    print("Warning: Very small dataset after filtering. Training quality may be poor.")



def split_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    try:

        train_df, temp_df = train_test_split(

            df,

            test_size=0.2,

            random_state=CONFIG["seed"],

            stratify=df["labels"],

        )

        val_df, test_df = train_test_split(

            temp_df,

            test_size=0.5,

            random_state=CONFIG["seed"],

            stratify=temp_df["labels"],

        )

        return train_df, val_df, test_df

    except ValueError as e:

        print(f"Stratified split warning: {e}")

        train_df, temp_df = train_test_split(

            df,

            test_size=0.2,

            random_state=CONFIG["seed"],

            stratify=df["labels"] if df["labels"].nunique() > 1 else None,

        )

        val_df, test_df = train_test_split(

            temp_df,

            test_size=0.5,

            random_state=CONFIG["seed"],

            stratify=None,

        )

        return train_df, val_df, test_df



train_df, val_df, test_df = split_dataframe(df_model)



def print_split_stats(name: str, split_df: pd.DataFrame) -> None:

    dist = split_df["labels"].value_counts().sort_index()

    print(f"\n{name} size: {len(split_df)}")

    for i in range(len(LABELS)):

        print(f"  {LABELS[i]}: {int(dist.get(i, 0))}")



print_split_stats("Train", train_df)

print_split_stats("Validation", val_df)

print_split_stats("Test", test_df)



def to_hf_dataset(split_df: pd.DataFrame) -> Dataset:

    cols = ["path", "labels"]

    ds = Dataset.from_pandas(split_df[cols], preserve_index=False)

    return ds



train_dataset: Dataset = to_hf_dataset(train_df)

val_dataset: Dataset = to_hf_dataset(val_df)

test_dataset: Dataset = to_hf_dataset(test_df)


class SERDataCollator:

    def __init__(self, extractor: Wav2Vec2FeatureExtractor, max_len: int) -> None:

        self.extractor = extractor

        self.max_len = max_len

        self.bad_file_count: int = 0



    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

        audios: List[np.ndarray] = []

        labels: List[int] = []



        for item in features:

            audio = load_audio_fixed_cached(str(item["path"]))

            if audio is None:

                self.bad_file_count += 1

                audio = np.zeros(self.max_len, dtype=np.float32)

            audios.append(audio)

            labels.append(int(item["labels"]))



        batch = self.extractor(

            audios,

            sampling_rate=CONFIG["sampling_rate"],

            max_length=self.max_len,

            truncation=True,

            padding="max_length",

            return_attention_mask=True,

            return_tensors="pt",

        )



        return {

            "input_values": batch["input_values"].float(),

            "attention_mask": batch["attention_mask"].long(),

            "labels": torch.tensor(labels, dtype=torch.long),

        }



data_collator = SERDataCollator(feature_extractor, target_len)



class Wav2Vec2ForSER(nn.Module):

    def __init__(self, num_labels: int, backbone_name: str, dropout: float = 0.25) -> None:

        super().__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(backbone_name)

        hidden_size: int = int(self.wav2vec2.config.hidden_size)

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(hidden_size, num_labels)



    def freeze_encoder(self) -> None:

        for p in self.wav2vec2.parameters():

            p.requires_grad = False



    def unfreeze_encoder(self) -> None:

        for p in self.wav2vec2.encoder.parameters():

            p.requires_grad = True

        for p in self.wav2vec2.feature_projection.parameters():

            p.requires_grad = True



    def forward(

        self,

        input_values: torch.Tensor,

        attention_mask: Optional[torch.Tensor] = None,

        labels: Optional[torch.Tensor] = None,

        **kwargs: Any,

    ) -> torch.Tensor:

        outputs = self.wav2vec2(input_values=input_values, attention_mask=attention_mask)

        hidden = outputs.last_hidden_state



        if attention_mask is not None:

            feature_mask: Optional[torch.Tensor] = None

            # attention_mask from the extractor is at waveform length; map it to feature length.

            if attention_mask.shape[1] == hidden.shape[1]:

                feature_mask = attention_mask

            elif hasattr(self.wav2vec2, "_get_feature_vector_attention_mask"):

                try:

                    feature_mask = self.wav2vec2._get_feature_vector_attention_mask(

                        hidden.shape[1],

                        attention_mask,

                    )

                except Exception:

                    feature_mask = None



            if feature_mask is not None:

                mask = feature_mask.unsqueeze(-1).expand_as(hidden).float()

                summed = (hidden * mask).sum(dim=1)

                denom = mask.sum(dim=1).clamp(min=1e-6)

                pooled = summed / denom

            else:

                pooled = hidden.mean(dim=1)

        else:

            pooled = hidden.mean(dim=1)



        logits = self.classifier(self.dropout(pooled))

        return logits



model = Wav2Vec2ForSER(

    num_labels=CONFIG["num_labels"],

    backbone_name=CONFIG["backbone_name"],

    dropout=0.25,

)

model.freeze_encoder()



def count_params(m: nn.Module) -> Tuple[int, int]:

    total = sum(p.numel() for p in m.parameters())

    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)

    return total, trainable



total_params, trainable_params = count_params(model)

print(f"\nTotal params: {total_params:,}")

print(f"Trainable params: {trainable_params:,}")



def compute_metrics(eval_pred: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:

    logits, labels = eval_pred

    if isinstance(logits, tuple):

        logits = logits[0]

    preds = np.argmax(logits, axis=-1)

    return {

        "accuracy": float(accuracy_score(labels, preds)),

        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),

        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),

    }



class SERTrainer(Trainer):

    def create_optimizer(self) -> torch.optim.Optimizer:

        if self.optimizer is None:

            self.optimizer = torch.optim.AdamW(

                self.model.parameters(),

                lr=self.args.learning_rate,

                weight_decay=self.args.weight_decay,

            )

        return self.optimizer



    def compute_loss(

        self,

        model: nn.Module,

        inputs: Dict[str, torch.Tensor],

        return_outputs: bool = False,

        num_items_in_batch: Optional[int] = None,

    ) -> Any:

        labels = inputs["labels"]

        model_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        logits = model(**model_inputs)

        loss = nn.CrossEntropyLoss()(logits, labels)

        if return_outputs:

            return loss, {"logits": logits}

        return loss



class FreezeSchedulerCallback(TrainerCallback):

    def __init__(self, freeze_epochs: int = 3) -> None:

        self.freeze_epochs = freeze_epochs

        self.unfrozen = False



    def on_epoch_begin(self, args, state, control, model=None, **kwargs):

        if self.unfrozen:

            return control

        if state.epoch is None:

            return control

        if float(state.epoch) >= float(self.freeze_epochs):

            if model is not None and hasattr(model, "unfreeze_encoder"):

                model.unfreeze_encoder()

                self.unfrozen = True

                print(f"Encoder unfrozen at epoch {state.epoch:.2f}")

        return control



def build_training_args() -> TrainingArguments:

    sig = inspect.signature(TrainingArguments.__init__).parameters



    kwargs: Dict[str, Any] = {

        "output_dir": "/content/ser_outputs",

        "num_train_epochs": CONFIG["epochs"],

        "per_device_train_batch_size": CONFIG["batch_size"],

        "per_device_eval_batch_size": CONFIG["batch_size"],

        "learning_rate": CONFIG["learning_rate"],

        "load_best_model_at_end": True,

        "metric_for_best_model": "eval_macro_f1",

        "greater_is_better": True,

        "fp16": use_fp16,

        "report_to": [],

        "save_total_limit": 2,

        "seed": CONFIG["seed"],

    }



    if "evaluation_strategy" in sig:

        kwargs["evaluation_strategy"] = "epoch"

    elif "eval_strategy" in sig:

        kwargs["eval_strategy"] = "epoch"



    if "save_strategy" in sig:

        kwargs["save_strategy"] = "epoch"

    if "logging_strategy" in sig:

        kwargs["logging_strategy"] = "epoch"

    if "dataloader_num_workers" in sig:

        cpu_count = os.cpu_count() or 1

        kwargs["dataloader_num_workers"] = min(CONFIG["num_workers"], cpu_count)

    if "dataloader_pin_memory" in sig:

        kwargs["dataloader_pin_memory"] = has_gpu

    if "eval_accumulation_steps" in sig:

        kwargs["eval_accumulation_steps"] = 8

    if "remove_unused_columns" in sig:

        kwargs["remove_unused_columns"] = False



    # Some old transformers builds don't support report_to; fall back cleanly.
    try:

        return TrainingArguments(**kwargs)

    except TypeError:

        kwargs.pop("report_to", None)

        return TrainingArguments(**kwargs)



training_args = build_training_args()



if has_gpu:

    gpu_mem_gb: float = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

    safe_batch_size = CONFIG["batch_size"]

    if gpu_mem_gb < 16:

        safe_batch_size = min(safe_batch_size, 8)

    if gpu_mem_gb < 10:

        safe_batch_size = min(safe_batch_size, 4)

    if safe_batch_size != training_args.per_device_train_batch_size:

        training_args.per_device_train_batch_size = safe_batch_size

        training_args.per_device_eval_batch_size = safe_batch_size

        training_args.gradient_accumulation_steps = max(1, CONFIG["batch_size"] // safe_batch_size)

        print(

            f"Auto-tuned batch size for memory safety: {safe_batch_size} "

            f"(gradient_accumulation_steps={training_args.gradient_accumulation_steps})"

        )



trainer = SERTrainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=val_dataset,

    data_collator=data_collator,

    compute_metrics=compute_metrics,

    callbacks=[

        FreezeSchedulerCallback(freeze_epochs=CONFIG["freeze_epochs"]),

        EarlyStoppingCallback(early_stopping_patience=3),

    ],

)



train_output = trainer.train()

print("\nTraining completed.")

print(train_output)



pred_output = trainer.predict(test_dataset)

test_logits = pred_output.predictions[0] if isinstance(pred_output.predictions, tuple) else pred_output.predictions

y_true = np.array(test_dataset["labels"])

y_pred = np.argmax(test_logits, axis=-1)



print("\nClassification Report:")

print(

    classification_report(

        y_true,

        y_pred,

        labels=list(range(CONFIG["num_labels"])),

        target_names=LABELS,

        zero_division=0,

    )

)



cm = confusion_matrix(y_true, y_pred, labels=list(range(CONFIG["num_labels"])))

plt.figure(figsize=(10, 8))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)

plt.title("Confusion Matrix")

plt.xlabel("Predicted")

plt.ylabel("True")

plt.tight_layout()

plt.savefig(CONFIG["confusion_matrix_path"], dpi=200)

plt.close()

print(f"Saved confusion matrix: {CONFIG['confusion_matrix_path']}")



history: List[Dict[str, Any]] = trainer.state.log_history

train_epochs: List[float] = []

train_losses: List[float] = []

eval_epochs: List[float] = []

eval_losses: List[float] = []

eval_accs: List[float] = []



for entry in history:

    if "loss" in entry and "epoch" in entry and "eval_loss" not in entry:

        train_epochs.append(float(entry["epoch"]))

        train_losses.append(float(entry["loss"]))

    if "eval_loss" in entry and "epoch" in entry:

        eval_epochs.append(float(entry["epoch"]))

        eval_losses.append(float(entry["eval_loss"]))

        eval_accs.append(float(entry.get("eval_accuracy", np.nan)))



plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)

if train_epochs and train_losses:

    plt.plot(train_epochs, train_losses, marker="o", label="train_loss")

if eval_epochs and eval_losses:

    plt.plot(eval_epochs, eval_losses, marker="o", label="val_loss")

plt.title("Loss Curves")

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()



plt.subplot(1, 2, 2)

if eval_epochs and eval_accs:

    plt.plot(eval_epochs, eval_accs, marker="o", label="val_accuracy")

plt.title("Validation Accuracy")

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.tight_layout()

plt.savefig(CONFIG["training_curves_path"], dpi=200)

plt.close()

print(f"Saved training curves: {CONFIG['training_curves_path']}")



def save_ser_model(

    trained_model: Wav2Vec2ForSER,

    extractor: Wav2Vec2FeatureExtractor,

    save_dir: str,

) -> None:

    os.makedirs(save_dir, exist_ok=True)

    torch.save(trained_model.state_dict(), os.path.join(save_dir, "pytorch_model.bin"))

    payload = {

        "num_labels": CONFIG["num_labels"],

        "labels": LABELS,

        "label_to_id": label_to_id,

        "id_to_label": {str(k): v for k, v in id_to_label.items()},

        "backbone_name": CONFIG["backbone_name"],

        "sampling_rate": CONFIG["sampling_rate"],

        "max_duration": CONFIG["max_duration"],

    }

    with open(os.path.join(save_dir, "ser_config.json"), "w", encoding="utf-8") as f:

        json.dump(payload, f, indent=2)

    extractor.save_pretrained(save_dir)



save_ser_model(trainer.model, feature_extractor, CONFIG["model_dir"])

print(f"Saved model directory: {CONFIG['model_dir']}")



def load_ser_model(save_dir: str, device: torch.device) -> Tuple[Wav2Vec2ForSER, Wav2Vec2FeatureExtractor, Dict[str, Any]]:

    with open(os.path.join(save_dir, "ser_config.json"), "r", encoding="utf-8") as f:

        cfg = json.load(f)



    loaded_model = Wav2Vec2ForSER(

        num_labels=int(cfg["num_labels"]),

        backbone_name=str(cfg["backbone_name"]),

        dropout=0.25,

    )

    state = torch.load(os.path.join(save_dir, "pytorch_model.bin"), map_location=device)

    loaded_model.load_state_dict(state, strict=True)

    loaded_model.to(device)

    loaded_model.eval()



    loaded_extractor = Wav2Vec2FeatureExtractor.from_pretrained(save_dir)

    return loaded_model, loaded_extractor, cfg



def predict_emotion(audio_path: str) -> Dict[str, float]:

    try:

        if not os.path.isfile(audio_path):

            return {"label": "unknown", "confidence": 0.0}



        if not hasattr(predict_emotion, "_cache"):

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            loaded_model, loaded_extractor, cfg = load_ser_model(CONFIG["model_dir"], device)

            setattr(

                predict_emotion,

                "_cache",

                {"model": loaded_model, "extractor": loaded_extractor, "cfg": cfg, "device": device},

            )



        cache = getattr(predict_emotion, "_cache")

        model_local: Wav2Vec2ForSER = cache["model"]

        extractor_local: Wav2Vec2FeatureExtractor = cache["extractor"]

        cfg_local: Dict[str, Any] = cache["cfg"]

        device_local: torch.device = cache["device"]



        audio = load_audio_fixed(audio_path)

        if audio is None:

            return {"label": "unknown", "confidence": 0.0}



        features = extractor_local(

            audio,

            sampling_rate=int(cfg_local["sampling_rate"]),

            max_length=int(cfg_local["sampling_rate"] * cfg_local["max_duration"]),

            truncation=True,

            padding="max_length",

            return_attention_mask=True,

            return_tensors="pt",

        )

        model_inputs = {k: v.to(device_local) for k, v in features.items()}



        with torch.no_grad():

            logits = model_local(**model_inputs)

            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]



        pred_id = int(np.argmax(probs))

        id_to_lbl = cfg_local.get("id_to_label", {})

        pred_label = id_to_lbl.get(str(pred_id), "unknown")

        confidence = float(probs[pred_id])



        return {"label": str(pred_label), "confidence": confidence}

    except (FileNotFoundError, OSError):

        return {"label": "unknown", "confidence": 0.0}

    except Exception:

        return {"label": "unknown", "confidence": 0.0}



zip_base = CONFIG["model_dir"]

zip_path = shutil.make_archive(zip_base, "zip", CONFIG["model_dir"])

print(f"Model zip created: {zip_path}")



try:

    from google.colab import files as colab_files

    colab_files.download(zip_path)

except Exception as e:

    print(f"Auto-download unavailable: {e}")


