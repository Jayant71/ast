# AST Mixer Dataset — Usage Guide

## Overview

This project fine-tunes the **Audio Spectrogram Transformer (AST)** on a custom **mixer sound dataset** with 6 classes:

| Index | Label               | Description                   |
|-------|---------------------|-------------------------------|
| 0     | `elec+mech`         | Electrical + Mechanical       |
| 1     | `elec+therm`        | Electrical + Thermal          |
| 2     | `electrical`        | Electrical only               |
| 3     | `mechanical-bearing`| Mechanical (bearing)          |
| 4     | `mechanical-bush`   | Mechanical (bush)             |
| 5     | `thermal`           | Thermal only                  |

Audio samples are recorded at 3 distances (5cm, 15cm, 30cm), 5 speed levels (1–5), and 3 intensities (low, medium, high) by two subjects (preethi, sujata).

---

## Prerequisites

```bash
cd /path/to/ast
python3 -m venv venvast
source venvast/bin/activate
pip install -r requirements.txt
```

Required packages (from `requirements.txt`): `torch==1.8.1`, `torchaudio==0.8.1`, `timm==0.4.5`, `numpy`, `scikit-learn`, `wget`.

> **Note:** `timm` must be exactly version `0.4.5` — the AST model code is not compatible with newer versions.

---

## Project Structure

```
ast/
├── src/                          # Core AST source code
│   ├── run.py                    # Training entry point
│   ├── traintest.py              # Training/validation loop
│   ├── dataloader.py             # Dataset loader
│   ├── models/ast_models.py      # AST model definition
│   └── get_norm_stats.py         # Compute normalization stats
├── egs/mixer/                    # Mixer dataset recipe
│   ├── run_mixer.sh              # Main training script
│   ├── prep_mixer.py             # Data preparation & audio preprocessing
│   ├── get_mixer_result.py       # Result summary
│   ├── analyze_audio.py          # Audio dataset analysis tool
│   ├── analyze_audio.sh          # Shell-based audio analysis
│   └── data/
│       ├── mixer_train_data.json           # Raw training metadata
│       ├── mixer_eval_data.json            # Raw eval metadata
│       ├── class_labels_indices.csv        # Label index mapping
│       ├── mixer_class_labels_indices.csv  # Copy of above
│       ├── mixer_train_data_processed.json # Processed training metadata
│       ├── mixer_eval_data_processed.json  # Processed eval metadata
│       └── processed_audio/                # Preprocessed .wav files
│           ├── elec+mech/
│           ├── elec+therm/
│           ├── electrical/
│           ├── mechanical-bearing/
│           ├── mechanical-bush/
│           └── thermal/
├── pretrained_models/            # Downloaded pretrained weights
└── requirements.txt
```

---

## Step-by-Step Usage

### Step 1: Prepare Your Dataset

Place your raw audio files organized by class label, then create two JSON metadata files:

**`mixer_train_data.json`** (training set):
```json
{
  "data": [
    {"wav": "/absolute/path/to/audio_file1.wav", "labels": "electrical"},
    {"wav": "/absolute/path/to/audio_file2.wav", "labels": "thermal"}
  ]
}
```

**`mixer_eval_data.json`** (evaluation set):
```json
{
  "data": [
    {"wav": "/absolute/path/to/audio_file3.wav", "labels": "mechanical-bearing"}
  ]
}
```

**`class_labels_indices.csv`** (label mapping):
```csv
index,mid,display_name
0,elec+mech,elec+mech
1,elec+therm,elec+therm
2,electrical,electrical
3,mechanical-bearing,mechanical-bearing
4,mechanical-bush,mechanical-bush
5,thermal,thermal
```

> The `mid` column values must match the `labels` field in your JSON files. The `index` determines the class index used by the model.

Place all three files in `egs/mixer/data/`.

### Step 2: Run Data Preparation

```bash
cd egs/mixer
python prep_mixer.py
```

This script will:
1. Copy source JSON/CSV files to `data/`
2. Preprocess each audio file:
   - Convert to mono
   - Trim silence (top_db=40)
   - Normalize amplitude (target=0.1)
   - Resample to 16kHz
   - Pad/truncate to 30 seconds (480,000 samples)
3. Save processed audio in `data/processed_audio/<label>/`
4. Generate `mixer_train_data_processed.json` and `mixer_eval_data_processed.json` with updated file paths

### Step 3: (Optional) Compute Dataset Normalization Stats

The default uses AudioSet stats (`mean=-4.27`, `std=4.57`). For better accuracy, compute your own:

```python
import sys
sys.path.insert(0, '../../src')
import torch
import numpy as np
import dataloader

audio_conf = {
    'num_mel_bins': 128, 'target_length': 3000, 'freqm': 0, 'timem': 0,
    'mixup': 0, 'skip_norm': True, 'mode': 'train', 'dataset': 'mixer',
    'mean': -4.27, 'std': 4.57, 'noise': False
}

train_loader = torch.utils.data.DataLoader(
    dataloader.AudiosetDataset(
        './data/mixer_train_data_processed.json',
        label_csv='./data/mixer_class_labels_indices.csv',
        audio_conf=audio_conf
    ), batch_size=48, shuffle=False, num_workers=4, pin_memory=True
)

mean_vals, std_vals = [], []
for audio_input, labels in train_loader:
    mean_vals.append(torch.mean(audio_input).item())
    std_vals.append(torch.std(audio_input).item())

print(f"Mean: {np.mean(mean_vals):.7f}")
print(f"Std:  {np.mean(std_vals):.7f}")
```

Update `dataset_mean` and `dataset_std` in `run_mixer.sh` with the computed values.

### Step 4: (Optional) Analyze Your Dataset

```bash
python analyze_audio.py ./data/mixer_train_data_processed.json
```

This reports per-class duration, RMS energy, and silence ratio statistics.

### Step 5: Train the Model

```bash
cd egs/mixer
./run_mixer.sh
```

Or with SLURM:
```bash
sbatch run_mixer.sh
```

### Step 6: View Results

```bash
python get_mixer_result.py --exp_path ./exp/test-mixer-f10-t10-impTrue-aspFalse-b24-lr0.0001
```

Output includes per-epoch: Accuracy, AUC, Precision, Recall, d-prime, Train Loss, Val Loss, Cumulative Accuracy, Cumulative AUC, and Learning Rate.

The best model is saved at `<exp_dir>/models/best_audio_model.pth`.

---

## Key Hyperparameters (in `run_mixer.sh`)

| Parameter           | Default | Description                                     |
|---------------------|---------|-------------------------------------------------|
| `lr`                | `1e-4`  | Learning rate (use `1e-5` with AudioSet pretrain)|
| `batch_size`        | `24`    | Batch size (reduce if OOM with 30s audio)       |
| `epoch`             | `50`    | Max training epochs                              |
| `fstride`/`tstride` | `10`    | Patch split stride (10 = overlap of 6)           |
| `freqm`             | `24`    | SpecAug frequency mask width                     |
| `timem`             | `282`   | SpecAug time mask width (~20% of 3000 frames)    |
| `audio_length`      | `3000`  | Spectrogram time frames (30s × 100Hz hop)        |
| `loss`              | `CE`    | CrossEntropy (single-label classification)        |
| `metrics`           | `acc`   | Primary evaluation metric                         |
| `dataset_mean`      | `-4.27` | Spectrogram normalization mean                    |
| `dataset_std`       | `4.57`  | Spectrogram normalization std                     |
| `lrscheduler_start` | `5`     | Epoch to start LR decay                           |
| `lrscheduler_decay` | `0.85`  | LR decay factor per step                          |

---

## Inference on New Audio

```python
import torch
import torchaudio
import sys
sys.path.insert(0, '../../src')
from models import ASTModel

LABELS = ['elec+mech', 'elec+therm', 'electrical', 'mechanical-bearing', 'mechanical-bush', 'thermal']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ASTModel(
    label_dim=6, fstride=10, tstride=10,
    input_fdim=128, input_tdim=3000,
    imagenet_pretrain=False, audioset_pretrain=False,
    model_size='base384', verbose=False
)
model = torch.nn.DataParallel(model)
sd = torch.load('./exp/<your_exp_dir>/models/best_audio_model.pth', map_location=device)
model.load_state_dict(sd)
model = model.to(device)
model.eval()

waveform, sr = torchaudio.load('test_audio.wav')
waveform = waveform - waveform.mean()
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    waveform = resampler(waveform)
    sr = 16000

fbank = torchaudio.compliance.kaldi.fbank(
    waveform, htk_compat=True, sample_frequency=sr,
    use_energy=False, window_type='hanning',
    num_mel_bins=128, dither=0.0, frame_shift=10
)

target_length = 3000
n_frames = fbank.shape[0]
p = target_length - n_frames
if p > 0:
    fbank = torch.nn.ZeroPad2d((0, 0, 0, p))(fbank)
elif p < 0:
    fbank = fbank[:target_length, :]

fbank = (fbank - (-4.27)) / (4.57 * 2)
fbank = fbank.unsqueeze(0).to(device)

with torch.no_grad():
    output = model(fbank)
    probs = torch.sigmoid(output)
    pred_idx = torch.argmax(probs, dim=1).item()
    print(f"Predicted: {LABELS[pred_idx]} (confidence: {probs[0, pred_idx]:.4f})")
```

---

## Validation Report

### Issues Found and Fixed

| # | Severity | Issue | File | Status |
|---|----------|-------|------|--------|
| 1 | **Critical** | `get_mixer_result.py` expected 8 columns but `result.csv` has 10 (accuracy, AUC, precision, recall, d_prime, train_loss, val_loss, cum_acc, cum_auc, lr) | `get_mixer_result.py` | Fixed |
| 2 | **Critical** | `prep_mixer.py` had hardcoded relative paths (`../../datafiles/`) that don't exist. Source data lookup now checks `data/` first, then falls back to `../../datafiles/` | `prep_mixer.py` | Fixed |
| 3 | **Critical** | `run_mixer.sh` had hardcoded venv path and didn't `cd` to script directory, causing path resolution failures | `run_mixer.sh` | Fixed |
| 4 | **High** | Processed JSON paths were relative to project root, not `egs/mixer/`. Re-running `prep_mixer.py` will regenerate with correct paths | `prep_mixer.py` | Fixed |

### Known Upstream Issues (Not Modified)

| # | Severity | Issue | Details |
|---|----------|-------|---------|
| 1 | **High** | CE loss validation bug in `traintest.py:291` | `torch.sigmoid()` is applied to model output before computing `CrossEntropyLoss`, which already applies `log_softmax` internally. This double-activation makes validation loss values incorrect (though the argmax prediction is unaffected). To fix, compute loss on raw logits before sigmoid. |
| 2 | **Medium** | Old PyTorch/torchaudio versions | `torch==1.8.1` and `torchaudio==0.8.1` are from 2021. The `torch.cuda.amp.autocast` and `GradScaler` imports are deprecated in newer PyTorch. |
| 3 | **Medium** | `timm==0.4.5` pinned strictly | Model code asserts this exact version (`ast_models.py:50`). |

### Recommendations

1. **Small dataset warning:** Your training set has ~600 samples across 6 classes (~100/class). This is very small for fine-tuning a transformer. Expect overfitting. Mitigation strategies:
   - Use `audioset_pretrain=True` and `lr=1e-5` for better transfer learning
   - Increase `mixup` to `0.5` for data augmentation
   - Reduce `epoch` to 25–30 and monitor val loss for early stopping
   - Consider data augmentation: time stretching, pitch shifting, background noise

2. **Compute your own normalization stats** instead of using AudioSet defaults. See Step 3 above.

3. **GPU memory:** With `audio_length=3000` (30s audio) and `batch_size=24`, you need ~16GB+ GPU memory. Reduce `batch_size` to 12 or 8 if you get OOM errors.

4. **Re-run `prep_mixer.py`** after the fixes to regenerate processed JSONs with correct absolute paths.

---

## Quick Start (One Command)

```bash
cd egs/mixer
python prep_mixer.py && ./run_mixer.sh
```
