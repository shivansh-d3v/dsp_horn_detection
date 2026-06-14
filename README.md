---
license: mit
tags:
  - audio-classification
  - horn-detection
  - mcu
  - esp32-s3
  - pytorch
  - cnn
  - direction-of-arrival
  - gcc-phat
  - smart-helmet
datasets:
  - custom
metrics:
  - accuracy
  - f1
  - recall
  - precision
pipeline_tag: audio-classification
---

# 🎺 Horn Detector — MCU-Optimised CNN for Smart Helmet

A lightweight binary audio classifier that detects vehicle horn sounds in real-time, designed for deployment on microcontroller units (ESP32-S3). The system also includes a GCC-PHAT based Direction-of-Arrival (DOA) estimator to localise the horn source direction using dual microphones.

## Key Features

- **Ultra-compact model**: ~326,785 parameters — fits in ESP32-S3 PSRAM
- **High recall**: 98.1% horn recall (mean across 5-fold CV) — critical for safety
- **Fast inference**: Single `(1, 64, 64)` log-Mel spectrogram → sigmoid → threshold
- **DOA estimation**: GCC-PHAT direction localisation with 76.7% bin accuracy at 10 dB SNR

---

## Model Architecture

```
Input:   (1, 64, 64) log-Mel spectrogram
         ↓
Conv2d(1→16, k=5, p=2) → ReLU → MaxPool2d(2)       → (16, 32, 32)
Conv2d(16→32, k=5, p=2) → ReLU → MaxPool2d(2)       → (32, 16, 16)
Conv2d(32→64, k=5, p=2) → ReLU → MaxPool2d(2)       → (64, 8, 8)
         ↓
Flatten(4096) → Dropout(0.5) → FC(64) → ReLU → FC(1)
         ↓
Output:  raw logit → sigmoid → threshold (0.62)
```

**MCU constraints preserved:**
- All operations are MCU-friendly: Conv2d, ReLU, MaxPool2d, Linear
- Dropout is disabled at inference via `model.eval()`
- FC hidden layer = 64 to minimise SRAM usage

---

## Audio Preprocessing

| Parameter       | Value           |
|-----------------|-----------------|
| Sample rate     | 16,000 Hz       |
| Duration        | 633 ms (10,128 samples) |
| N_FFT           | 400 (25 ms window) |
| Hop length      | 160 (10 ms hop)  |
| Mel bands       | 64              |
| Log transform   | `log(mel + 0.01)` |
| Output shape    | `(1, 64, 64)` float32 |

---

## Quick Start — Inference

```bash
pip install torch librosa numpy
```

**Test on a single WAV file:**
```bash
python inference.py path/to/your_audio.wav
```

**From Python:**
```python
from model import HornDetectorCNN
from inference import preprocess_audio
import torch

model = HornDetectorCNN()
model.load_state_dict(torch.load("best_horn_detector_mcu.pth", map_location="cpu"))
model.eval()

input_tensor = preprocess_audio("your_audio.wav")
with torch.no_grad():
    logit = model(input_tensor)
    prob = torch.sigmoid(logit).item()

print(f"Horn probability: {prob:.2%}")
print(f"Prediction: {'HORN' if prob >= 0.62 else 'NOISE'}")
```

---

## 5-Fold Cross-Validation Results

Trained with `pos_weight = 4.33` to correct class imbalance (31,692 noise / 7,320 horn per fold).

| Fold | Accuracy | Horn Recall | Horn Precision | Horn F1 | Best Epoch |
|------|----------|-------------|----------------|---------|------------|
| 1    | 0.9782   | 0.9918      | 0.7697         | 0.8668  | 50         |
| 2    | 0.9711   | 0.9525      | 0.7272         | 0.8247  | 50         |
| 3    | 0.9755   | 0.9885      | 0.7491         | 0.8523  | 46         |
| 4    | 0.9742   | 0.9820      | 0.7413         | 0.8449  | 44         |
| 5    | 0.9822   | 0.9902      | 0.8053         | 0.8882  | 45         |
| **Mean** | **0.9762 ± 0.0038** | **0.9810 ± 0.0146** | **0.7585 ± 0.0271** | **0.8554 ± 0.0213** | — |

**Best fold**: Fold 5 (Horn F1 = 0.8882)

### Best Fold (Fold 5) Confusion Matrix

```
                  Predicted Horn   Predicted Noise
Actual Horn            604              6
Actual Noise           146            7,777
```

- **Horn Recall**: 99.02% — only 6 horns missed out of 610
- **Horn Precision**: 80.53%
- **Overall Accuracy**: 98.22%

---

## Direction-of-Arrival (DOA) Estimation

Uses GCC-PHAT (Generalised Cross-Correlation with Phase Transform) to estimate the direction of the horn source from dual microphones on the helmet.

**Hardware Parameters:**
- Microphone separation: 21 cm
- Sampling frequency: 16,000 Hz
- Angle range: −90° to +90° (left to right)

### Angular RMSE vs SNR

| SNR (dB) | Mean Angular RMSE |
|----------|-------------------|
| −5       | 47.30°            |
| 0        | 30.69°            |
| 5        | 18.50°            |
| 10       | 15.04°            |
| 15       | 12.24°            |
| 20       | 10.03°            |

**Direction bin accuracy at SNR = 10 dB: 76.7%** (5 bins: Far Left, Left, Center, Right, Far Right)

---

## Dataset

- **Total horn clips**: 3,050 (633 ms each, 16 kHz)
- **Total noise clips**: 39,615 (urban ambient sounds — traffic, chatter, wind, etc.)
- **Augmentation**: Each fold's training horn set is augmented 2× via pitch shifting (±2 semitones) and time stretching (0.9–1.1×), reaching 7,320 horn samples per fold
- **Split**: Stratified 5-fold CV with data leakage checks

---

## Training

**Prerequisites:**
```bash
pip install -r requirements.txt
```

**Full pipeline:**
```bash
# Step 1: Create 5-fold splits from raw dataset
python create_kfold.py

# Step 2: Precompute spectrograms (converts .wav → .pt tensors)
python precompute_spectrograms.py

# Step 3: Train with 5-fold cross-validation
python train_kfold_cv.py

# Step 4: (Optional) Run DOA simulation for the research paper
python doa_simulation.py
```

### Training Configuration

| Parameter       | Value               |
|-----------------|---------------------|
| Epochs          | 50                  |
| Batch size      | 512                 |
| Learning rate   | 1e-3 (Adam)         |
| Weight decay    | 1e-4                |
| Scheduler       | ReduceLROnPlateau (factor=0.5, patience=5) |
| pos_weight      | 4.33                |
| Threshold       | 0.62                |
| Mixed precision | AMP on CUDA         |
| Initialisation  | Kaiming (He)        |

---

## Repository Structure

```
├── README.md                        # This file
├── requirements.txt                 # Python dependencies
├── best_horn_detector_mcu.pth       # Trained model weights (~1.3 MB)
├── model.py                         # HornDetectorCNN architecture definition
├── inference.py                     # Single-file inference script
├── train_kfold_cv.py                # 5-fold CV training (loads .pt tensors)
├── precompute_spectrograms.py       # .wav → .pt spectrogram conversion
├── create_kfold.py                  # Dataset splitting into 5 folds
├── augment_horns.py                 # Horn data augmentation pipeline
├── evaluate_mcu.py                  # Batch evaluation script
├── plot_confusion_matrix.py         # Confusion matrix visualisation
├── doa_simulation.py                # GCC-PHAT DOA Monte Carlo simulation
└── results/
    ├── cv_summary.json              # Aggregate 5-fold metrics
    ├── fold5_confusion_matrix.png   # Best fold confusion matrix plot
    ├── fold_1_report.txt            # Per-fold classification reports
    ├── fold_2_report.txt
    ├── fold_3_report.txt
    ├── fold_4_report.txt
    ├── fold_5_report.txt
    ├── doa_rmse_vs_snr_a.png        # DOA RMSE vs SNR plot
    └── doa_rmse_vs_snr_b.png        # DOA RMSE per angle plot
```

---

## Hardware Target

| Component              | Specification                    |
|------------------------|----------------------------------|
| MCU                    | ESP32-S3                         |
| Model storage          | PSRAM (~1.3 MB weights)          |
| Inference input        | (1, 64, 64) log-Mel spectrogram  |
| Audio capture          | I2S MEMS microphone, 16 kHz      |
| DOA                    | Dual MEMS mics, 21 cm separation |
| Application            | Smart helmet for road safety     |

---

## License

MIT
