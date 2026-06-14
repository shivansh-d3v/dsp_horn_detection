import os
import sys
import numpy as np
import librosa
import torch
from model import HornDetectorCNN

# Audio / spectrogram parameters (must match training/precomputation exactly)
SAMPLE_RATE    = 16_000
DURATION_MS    = 633
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION_MS / 1000)  # 10,128 samples
N_FFT          = 400
HOP_LENGTH     = 160
N_MELS         = 64
LOG_OFFSET     = 0.01

def preprocess_audio(wav_path):
    """
    Load a .wav file and preprocess it into a log-Mel spectrogram tensor.
    """
    if not os.path.exists(wav_path):
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    # Load and resample
    y, _ = librosa.load(wav_path, sr=SAMPLE_RATE)

    # Pad or truncate to fixed length
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode="constant")
    else:
        y = y[:TARGET_SAMPLES]

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE,
        n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
    )

    # Log transform
    log_mel = np.log(mel + LOG_OFFSET)

    # (1, 1, 64, 64) batch-sized float32 tensor
    tensor = torch.from_numpy(log_mel).unsqueeze(0).unsqueeze(0).float()
    return tensor

def run_inference(wav_path, weights_path="best_horn_detector_mcu.pth", threshold=0.62):
    """
    Run inference on a single audio file.
    """
    if not os.path.exists(weights_path):
        # Fall back to checking results/best_horn_detector_mcu.pth
        fallback_path = os.path.join("results", "best_horn_detector_mcu.pth")
        if os.path.exists(fallback_path):
            weights_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Weights file not found at: {weights_path} (or fallback '{fallback_path}')\n"
                f"Please run training or download the model weights first."
            )

    # Load spectrogram tensor
    input_tensor = preprocess_audio(wav_path)

    # Initialize model and load weights
    model = HornDetectorCNN()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    # Predict
    with torch.no_grad():
        logit = model(input_tensor)
        probability = torch.sigmoid(logit).item()
    
    is_horn = probability >= threshold
    label = "Horn" if is_horn else "Noise"

    print("-" * 45)
    print(f"Results for {os.path.basename(wav_path)}:")
    print(f"  Probability : {probability:.4%}")
    print(f"  Threshold   : {threshold:.4%}")
    print(f"  Prediction  : {label.upper()}")
    print("-" * 45)

    return is_horn, probability

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_audio_file.wav> [path_to_weights.pth]")
        sys.exit(1)

    audio_file = sys.argv[1]
    weights_file = sys.argv[2] if len(sys.argv) > 2 else "best_horn_detector_mcu.pth"

    run_inference(audio_file, weights_file)
