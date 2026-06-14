"""
augment_horns.py
================
Balances the horn vs. noise classes across a 5-fold cross-validation
dataset by generating synthetic horn samples via random augmentation.

Target: Each Fold_k/Train/horns/ directory is filled to exactly 31,692
files (matching the noise count) by augmenting the original 2,440 horns.

Augmentation paths (randomly chosen per sample):
  1. Pitch shift only
  2. Time stretch only
  3. Add background noise only
  4. Pitch + Time
  5. Pitch + Noise
  6. Time + Noise
  7. Pitch + Time + Noise

CRITICAL: Test/ directories are NEVER touched.
"""

import os
import uuid
import random

import numpy as np
import librosa
import soundfile as sf

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

BASE_DIR       = "kfold_dataset"
NUM_FOLDS      = 5
TARGET_COUNT   = 31_692       # match noise file count per fold
ORIGINAL_HORNS = 2_440        # original horn files per fold train set
SAMPLE_RATE    = 22_050       # target sample rate for all processing

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ──────────────────────────────────────────────────────────
# AUGMENTATION PRIMITIVES
# ──────────────────────────────────────────────────────────

def pitch_shift(y, sr):
    """Shift pitch by a random amount between -2 and +2 semitones."""
    n_steps = random.uniform(-2.0, 2.0)
    return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)


def time_stretch(y):
    """Time-stretch by a random rate between 0.9× and 1.1×."""
    rate = random.uniform(0.9, 1.1)
    return librosa.effects.time_stretch(y, rate=rate)


def add_noise(y, noise_dir, noise_files, sr, snr_db=None):
    """
    Mix a random noise file into the signal at a random SNR (10–20 dB).
    The noise is trimmed or looped to match the signal length.
    """
    if snr_db is None:
        snr_db = random.uniform(10.0, 20.0)

    # Load a random noise file
    noise_file = random.choice(noise_files)
    noise_path = os.path.join(noise_dir, noise_file)
    n, _ = librosa.load(noise_path, sr=sr)

    # Match lengths: loop if noise is shorter, trim if longer
    if len(n) < len(y):
        repeats = (len(y) // len(n)) + 1
        n = np.tile(n, repeats)
    n = n[:len(y)]

    # Scale noise to achieve the desired SNR
    signal_power = np.mean(y ** 2)
    noise_power  = np.mean(n ** 2)

    if noise_power == 0:
        return y  # silent noise file — skip mixing

    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / noise_power)

    return y + scale * n


# ──────────────────────────────────────────────────────────
# AUGMENTATION DISPATCHER
# ──────────────────────────────────────────────────────────

def augment(y, sr, noise_dir, noise_files):
    """
    Randomly select one of 7 augmentation paths and apply it.
    Returns the augmented waveform.
    """
    path = random.randint(1, 7)

    if path == 1:
        # Pitch only
        y = pitch_shift(y, sr)

    elif path == 2:
        # Time stretch only
        y = time_stretch(y)

    elif path == 3:
        # Background noise only
        y = add_noise(y, noise_dir, noise_files, sr)

    elif path == 4:
        # Pitch + Time
        y = pitch_shift(y, sr)
        y = time_stretch(y)

    elif path == 5:
        # Pitch + Noise
        y = pitch_shift(y, sr)
        y = add_noise(y, noise_dir, noise_files, sr)

    elif path == 6:
        # Time + Noise
        y = time_stretch(y)
        y = add_noise(y, noise_dir, noise_files, sr)

    elif path == 7:
        # Pitch + Time + Noise
        y = pitch_shift(y, sr)
        y = time_stretch(y)
        y = add_noise(y, noise_dir, noise_files, sr)

    return y


# ──────────────────────────────────────────────────────────
# MAIN LOOP — Process each fold
# ──────────────────────────────────────────────────────────

def main():
    for fold in range(1, NUM_FOLDS + 1):
        horn_dir  = os.path.join(BASE_DIR, f"Fold_{fold}", "Train", "horns")
        noise_dir = os.path.join(BASE_DIR, f"Fold_{fold}", "Train", "noise")

        # Collect the ORIGINAL horn files (snapshot before augmentation)
        original_horn_files = sorted([
            f for f in os.listdir(horn_dir) if f.lower().endswith(".wav")
        ])
        assert len(original_horn_files) == ORIGINAL_HORNS, (
            f"Fold_{fold}: Expected {ORIGINAL_HORNS} original horns, "
            f"found {len(original_horn_files)}"
        )

        # Collect noise filenames for the noise-mixing augmentation
        noise_files = sorted([
            f for f in os.listdir(noise_dir) if f.lower().endswith(".wav")
        ])
        assert len(noise_files) == TARGET_COUNT, (
            f"Fold_{fold}: Expected {TARGET_COUNT} noise files, "
            f"found {len(noise_files)}"
        )

        # Current count (including any previously generated augments)
        current_count = len([
            f for f in os.listdir(horn_dir) if f.lower().endswith(".wav")
        ])

        print(f"\n{'='*55}")
        print(f"  Fold_{fold}  |  Current horns: {current_count}  |  "
              f"Target: {TARGET_COUNT}")
        print(f"{'='*55}")

        generated = 0
        while current_count < TARGET_COUNT:
            # Pick a random original horn file
            src_name = random.choice(original_horn_files)
            src_path = os.path.join(horn_dir, src_name)

            # Load the source horn clip
            y, sr = librosa.load(src_path, sr=SAMPLE_RATE)

            # Apply a random augmentation path
            y_aug = augment(y, sr, noise_dir, noise_files)

            # Save with a unique filename
            unique_id = uuid.uuid4().hex[:12]
            out_name  = f"aug_horn_{unique_id}.wav"
            out_path  = os.path.join(horn_dir, out_name)
            sf.write(out_path, y_aug, sr)

            current_count += 1
            generated += 1

            # Progress reporting every 1,000 files
            if generated % 1000 == 0:
                print(f"  [Fold_{fold}] Generated {generated} / "
                      f"{TARGET_COUNT - ORIGINAL_HORNS} augmented samples...")

        print(f"  [✓] Fold_{fold} complete — "
              f"{generated} augmented horns generated.  "
              f"Total: {current_count}")

    print(f"\n{'='*55}")
    print("  Augmentation finished across all folds!")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
