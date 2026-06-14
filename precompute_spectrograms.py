"""
precompute_spectrograms.py
==========================
Converts every .wav file in the 5-fold dataset into a precomputed
PyTorch tensor (.pt) file, eliminating on-the-fly audio processing
during training and dramatically speeding up GPU utilisation.

Each .pt file contains a (1, 64, 64) float32 log-Mel spectrogram tensor,
computed with identical parameters to the original training pipeline:
    - Sample rate   : 16,000 Hz
    - Duration      : 633 ms  → 10,128 samples
    - N_FFT         : 400  (25 ms window)
    - Hop length    : 160  (10 ms hop)
    - Mel bands     : 64
    - Log transform : log(mel + 0.01)

Original .wav files are NEVER deleted or modified.
Already-existing .pt files are skipped (safe to re-run).

Usage:
    python precompute_spectrograms.py
"""

import os
import glob
import time
import traceback

import numpy as np
import librosa
import torch

# ──────────────────────────────────────────────────────────
# CONFIGURATION — edit FOLDS_ROOT if your path differs
# ──────────────────────────────────────────────────────────

FOLDS_ROOT = os.path.join(
    "C:\\", "Users", "idk22", "Downloads", "Music", "horn", "folds"
)

NUM_FOLDS      = 5
SAMPLE_RATE    = 16_000
DURATION_MS    = 633
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION_MS / 1000)  # 10,128
N_FFT          = 400
HOP_LENGTH     = 160
N_MELS         = 64
LOG_OFFSET     = 0.01   # log(mel + 0.01)
PROGRESS_EVERY = 1_000  # print progress every N files


# ──────────────────────────────────────────────────────────
# SPECTROGRAM COMPUTATION
# ──────────────────────────────────────────────────────────

def wav_to_tensor(wav_path):
    """
    Load a .wav file and return a (1, 64, 64) log-Mel tensor.
    Raises on any error — caller handles exceptions.
    """
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

    # (1, 64, 64) float32 tensor
    tensor = torch.from_numpy(log_mel).unsqueeze(0).float()
    return tensor


# ──────────────────────────────────────────────────────────
# DIRECTORY PROCESSOR
# ──────────────────────────────────────────────────────────

def process_directory(directory, global_counts):
    """
    Process all .wav files in a single directory.
    Updates global_counts dict in-place.
    """
    wav_files = sorted(glob.glob(os.path.join(directory, "*.wav")))
    n_wav     = len(wav_files)

    if n_wav == 0:
        print(f"  [WARNING] No .wav files found in:\n    {directory}")
        return

    print(f"\n  Processing: ...{os.sep.join(directory.split(os.sep)[-3:])}")
    print(f"  Found {n_wav:,} .wav files")

    created  = 0
    skipped  = 0
    failed   = 0
    t0       = time.time()

    for i, wav_path in enumerate(wav_files, start=1):
        pt_path = os.path.splitext(wav_path)[0] + ".pt"

        # Skip if already precomputed
        if os.path.exists(pt_path):
            skipped += 1
            global_counts["skipped"] += 1
            global_counts["total_wav"] += 1
            continue

        try:
            tensor = wav_to_tensor(wav_path)
            torch.save(tensor, pt_path)
            created += 1
            global_counts["created"] += 1

        except Exception as e:
            failed += 1
            global_counts["failed"] += 1
            print(f"\n  [FAILED] {os.path.basename(wav_path)}")
            print(f"    Error: {e}")

        global_counts["total_wav"] += 1

        # Progress update
        processed = created + skipped + failed
        if processed % PROGRESS_EVERY == 0:
            elapsed = time.time() - t0
            rate    = processed / elapsed if elapsed > 0 else 0
            remaining = (n_wav - processed) / rate if rate > 0 else 0
            print(f"    [{processed:>6,} / {n_wav:,}]  "
                  f"created={created:,}  skipped={skipped:,}  "
                  f"failed={failed:,}  "
                  f"~{remaining:.0f}s remaining")

    # Verify counts match
    pt_files = glob.glob(os.path.join(directory, "*.pt"))
    n_pt     = len(pt_files)
    elapsed  = time.time() - t0

    status = "✅" if (n_pt >= n_wav - failed) else "⚠️  MISMATCH"
    print(f"  Done in {elapsed:.1f}s — "
          f".wav={n_wav:,}  .pt={n_pt:,}  "
          f"created={created:,}  skipped={skipped:,}  "
          f"failed={failed:,}  {status}")

    if n_pt < n_wav - failed:
        print(f"  [WARNING] Expected {n_wav - failed:,} .pt files "
              f"but found {n_pt:,} — check for errors above.")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    t0_global = time.time()

    # Resolve and verify root path
    folds_root = os.path.abspath(FOLDS_ROOT)
    print("=" * 60)
    print("  PRECOMPUTE SPECTROGRAMS")
    print("=" * 60)
    print(f"  Folds root : {folds_root}")
    print(f"  Folds      : {NUM_FOLDS}")
    print(f"  Audio params:")
    print(f"    Sample rate    : {SAMPLE_RATE:,} Hz")
    print(f"    Duration       : {DURATION_MS} ms → {TARGET_SAMPLES:,} samples")
    print(f"    N_FFT          : {N_FFT}")
    print(f"    Hop length     : {HOP_LENGTH}")
    print(f"    Mel bands      : {N_MELS}")
    print(f"    Log offset     : {LOG_OFFSET}")
    print(f"    Output shape   : (1, 64, 64) float32")
    print(f"  Original .wav files will NOT be deleted.")
    print("=" * 60)

    if not os.path.isdir(folds_root):
        raise RuntimeError(
            f"[ERROR] Folds root not found:\n  {folds_root}\n"
            f"Check the FOLDS_ROOT path in this script."
        )

    # Global counters
    global_counts = {
        "total_wav": 0,
        "created":   0,
        "skipped":   0,
        "failed":    0,
    }

    # Process in order: fold_1..5, within each: train/horn, train/noise,
    # test/horn, test/noise
    subdirs = [
        os.path.join("train", "horn"),
        os.path.join("train", "noise"),
        os.path.join("test",  "horn"),
        os.path.join("test",  "noise"),
    ]

    for fold in range(1, NUM_FOLDS + 1):
        print(f"\n{'─' * 60}")
        print(f"  FOLD {fold} / {NUM_FOLDS}")
        print(f"{'─' * 60}")

        for subdir in subdirs:
            directory = os.path.join(folds_root, f"fold_{fold}", subdir)

            if not os.path.isdir(directory):
                print(f"  [ERROR] Directory missing: {directory}")
                print(f"  Skipping — check your fold structure.")
                continue

            process_directory(directory, global_counts)

    # ── Final summary ──────────────────────────────────────
    total_time = time.time() - t0_global

    # Estimate storage used by .pt files
    # Each tensor: 1 × 64 × 64 × 4 bytes = 16,384 bytes
    bytes_per_pt  = 1 * 64 * 64 * 4
    total_pt      = global_counts["created"] + global_counts["skipped"]
    storage_gb    = (total_pt * bytes_per_pt) / 1e9

    print(f"\n{'=' * 60}")
    print(f"  PRECOMPUTE COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Total .wav files found    : {global_counts['total_wav']:,}")
    print(f"  Total .pt  files created  : {global_counts['created']:,}")
    print(f"  Total .pt  files skipped  : {global_counts['skipped']:,}  "
          f"(already existed)")
    print(f"  Failed conversions         : {global_counts['failed']:,}")
    print(f"  Estimated .pt storage used : {storage_gb:.2f} GB")
    print(f"  Total time                 : {total_time:.1f}s "
          f"({total_time/60:.1f} min)")

    if global_counts["failed"] > 0:
        print(f"\n  [WARNING] {global_counts['failed']:,} files failed. "
              f"Check errors above and re-run — "
              f"the script will skip already-completed files.")
    else:
        print(f"\n  All files converted successfully.")
        print(f"  You can now run train_kfold_cv.py")

    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
