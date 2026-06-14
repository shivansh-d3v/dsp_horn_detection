#!/usr/bin/env python3
"""
5-Fold Cross-Validation Dataset Creator for Binary Horn Classifier
==================================================================
Creates stratified 5-fold splits with augmentation on horn training files.

Requirements: librosa, soundfile, numpy
"""

import os
import sys
import re
import random
import shutil
import time

import librosa
import soundfile as sf
import numpy as np


# ──────────────────────────────────────────────────────────────────────
# HARDCODED PATHS (relative to this script's location)
# ──────────────────────────────────────────────────────────────────────
HORN_SRC   = os.path.join(".", "dataset", "horn", "sliced_horns")
NOISE_SRC  = os.path.join(".", "dataset", "background", "sliced_noise")
OUTPUT_ROOT = os.path.join(".", "folds")

# ──────────────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────────────
SEED = 42
NUM_FOLDS = 5
HORN_TOTAL = 3_050
NOISE_TOTAL = 39_615
HORN_PER_BUCKET = 610       # 3050 / 5
NOISE_PER_BUCKET = 7_923    # 39615 / 5
TRAIN_HORN_REAL = 2_440     # 4 * 610
TRAIN_HORN_AUG = 4_880      # 2 * 2440
TRAIN_HORN_TOTAL = 7_320    # 2440 + 4880
TRAIN_NOISE_TOTAL = 31_692  # 4 * 7923
TEST_HORN = 610
TEST_NOISE = 7_923
SR = 16_000
PROGRESS_EVERY = 500


def resolve_and_print_paths():
    """Resolve all relative paths to absolute and print them for verification."""
    global HORN_SRC, NOISE_SRC, OUTPUT_ROOT
    HORN_SRC = os.path.abspath(HORN_SRC)
    NOISE_SRC = os.path.abspath(NOISE_SRC)
    OUTPUT_ROOT = os.path.abspath(OUTPUT_ROOT)
    print("=" * 60)
    print("  RESOLVED ABSOLUTE PATHS")
    print("=" * 60)
    print(f"  Horn source  : {HORN_SRC}")
    print(f"  Noise source : {NOISE_SRC}")
    print(f"  Output root  : {OUTPUT_ROOT}")
    print("=" * 60)
    print()


def collect_audio_files(directory):
    """Return sorted list of absolute paths to audio files in directory."""
    files = []
    for f in os.listdir(directory):
        if f.lower().endswith(('.wav', '.flac', '.ogg', '.mp3')):
            files.append(os.path.join(directory, f))
    files.sort()
    return files


def split_into_buckets(file_list, bucket_size, name):
    """Split a list into exactly NUM_FOLDS buckets of bucket_size each."""
    if len(file_list) != bucket_size * NUM_FOLDS:
        raise RuntimeError(
            f"[ERROR] {name}: expected {bucket_size * NUM_FOLDS} files, "
            f"found {len(file_list)}. Cannot proceed."
        )
    buckets = []
    for i in range(NUM_FOLDS):
        start = i * bucket_size
        end = start + bucket_size
        buckets.append(file_list[start:end])
    return buckets


def verify_buckets(horn_buckets, noise_buckets):
    """Print bucket counts and raise if any are wrong."""
    print("Bucket verification:")
    all_ok = True
    for i in range(NUM_FOLDS):
        h = len(horn_buckets[i])
        n = len(noise_buckets[i])
        h_ok = "OK" if h == HORN_PER_BUCKET else "FAIL"
        n_ok = "OK" if n == NOISE_PER_BUCKET else "FAIL"
        print(f"  Bucket {i+1}: horn={h} [{h_ok}]  noise={n} [{n_ok}]")
        if h != HORN_PER_BUCKET or n != NOISE_PER_BUCKET:
            all_ok = False
    if not all_ok:
        raise RuntimeError("[ERROR] Bucket counts do not match. Aborting.")
    print()


def create_fold_dirs(fold_num):
    """Create the fold_N directory structure and return the four paths."""
    fold_dir = os.path.join(OUTPUT_ROOT, f"fold_{fold_num}")
    train_horn_dir  = os.path.join(fold_dir, "train", "horn")
    train_noise_dir = os.path.join(fold_dir, "train", "noise")
    test_horn_dir   = os.path.join(fold_dir, "test", "horn")
    test_noise_dir  = os.path.join(fold_dir, "test", "noise")
    for d in [train_horn_dir, train_noise_dir, test_horn_dir, test_noise_dir]:
        os.makedirs(d, exist_ok=True)
    return train_horn_dir, train_noise_dir, test_horn_dir, test_noise_dir


def copy_files(file_list, dest_dir, label=""):
    """Copy files using shutil.copy2. Never move or modify sources."""
    for src in file_list:
        fname = os.path.basename(src)
        dst = os.path.join(dest_dir, fname)
        shutil.copy2(src, dst)


def augment_horn_training(train_horn_files, dest_dir, fold_num):
    """
    Generate exactly 2 augmented copies for each real horn training file.
    aug1 = pitch shift (random between -2.0 and +2.0 semitones)
    aug2 = time stretch (random rate between 0.90 and 1.10)
    """
    total_aug = len(train_horn_files) * 2
    aug_count = 0

    for index, src_path in enumerate(train_horn_files):
        # Reproducible but varied augmentations
        random.seed(index)
        np.random.seed(index)

        # Load at exactly 16 kHz
        y, _ = librosa.load(src_path, sr=SR)

        basename = os.path.splitext(os.path.basename(src_path))[0]

        # ── aug1: pitch shift ──
        semitones = random.uniform(-2.0, 2.0)
        y_pitch = librosa.effects.pitch_shift(y, sr=SR, n_steps=semitones)
        aug1_name = f"{basename}_aug1.wav"
        sf.write(os.path.join(dest_dir, aug1_name), y_pitch, SR)
        aug_count += 1

        # ── aug2: time stretch ──
        rate = random.uniform(0.90, 1.10)
        y_stretch = librosa.effects.time_stretch(y, rate=rate)
        aug2_name = f"{basename}_aug2.wav"
        sf.write(os.path.join(dest_dir, aug2_name), y_stretch, SR)
        aug_count += 1

        # Progress reporting
        if aug_count % PROGRESS_EVERY == 0 or aug_count == total_aug:
            print(f"  [Fold {fold_num}] Augmenting... {aug_count}/{total_aug} done")


def count_files(directory):
    """Count files (not directories) in a directory."""
    return len([f for f in os.listdir(directory)
                if os.path.isfile(os.path.join(directory, f))])


def strip_aug_suffix(filename):
    """Remove _aug1 or _aug2 suffix to get the base filename."""
    name = os.path.splitext(filename)[0]
    name = re.sub(r'_aug[12]$', '', name)
    return name


def verify_fold(fold_num, train_horn_dir, train_noise_dir,
                test_horn_dir, test_noise_dir):
    """
    Count files and check for data leakage.
    Raise immediately on any violation.
    """
    th = count_files(train_horn_dir)
    tn = count_files(train_noise_dir)
    teh = count_files(test_horn_dir)
    ten = count_files(test_noise_dir)

    print(f"\n  [Fold {fold_num}] Verification:")
    print(f"    train/horn/  = {th:,}  (expected {TRAIN_HORN_TOTAL:,})")
    print(f"    train/noise/ = {tn:,}  (expected {TRAIN_NOISE_TOTAL:,})")
    print(f"    test/horn/   = {teh:,}  (expected {TEST_HORN:,})")
    print(f"    test/noise/  = {ten:,}  (expected {TEST_NOISE:,})")

    errors = []
    if th != TRAIN_HORN_TOTAL:
        errors.append(f"train/horn count {th} != {TRAIN_HORN_TOTAL}")
    if tn != TRAIN_NOISE_TOTAL:
        errors.append(f"train/noise count {tn} != {TRAIN_NOISE_TOTAL}")
    if teh != TEST_HORN:
        errors.append(f"test/horn count {teh} != {TEST_HORN}")
    if ten != TEST_NOISE:
        errors.append(f"test/noise count {ten} != {TEST_NOISE}")

    if errors:
        for e in errors:
            print(f"    [FAIL] {e}")
        raise RuntimeError(
            f"[ERROR] Fold {fold_num} file counts do not match. Aborting."
        )
    print(f"    [PASS] All counts correct.")

    # ── Data leakage check ──
    train_base_names = set()
    for f in os.listdir(train_horn_dir):
        if os.path.isfile(os.path.join(train_horn_dir, f)):
            train_base_names.add(strip_aug_suffix(f))

    test_base_names = set()
    for f in os.listdir(test_horn_dir):
        if os.path.isfile(os.path.join(test_horn_dir, f)):
            test_base_names.add(strip_aug_suffix(f))

    overlap = train_base_names & test_base_names
    if overlap:
        overlap_list = sorted(overlap)[:20]  # show at most 20
        print(f"    [FAIL] DATA LEAKAGE DETECTED! {len(overlap)} overlapping base names:")
        for name in overlap_list:
            print(f"      - {name}")
        raise RuntimeError(
            f"[ERROR] Fold {fold_num}: data leakage found "
            f"({len(overlap)} overlapping files). Aborting."
        )
    print(f"    [PASS] No data leakage detected.")
    print()


def print_final_summary():
    """Print the exact final summary block."""
    print("=" * 60)
    print("  ALL 5 FOLDS CREATED AND VERIFIED SUCCESSFULLY")
    print("=" * 60)
    print("  train/horn  per fold : 7,320  (2,440 real + 4,880 augmented)")
    print("  train/noise per fold : 31,692 (all real, no augmentation)")
    print("  test/horn   per fold : 610    (all real, never augmented)")
    print("  test/noise  per fold : 7,923  (all real, never augmented)")
    print("  pos_weight for training loss  : 4.33  (31692 / 7320)")
    print("  Decision threshold at inference: 0.62")
    print("=" * 60)


def main():
    t0 = time.time()

    # ── Step 0: Resolve and display paths ──
    resolve_and_print_paths()

    # Validate source directories exist
    if not os.path.isdir(HORN_SRC):
        raise RuntimeError(f"[ERROR] Horn source not found: {HORN_SRC}")
    if not os.path.isdir(NOISE_SRC):
        raise RuntimeError(f"[ERROR] Noise source not found: {NOISE_SRC}")

    # ── Step 1: Collect, shuffle, bucket ──
    print("Collecting audio files...")
    horn_files = collect_audio_files(HORN_SRC)
    noise_files = collect_audio_files(NOISE_SRC)
    print(f"  Found {len(horn_files)} horn files")
    print(f"  Found {len(noise_files)} noise files")
    print()

    # Single shuffle with fixed seed — never reshuffle after this
    random.seed(SEED)
    random.shuffle(horn_files)

    random.seed(SEED)
    random.shuffle(noise_files)

    # Split into 5 equal buckets
    horn_buckets = split_into_buckets(horn_files, HORN_PER_BUCKET, "horn")
    noise_buckets = split_into_buckets(noise_files, NOISE_PER_BUCKET, "noise")
    verify_buckets(horn_buckets, noise_buckets)

    # ── Step 2: Create each fold ──
    # Fold rotation table (hardcoded):
    # Fold 1: test=Bucket1, train=2+3+4+5
    # Fold 2: test=Bucket2, train=1+3+4+5
    # Fold 3: test=Bucket3, train=1+2+4+5
    # Fold 4: test=Bucket4, train=1+2+3+5
    # Fold 5: test=Bucket5, train=1+2+3+4

    for fold in range(1, NUM_FOLDS + 1):
        fold_start = time.time()
        print(f"{'='*60}")
        print(f"  CREATING FOLD {fold}")
        print(f"{'='*60}")

        test_bucket_idx = fold - 1  # 0-indexed
        train_bucket_indices = [i for i in range(NUM_FOLDS) if i != test_bucket_idx]

        # Create directory structure
        train_horn_dir, train_noise_dir, test_horn_dir, test_noise_dir = \
            create_fold_dirs(fold)

        # ── Test: copy bucket_N files (NEVER augment, NEVER modify) ──
        print(f"  Copying test horn files (Bucket {fold})...")
        copy_files(horn_buckets[test_bucket_idx], test_horn_dir,
                   label=f"test horn fold {fold}")

        print(f"  Copying test noise files (Bucket {fold})...")
        copy_files(noise_buckets[test_bucket_idx], test_noise_dir,
                   label=f"test noise fold {fold}")

        # ── Train horn: copy 4 remaining buckets ──
        train_horn_files = []
        for idx in train_bucket_indices:
            train_horn_files.extend(horn_buckets[idx])
        print(f"  Copying {len(train_horn_files)} train horn files "
              f"(Buckets {[i+1 for i in train_bucket_indices]})...")
        copy_files(train_horn_files, train_horn_dir,
                   label=f"train horn fold {fold}")

        # ── Train horn: augment (2 copies per real file) ──
        print(f"  Augmenting train horn files ({TRAIN_HORN_AUG} augmented files)...")
        augment_horn_training(train_horn_files, train_horn_dir, fold)

        # ── Train noise: copy 4 remaining buckets (NO augmentation) ──
        train_noise_files = []
        for idx in train_bucket_indices:
            train_noise_files.extend(noise_buckets[idx])
        print(f"  Copying {len(train_noise_files)} train noise files "
              f"(Buckets {[i+1 for i in train_bucket_indices]})...")
        copy_files(train_noise_files, train_noise_dir,
                   label=f"train noise fold {fold}")

        # ── Step 3: Verify this fold ──
        verify_fold(fold, train_horn_dir, train_noise_dir,
                    test_horn_dir, test_noise_dir)

        elapsed = time.time() - fold_start
        print(f"  Fold {fold} completed in {elapsed:.1f}s\n")

    # ── Step 4: Final summary ──
    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s\n")
    print_final_summary()


if __name__ == "__main__":
    main()
