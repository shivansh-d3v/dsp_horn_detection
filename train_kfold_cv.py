"""
train_kfold_cv.py
=================
5-Fold Cross-Validation training for binary horn detection.
Designed for MCU deployment — uses the compact MCU-optimised architecture.

REQUIRES: Run precompute_spectrograms.py first to generate .pt tensor files.
          Training loads precomputed (1, 64, 64) log-Mel tensors directly —
          no on-the-fly audio processing, maximising GPU utilisation.

Architecture (MCU-ready):
    Input:   (1, 64, 64) log-Mel spectrogram
    Conv2d(1→16,  k=5, p=2) → ReLU → MaxPool2d(2)
    Conv2d(16→32, k=5, p=2) → ReLU → MaxPool2d(2)
    Conv2d(32→64, k=5, p=2) → ReLU → MaxPool2d(2)
    Flatten(4096) → Dropout(0.5) → FC(64) → ReLU → FC(1)

MCU constraints preserved:
    - ~326,785 params (stored in PSRAM during inference on ESP32-S3)
    - Dropout(0.5) for regularisation (disabled at inference via model.eval())
    - FC hidden=64 to minimise SRAM usage on MCU
    - Fixed input: 633 ms @ 16 kHz → 64-band log-Mel → (1, 64, 64)
    - Single logit output → sigmoid → threshold

Training specifics:
    - pos_weight = 4.33  (31,692 noise / 7,320 horn) for class imbalance
    - Decision threshold = 0.62 at inference (tuned on validation set)
    - Mixed-precision (AMP) on CUDA, fallback to float32 on CPU
    - Best checkpoint per fold saved by lowest test loss
    - ReduceLROnPlateau scheduler (factor=0.5, patience=5)
    - Kaiming weight initialisation
    - Classification report + confusion matrix per fold
    - Final 5-fold CV summary with mean ± std
    - All results saved to ./results/
"""

import os
import sys
import glob
import time
import json
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

from sklearn.metrics import classification_report, confusion_matrix

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

FOLDS_ROOT     = os.path.join(".", "folds")
NUM_FOLDS      = 5
NUM_EPOCHS     = 50
BATCH_SIZE     = 512          # RTX 4050 6GB — safe with precomputed tensors
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 1e-4         # mild L2 regularisation
NUM_WORKERS    = 4            # 0 = safest on Windows; .pt loading is fast
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR    = "results"

# Imbalance correction: 31,692 noise / 7,320 horn = 4.33
POS_WEIGHT_VAL = 4.33

# Tuned decision threshold at inference
THRESHOLD = 0.62

# Device selection
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"


# ──────────────────────────────────────────────────────────
# 1. DATASET — loads precomputed .pt tensors
# ──────────────────────────────────────────────────────────

class AudioSpectrogramDataset(Dataset):
    """
    Loads precomputed log-Mel spectrogram tensors (.pt files).
    Each .pt file contains a (1, 64, 64) float32 tensor precomputed
    from 633 ms @ 16 kHz audio using:
        N_FFT=400, HOP_LENGTH=160, N_MELS=64
        log_mel = log(mel + 0.01)

    horn/*.pt  -> label 1  (positive class — horn present)
    noise/*.pt -> label 0  (negative class — ambient noise only)

    REQUIRES: precompute_spectrograms.py must be run before training.
    """

    def __init__(self, root_dir):
        self.samples = []   # list of (filepath, label)

        horn_dir  = os.path.join(root_dir, "horn")
        noise_dir = os.path.join(root_dir, "noise")

        for f in sorted(glob.glob(os.path.join(horn_dir, "*.pt"))):
            self.samples.append((f, 1))
        for f in sorted(glob.glob(os.path.join(noise_dir, "*.pt"))):
            self.samples.append((f, 0))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"\n[ERROR] No .pt files found in: {root_dir}\n"
                f"Run precompute_spectrograms.py first to generate them.\n"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        # Load precomputed tensor — no audio processing needed
        spec_tensor  = torch.load(filepath, weights_only=True)  # (1, 64, 64)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        return spec_tensor, label_tensor


# ──────────────────────────────────────────────────────────
# 2. MODEL — MCU-optimised CNN
# ──────────────────────────────────────────────────────────

class HornDetectorCNN(nn.Module):
    """
    MCU-ready 3-block CNN for binary horn detection.

    Dimension walkthrough:
        Input:   (B,  1, 64, 64)
        Block 1: (B, 16, 32, 32)   Conv2d(1→16,  k=5, p=2) → ReLU → MaxPool2d(2)
        Block 2: (B, 32, 16, 16)   Conv2d(16→32, k=5, p=2) → ReLU → MaxPool2d(2)
        Block 3: (B, 64,  8,  8)   Conv2d(32→64, k=5, p=2) → ReLU → MaxPool2d(2)
        Flatten: (B, 4096)
        FC:      (B, 64)           Dropout(0.5) → Linear(4096→64) → ReLU
        Output:  (B, 1)            Linear(64→1) — raw logit, sigmoid at inference

    MCU notes:
        - ~326,785 parameters; stored in PSRAM during inference on ESP32-S3
        - Dropout(0.5) disabled automatically at inference via model.eval()
        - All ops MCU-friendly: Conv2d, ReLU, MaxPool2d, Linear
    """

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(1,  16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),           # raw logit
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming / He initialisation for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="relu"
                )
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # (B, 4096)
        x = self.classifier(x)
        return x                      # (B, 1) raw logit


# ──────────────────────────────────────────────────────────
# 3. TRAINING & EVALUATION HELPERS
# ──────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    """One training epoch with mixed-precision on GPU."""
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for specs, labels in loader:
        specs  = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)  # (B, 1)

        optimizer.zero_grad(set_to_none=True)

        with autocast("cuda", enabled=USE_AMP):
            logits = model(specs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * specs.size(0)
        preds    = (torch.sigmoid(logits) >= THRESHOLD).float()
        correct += (preds == labels).sum().item()
        total   += specs.size(0)

    return running_loss / total, correct / total


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    """Evaluate model — no gradients."""
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    for specs, labels in loader:
        specs  = specs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True).unsqueeze(1)

        with autocast("cuda", enabled=USE_AMP):
            logits = model(specs)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * specs.size(0)
        preds    = (torch.sigmoid(logits) >= THRESHOLD).float()
        correct += (preds == labels).sum().item()
        total   += specs.size(0)

    return running_loss / total, correct / total


@torch.inference_mode()
def full_evaluation(model, loader, device):
    """
    Full evaluation returning all predictions, labels, and probabilities
    for classification report and confusion matrix.
    """
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for specs, labels in loader:
        specs = specs.to(device, non_blocking=True)

        with autocast("cuda", enabled=USE_AMP):
            logits = model(specs)

        probs = torch.sigmoid(logits).cpu().view(-1).numpy()
        preds = (probs >= THRESHOLD).astype(int)

        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().astype(int).tolist())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ──────────────────────────────────────────────────────────
# 4. MAIN — 5-FOLD CROSS-VALIDATION LOOP
# ──────────────────────────────────────────────────────────

def main():
    t0_global = time.time()

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR,    exist_ok=True)

    # ── Hardware + config summary ──────────────────────────
    print("=" * 62)
    print("  TRAINING CONFIGURATION")
    print("=" * 62)
    if DEVICE.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU         : {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"  Mixed-prec  : Enabled (AMP)")
    else:
        print(f"  Device      : CPU")
        print(f"  Mixed-prec  : Disabled")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  pos_weight  : {POS_WEIGHT_VAL}")
    print(f"  Threshold   : {THRESHOLD}")
    print(f"  Epochs      : {NUM_EPOCHS}")
    print(f"  Batch size  : {BATCH_SIZE}")
    print(f"  LR          : {LEARNING_RATE}")
    print(f"  Weight dec  : {WEIGHT_DECAY}")
    print(f"  Model       : HornDetectorCNN "
          f"(~326,785 params, PSRAM inference on ESP32-S3)")
    print(f"  Dataset     : Precomputed .pt tensors (1, 64, 64)")
    print("=" * 62)
    print()

    fold_results = []

    for fold in range(1, NUM_FOLDS + 1):
        t0_fold = time.time()

        print(f"\n{'=' * 62}")
        print(f"  FOLD {fold} / {NUM_FOLDS}")
        print(f"{'=' * 62}")

        # ── Paths ──────────────────────────────────────────
        train_dir = os.path.join(FOLDS_ROOT, f"fold_{fold}", "train")
        test_dir  = os.path.join(FOLDS_ROOT, f"fold_{fold}", "test")

        # Verify all 4 subdirectories exist
        for split, parent in [("train", train_dir), ("test", test_dir)]:
            for cls in ["horn", "noise"]:
                p = os.path.join(parent, cls)
                if not os.path.isdir(p):
                    raise RuntimeError(f"[ERROR] Missing directory: {p}")

        # ── Datasets & loaders ─────────────────────────────
        train_dataset = AudioSpectrogramDataset(train_dir)
        test_dataset  = AudioSpectrogramDataset(test_dir)

        train_loader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
            persistent_workers=(NUM_WORKERS > 0),
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(DEVICE.type == "cuda"),
            persistent_workers=(NUM_WORKERS > 0),
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
        )

        # train_loader = DataLoader(
        #     train_dataset,
        #     batch_size=BATCH_SIZE,
        #     shuffle=True,
        #     num_workers=NUM_WORKERS,
        #     pin_memory=(DEVICE.type == "cuda"),
        # )
        # test_loader = DataLoader(
        #     test_dataset,
        #     batch_size=BATCH_SIZE,
        #     shuffle=False,
        #     num_workers=NUM_WORKERS,
        #     pin_memory=(DEVICE.type == "cuda"),
        # )

        # Count files for display
        horn_train  = len(glob.glob(
            os.path.join(train_dir, "horn",  "*.pt")))
        noise_train = len(glob.glob(
            os.path.join(train_dir, "noise", "*.pt")))
        horn_test   = len(glob.glob(
            os.path.join(test_dir,  "horn",  "*.pt")))
        noise_test  = len(glob.glob(
            os.path.join(test_dir,  "noise", "*.pt")))

        print(f"  Train : {len(train_dataset):,} total "
              f"(horn={horn_train:,}, noise={noise_train:,})")
        print(f"  Test  : {len(test_dataset):,} total "
              f"(horn={horn_test:,}, noise={noise_test:,})")

        # ── Fresh model for each fold ──────────────────────
        model = HornDetectorCNN().to(DEVICE)

        # Print parameter count once
        if fold == 1:
            total_params = sum(p.numel() for p in model.parameters())
            print(f"  Params: {total_params:,}")

        # Imbalance correction
        pos_weight = torch.tensor([POS_WEIGHT_VAL]).to(DEVICE)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        scaler = GradScaler("cuda", enabled=USE_AMP)

        # Reduce LR on plateau — halve when test loss stops improving
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )

        best_test_loss = float("inf")
        best_test_acc  = 0.0
        best_epoch     = 0
        ckpt_path      = os.path.join(
            CHECKPOINT_DIR, f"best_fold_{fold}.pth"
        )

        # ── Header row ────────────────────────────────────
        print(f"\n  {'Epoch':>5}  │  "
              f"{'Train Loss':>10}  {'Train Acc':>9}  │  "
              f"{'Test Loss':>9}  {'Test Acc':>8}  │  "
              f"{'LR':>8}  {'':>6}")
        print(f"  {'─'*70}")

        # ── Epoch loop ────────────────────────────────────
        for epoch in range(1, NUM_EPOCHS + 1):
            ep_start = time.time()

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, scaler, DEVICE
            )
            test_loss, test_acc = evaluate(
                model, test_loader, criterion, DEVICE
            )

            scheduler.step(test_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            # Save checkpoint if best test loss
            marker = ""
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_test_acc  = test_acc
                best_epoch     = epoch
                torch.save({
                    "fold":                 fold,
                    "epoch":                epoch,
                    "model_state_dict":     model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_test_loss":       best_test_loss,
                    "best_test_acc":        best_test_acc,
                    "threshold":            THRESHOLD,
                    "pos_weight":           POS_WEIGHT_VAL,
                }, ckpt_path)
                marker = "★ saved"

            ep_time = time.time() - ep_start
            print(
                f"  {epoch:>5}  │  "
                f"{train_loss:>10.4f}  {train_acc:>9.4f}  │  "
                f"{test_loss:>9.4f}  {test_acc:>8.4f}  │  "
                f"{current_lr:>8.1e}  {ep_time:>4.1f}s  {marker}"
            )

        # ── Load best checkpoint & full evaluation ─────────
        print(f"\n  Loading best checkpoint (epoch {best_epoch}, "
              f"test loss {best_test_loss:.4f})...")

        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])

        labels, preds, probs = full_evaluation(model, test_loader, DEVICE)

        # Classification report
        report_str  = classification_report(
            labels, preds,
            target_names=["Noise", "Horn"],
            digits=4,
        )
        report_dict = classification_report(
            labels, preds,
            target_names=["Noise", "Horn"],
            digits=4,
            output_dict=True,
        )

        cm = confusion_matrix(labels, preds)

        print(f"\n  [Fold {fold}] Classification Report "
              f"(threshold = {THRESHOLD}):")
        print(report_str)
        print(f"  [Fold {fold}] Confusion Matrix:")
        print(f"                  Pred Noise   Pred Horn")
        print(f"  Actual Noise     {cm[0][0]:>7,}     {cm[0][1]:>7,}")
        print(f"  Actual Horn      {cm[1][0]:>7,}     {cm[1][1]:>7,}")

        fold_accuracy = report_dict["accuracy"]
        fold_time     = time.time() - t0_fold

        horn_recall    = report_dict["Horn"]["recall"]
        horn_precision = report_dict["Horn"]["precision"]
        horn_f1        = report_dict["Horn"]["f1-score"]

        print(f"\n  [✓] Fold {fold} best checkpoint  : "
              f"epoch {best_epoch}, loss {best_test_loss:.4f}")
        print(f"  [✓] Fold {fold} eval accuracy    : {fold_accuracy:.4f}")
        print(f"  [✓] Fold {fold} horn recall      : {horn_recall:.4f}")
        print(f"  [✓] Fold {fold} horn precision   : {horn_precision:.4f}")
        print(f"  [✓] Fold {fold} horn F1          : {horn_f1:.4f}")
        print(f"  [✓] Fold {fold} time             : {fold_time:.1f}s")

        # Store results
        fold_info = {
            "fold":               fold,
            "best_epoch":         best_epoch,
            "best_test_loss":     float(best_test_loss),
            "best_test_acc":      float(best_test_acc),
            "eval_accuracy":      float(fold_accuracy),
            "horn_recall":        float(horn_recall),
            "horn_precision":     float(horn_precision),
            "horn_f1":            float(horn_f1),
            "classification_report": report_dict,
            "confusion_matrix":   cm.tolist(),
            "checkpoint_path":    ckpt_path,
        }
        fold_results.append(fold_info)

        # Save per-fold text report
        report_path = os.path.join(RESULTS_DIR, f"fold_{fold}_report.txt")
        with open(report_path, "w") as fh:
            fh.write(f"Fold {fold}\n")
            fh.write(f"Best epoch  : {best_epoch}\n")
            fh.write(f"Threshold   : {THRESHOLD}\n")
            fh.write(f"pos_weight  : {POS_WEIGHT_VAL}\n\n")
            fh.write("Classification Report:\n")
            fh.write(report_str)
            fh.write("\nConfusion Matrix:\n")
            fh.write(f"                Pred Noise   Pred Horn\n")
            fh.write(f"Actual Noise     {cm[0][0]:>7,}     {cm[0][1]:>7,}\n")
            fh.write(f"Actual Horn      {cm[1][0]:>7,}     {cm[1][1]:>7,}\n")

    # ──────────────────────────────────────────────────────
    # 5. AGGREGATE 5-FOLD RESULTS
    # ──────────────────────────────────────────────────────

    accuracies  = [r["eval_accuracy"]  for r in fold_results]
    recalls     = [r["horn_recall"]    for r in fold_results]
    precisions  = [r["horn_precision"] for r in fold_results]
    f1s         = [r["horn_f1"]        for r in fold_results]

    mean_acc = np.mean(accuracies);  std_acc = np.std(accuracies)
    mean_rec = np.mean(recalls);     std_rec = np.std(recalls)
    mean_pre = np.mean(precisions);  std_pre = np.std(precisions)
    mean_f1  = np.mean(f1s);         std_f1  = np.std(f1s)

    best_fold_idx = int(np.argmax(f1s))   # best by horn F1, not accuracy
    best_fold     = fold_results[best_fold_idx]

    total_time = time.time() - t0_global

    print(f"\n{'=' * 62}")
    print(f"  5-FOLD CROSS-VALIDATION RESULTS")
    print(f"{'=' * 62}")
    print(f"  {'Fold':>4}  {'Accuracy':>9}  {'Horn Recall':>11}  "
          f"{'Horn Prec':>10}  {'Horn F1':>8}  {'Epoch':>5}")
    print(f"  {'─' * 55}")
    for r in fold_results:
        print(f"  {r['fold']:>4}  "
              f"{r['eval_accuracy']:>9.4f}  "
              f"{r['horn_recall']:>11.4f}  "
              f"{r['horn_precision']:>10.4f}  "
              f"{r['horn_f1']:>8.4f}  "
              f"{r['best_epoch']:>5}")
    print(f"  {'─' * 55}")
    print(f"  {'Mean':>4}  "
          f"{mean_acc:>9.4f}  "
          f"{mean_rec:>11.4f}  "
          f"{mean_pre:>10.4f}  "
          f"{mean_f1:>8.4f}")
    print(f"  {'±Std':>4}  "
          f"{std_acc:>9.4f}  "
          f"{std_rec:>11.4f}  "
          f"{std_pre:>10.4f}  "
          f"{std_f1:>8.4f}")
    print(f"\n  Best Fold (by Horn F1) : Fold {best_fold['fold']} "
          f"(F1={best_fold['horn_f1']:.4f})")
    print(f"  Total Training Time    : {total_time:.1f}s "
          f"({total_time/60:.1f} min)")
    print(f"{'=' * 62}")

    # ── Best fold detailed results ─────────────────────────
    print(f"\n{'=' * 62}")
    print(f"  BEST FOLD (Fold {best_fold['fold']}) — DETAILED RESULTS")
    print(f"{'=' * 62}")
    print(f"  Checkpoint : {best_fold['checkpoint_path']}")
    print(f"  Threshold  : {THRESHOLD}")
    print(f"  pos_weight : {POS_WEIGHT_VAL}")

    best_cm     = np.array(best_fold["confusion_matrix"])
    best_report = best_fold["classification_report"]

    print(f"\n  Classification Report:")
    print(f"  {'─' * 54}")
    header = (f"  {'':>12s}  {'precision':>10s}  {'recall':>10s}  "
              f"{'f1-score':>10s}  {'support':>9s}")
    print(header)
    for cls_name in ["Noise", "Horn"]:
        r = best_report[cls_name]
        print(f"  {cls_name:>12s}  {r['precision']:>10.4f}  "
              f"{r['recall']:>10.4f}  {r['f1-score']:>10.4f}  "
              f"{int(r['support']):>9,}")
    print(f"  {'─' * 54}")
    total_support = int(best_report["weighted avg"]["support"])
    print(f"  {'accuracy':>12s}  {'':>10s}  {'':>10s}  "
          f"{best_report['accuracy']:>10.4f}  "
          f"{total_support:>9,}")

    print(f"\n  Confusion Matrix:")
    print(f"                    Pred Noise   Pred Horn")
    print(f"  Actual Noise       {best_cm[0][0]:>7,}     {best_cm[0][1]:>7,}")
    print(f"  Actual Horn        {best_cm[1][0]:>7,}     {best_cm[1][1]:>7,}")
    print(f"\n  True  Negatives (noise correctly ignored) : "
          f"{best_cm[0][0]:>7,}")
    print(f"  False Positives (noise flagged as horn)   : "
          f"{best_cm[0][1]:>7,}")
    print(f"  False Negatives (horn missed — safety)    : "
          f"{best_cm[1][0]:>7,}")
    print(f"  True  Positives (horn correctly detected) : "
          f"{best_cm[1][1]:>7,}")
    print(f"{'=' * 62}")

    # ── Save aggregate JSON ────────────────────────────────
    summary = {
        "mean_accuracy":       float(mean_acc),
        "std_accuracy":        float(std_acc),
        "mean_horn_recall":    float(mean_rec),
        "std_horn_recall":     float(std_rec),
        "mean_horn_precision": float(mean_pre),
        "std_horn_precision":  float(std_pre),
        "mean_horn_f1":        float(mean_f1),
        "std_horn_f1":         float(std_f1),
        "best_fold":           best_fold["fold"],
        "best_fold_f1":        float(best_fold["horn_f1"]),
        "threshold":           THRESHOLD,
        "pos_weight":          POS_WEIGHT_VAL,
        "model":               "HornDetectorCNN (~326,785 params)",
        "batch_size":          BATCH_SIZE,
        "learning_rate":       LEARNING_RATE,
        "weight_decay":        WEIGHT_DECAY,
        "total_time_seconds":  total_time,
        "device":              str(DEVICE),
        "per_fold": [
            {
                "fold":          r["fold"],
                "accuracy":      r["eval_accuracy"],
                "horn_recall":   r["horn_recall"],
                "horn_precision":r["horn_precision"],
                "horn_f1":       r["horn_f1"],
                "best_epoch":    r["best_epoch"],
                "best_test_loss":r["best_test_loss"],
                "checkpoint":    r["checkpoint_path"],
            }
            for r in fold_results
        ],
    }

    summary_path = os.path.join(RESULTS_DIR, "cv_summary.json")
    with open(summary_path, "w") as fh:
        json.dump(summary, fh, indent=2)

    # Save best fold model weights as standalone file
    best_standalone = os.path.join(RESULTS_DIR, "best_horn_detector_mcu.pth")
    best_ckpt = torch.load(
        best_fold["checkpoint_path"], map_location="cpu", weights_only=True
    )
    torch.save(best_ckpt["model_state_dict"], best_standalone)

    print(f"\n  Summary JSON  : {summary_path}")
    print(f"  Per-fold TXTs : {RESULTS_DIR}/fold_N_report.txt")
    print(f"  Best model    : {best_standalone}")
    print()


if __name__ == "__main__":
    main()
