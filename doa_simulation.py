"""
doa_simulation.py
=================
Quantitative evaluation of the GCC-PHAT Direction-of-Arrival (DOA)
estimator for the smart helmet horn detection system.

Produces three deliverables for the research paper:
    1. Angular RMSE vs SNR table  (Table)
    2. Direction bin confusion matrix (Table)
    3. Angular error vs SNR line plot (Figure)

Simulation methodology:
    - True angle θ known exactly (ground truth)
    - Stereo signal generated with exact TDOA matching θ
    - Gaussian noise added at controlled SNR levels
    - GCC-PHAT run to estimate TDOA → converted to angle
    - Error = |estimated angle - true angle| in degrees
    - 200 Monte Carlo trials per (angle, SNR) combination

Hardware parameters (ESP32-S3 smart helmet):
    - Microphone separation : d = 0.21 m (conservative minimum)
    - Sampling frequency    : fs = 16,000 Hz
    - Speed of sound        : v = 343 m/s
    - Angle range           : -90° to +90° (left to right)

References:
    GCC-PHAT: Knapp & Carter, IEEE Trans. ASSP, 1976
    TDOA→angle: θ = arcsin(n·v / (fs·d))
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe on all platforms
from scipy.signal import fftconvolve

# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────

# Hardware parameters
D           = 0.21          # microphone separation (metres) — conservative
FS          = 16_000        # sampling frequency (Hz)
V           = 343.0         # speed of sound (m/s)
SIGNAL_LEN  = 4096          # samples per test signal (~256 ms at 16 kHz)
N_TRIALS    = 200           # Monte Carlo trials per (angle, SNR) pair

# Maximum TDOA in samples (physical limit)
MAX_TDOA_SAMPLES = int(np.ceil(D * FS / V))   # = ceil(0.21*16000/343) = 10

# Evaluation grid
TRUE_ANGLES_DEG = np.array([
    -90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90
])
SNR_LEVELS_DB = np.array([-5, 0, 5, 10, 15, 20])

# Direction bins for confusion matrix
# 5 bins covering -90° to +90°
DIRECTION_BINS = {
    "Far Left":  (-90, -54),
    "Left":      (-54, -18),
    "Center":    (-18,  18),
    "Right":     ( 18,  54),
    "Far Right": ( 54,  90),
}
BIN_NAMES  = list(DIRECTION_BINS.keys())
N_BINS     = len(BIN_NAMES)

# Output directory
OUT_DIR = "results"

# ──────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ──────────────────────────────────────────────────────────

def angle_to_tdoa_samples(angle_deg, d=D, fs=FS, v=V):
    """
    Convert a true angle to TDOA in fractional samples.
    θ → τ = d·sin(θ)/v  →  n = τ·fs
    Positive TDOA = sound arrives at right mic first (source on right).
    """
    theta_rad = np.radians(angle_deg)
    tau       = d * np.sin(theta_rad) / v      # seconds
    return tau * fs                             # fractional samples


def generate_signal(length, signal_type="horn"):
    """
    Generate a test signal resembling horn acoustics.
    Uses a combination of harmonics to mimic horn frequency content.
    """
    t = np.arange(length) / FS

    if signal_type == "horn":
        # Horn-like: 300 Hz fundamental + harmonics (600, 900, 1200 Hz)
        sig = (
            0.5  * np.sin(2 * np.pi * 300  * t) +
            0.3  * np.sin(2 * np.pi * 600  * t) +
            0.15 * np.sin(2 * np.pi * 900  * t) +
            0.05 * np.sin(2 * np.pi * 1200 * t)
        )
    else:
        # White noise fallback
        sig = np.random.randn(length)

    # Normalise to unit power
    sig = sig / (np.std(sig) + 1e-8)
    return sig


def apply_fractional_delay(signal, delay_samples):
    """
    Apply a fractional sample delay using sinc interpolation.
    Positive delay → signal arrives later (source on that side).
    """
    # Integer and fractional parts
    int_delay  = int(np.floor(delay_samples))
    frac_delay = delay_samples - int_delay

    # Sinc interpolation kernel (length 64, windowed)
    kernel_len = 64
    n          = np.arange(-kernel_len // 2, kernel_len // 2)
    kernel     = np.sinc(n - frac_delay)
    kernel    *= np.hanning(kernel_len)
    kernel    /= (np.sum(kernel) + 1e-8)

    # Convolve signal with kernel
    delayed = fftconvolve(signal, kernel, mode="full")
    # Shift slice to align and discard the kernel_len // 2 convolution latency
    delayed = delayed[kernel_len // 2 : kernel_len // 2 + len(signal)]

    # Apply integer delay via circular shift
    if int_delay != 0:
        delayed = np.roll(delayed, int_delay)
        if int_delay > 0:
            delayed[:int_delay] = 0
        else:
            delayed[int_delay:] = 0

    return delayed


def add_noise_at_snr(signal, snr_db):
    """
    Add Gaussian white noise to achieve the target SNR.
    SNR_dB = 10·log10(P_signal / P_noise)
    """
    sig_power   = np.mean(signal ** 2) + 1e-10
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise       = np.sqrt(noise_power) * np.random.randn(len(signal))
    return signal + noise


def gcc_phat(left, right, fs=FS, max_delay=None):
    """
    Generalised Cross-Correlation with Phase Transform (GCC-PHAT).
    Returns estimated TDOA in samples (positive = right arrives first).

    Args:
        left      : left microphone signal
        right     : right microphone signal
        fs        : sampling frequency
        max_delay : search window in samples (None = full range)

    Returns:
        tdoa_samples : estimated TDOA in samples
    """
    n      = len(left) + len(right) - 1
    n_fft  = int(2 ** np.ceil(np.log2(n)))   # next power of 2

    # FFTs
    L = np.fft.rfft(left,  n=n_fft)
    R = np.fft.rfft(right, n=n_fft)

    # Cross-power spectrum
    cross = L * np.conj(R)

    # PHAT weighting — normalise by magnitude
    denom = np.abs(cross) + 1e-10
    cross_phat = cross / denom

    # Inverse FFT → GCC-PHAT function
    gcc = np.fft.irfft(cross_phat, n=n_fft)

    # Rearrange so zero lag is at centre
    gcc = np.fft.fftshift(gcc[:n_fft])
    lags = np.arange(-(n_fft // 2), n_fft // 2)

    # Restrict search to physically meaningful delays
    if max_delay is not None:
        mask = np.abs(lags) <= max_delay
        gcc_search = np.where(mask, gcc, -np.inf)
    else:
        gcc_search = gcc

    # Peak = estimated TDOA
    tdoa_samples = lags[np.argmax(gcc_search)]
    return float(tdoa_samples)


def tdoa_to_angle(tdoa_samples, d=D, fs=FS, v=V):
    """
    Convert TDOA (samples) to angle in degrees.
    θ = arcsin(n·v / (fs·d))
    Clamps argument to [-1, 1] to handle numerical edge cases.
    """
    arg   = tdoa_samples * v / (fs * d)
    arg   = np.clip(arg, -1.0, 1.0)
    angle = np.degrees(np.arcsin(arg))
    return angle


def angle_to_bin(angle_deg):
    """Map an angle in degrees to its direction bin name."""
    for bin_name, (lo, hi) in DIRECTION_BINS.items():
        if lo <= angle_deg <= hi:
            return bin_name
    # Edge case: exactly ±90 — clamp to outermost bin
    if angle_deg < -90:
        return "Far Left"
    return "Far Right"


# ──────────────────────────────────────────────────────────
# SIMULATION RUNNERS
# ──────────────────────────────────────────────────────────

def run_rmse_simulation():
    """
    Run Monte Carlo simulation for Angular RMSE vs SNR.
    Returns rmse_table: shape (len(SNR_LEVELS_DB), len(TRUE_ANGLES_DEG))
    and mean_rmse_per_snr: shape (len(SNR_LEVELS_DB),)
    """
    print("\n" + "=" * 60)
    print("  SIMULATION 1 — Angular RMSE vs SNR")
    print("=" * 60)
    print(f"  Angles     : {TRUE_ANGLES_DEG}")
    print(f"  SNR levels : {SNR_LEVELS_DB} dB")
    print(f"  Trials     : {N_TRIALS} per (angle, SNR) pair")
    print(f"  Total runs : "
          f"{len(TRUE_ANGLES_DEG) * len(SNR_LEVELS_DB) * N_TRIALS:,}")

    rmse_table = np.zeros(
        (len(SNR_LEVELS_DB), len(TRUE_ANGLES_DEG))
    )
    mae_table = np.zeros_like(rmse_table)

    for s_idx, snr in enumerate(SNR_LEVELS_DB):
        print(f"\n  SNR = {snr:>3} dB", end="  ")

        for a_idx, true_angle in enumerate(TRUE_ANGLES_DEG):
            errors = []
            tdoa_true = angle_to_tdoa_samples(true_angle)

            for _ in range(N_TRIALS):
                # Generate source signal
                source = generate_signal(SIGNAL_LEN, signal_type="horn")

                # Create left and right channel with appropriate delay
                if tdoa_true >= 0:
                    # Source on right: right arrives first (negative delay on right)
                    left_sig  = apply_fractional_delay(source, tdoa_true)
                    right_sig = source.copy()
                else:
                    # Source on left: left arrives first
                    left_sig  = source.copy()
                    right_sig = apply_fractional_delay(source, -tdoa_true)

                # Add independent noise to each channel
                left_noisy  = add_noise_at_snr(left_sig,  snr)
                right_noisy = add_noise_at_snr(right_sig, snr)

                # Run GCC-PHAT
                tdoa_est = gcc_phat(
                    left_noisy, right_noisy,
                    max_delay=MAX_TDOA_SAMPLES + 2
                )

                # Convert to angle
                est_angle = tdoa_to_angle(tdoa_est)

                # Angular error
                error = abs(est_angle - true_angle)
                errors.append(error)

            errors     = np.array(errors)
            rmse       = np.sqrt(np.mean(errors ** 2))
            mae        = np.mean(errors)

            rmse_table[s_idx, a_idx] = rmse
            mae_table[s_idx, a_idx]  = mae
            print(".", end="", flush=True)

        print(f"  mean RMSE = "
              f"{np.mean(rmse_table[s_idx]):.2f}°")

    mean_rmse_per_snr = np.mean(rmse_table, axis=1)
    return rmse_table, mae_table, mean_rmse_per_snr


def run_confusion_simulation(snr_db=10):
    """
    Run direction bin confusion matrix simulation at a fixed SNR.
    Uses all true angles, maps both true and estimated to 5 direction bins.
    Returns confusion matrix (N_BINS x N_BINS).
    """
    print("\n" + "=" * 60)
    print(f"  SIMULATION 2 — Direction Confusion Matrix @ SNR={snr_db} dB")
    print("=" * 60)

    cm = np.zeros((N_BINS, N_BINS), dtype=int)

    # Use finer angle grid for confusion matrix
    fine_angles = np.linspace(-90, 90, 45)   # 45 test angles

    for true_angle in fine_angles:
        true_bin_name = angle_to_bin(true_angle)
        true_bin_idx  = BIN_NAMES.index(true_bin_name)
        tdoa_true     = angle_to_tdoa_samples(true_angle)

        for _ in range(N_TRIALS):
            source = generate_signal(SIGNAL_LEN, signal_type="horn")

            if tdoa_true >= 0:
                left_sig  = apply_fractional_delay(source, tdoa_true)
                right_sig = source.copy()
            else:
                left_sig  = source.copy()
                right_sig = apply_fractional_delay(source, -tdoa_true)

            left_noisy  = add_noise_at_snr(left_sig,  snr_db)
            right_noisy = add_noise_at_snr(right_sig, snr_db)

            tdoa_est  = gcc_phat(
                left_noisy, right_noisy,
                max_delay=MAX_TDOA_SAMPLES + 2
            )
            est_angle    = tdoa_to_angle(tdoa_est)
            est_bin_name = angle_to_bin(est_angle)
            est_bin_idx  = BIN_NAMES.index(est_bin_name)

            cm[true_bin_idx, est_bin_idx] += 1

        print(f"  {true_angle:>+6.1f}° -> bin '{true_bin_name}'  done")

    return cm


# ──────────────────────────────────────────────────────────
# OUTPUT — TABLES AND PLOT
# ──────────────────────────────────────────────────────────

def print_rmse_table(rmse_table, mean_rmse_per_snr):
    """Print Angular RMSE vs SNR table to console and file."""
    print("\n" + "=" * 60)
    print("  TABLE 1 — Angular RMSE (degrees) vs SNR")
    print(f"  Microphone separation: d = {D*100:.0f} cm")
    print(f"  Sampling rate: fs = {FS:,} Hz")
    print("=" * 60)

    # Header
    header = f"  {'SNR (dB)':>8}  {'Mean RMSE (°)':>14}  "
    header += "  ".join(f"{a:>+5}°" for a in TRUE_ANGLES_DEG)
    print(header)
    print("  " + "-" * (len(header) - 2))

    for s_idx, snr in enumerate(SNR_LEVELS_DB):
        row = f"  {snr:>8}  {mean_rmse_per_snr[s_idx]:>13.2f}°  "
        row += "  ".join(
            f"{rmse_table[s_idx, a_idx]:>5.2f}°"
            for a_idx in range(len(TRUE_ANGLES_DEG))
        )
        print(row)

    print("=" * 60)

    # Save to file
    out_path = os.path.join(OUT_DIR, "doa_rmse_table.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("Angular RMSE (degrees) vs SNR\n")
        f.write(f"Microphone separation : d = {D*100:.0f} cm\n")
        f.write(f"Sampling rate         : fs = {FS:,} Hz\n")
        f.write(f"Speed of sound        : v = {V} m/s\n")
        f.write(f"Trials per condition  : {N_TRIALS}\n\n")
        f.write(f"{'SNR (dB)':>8}  {'Mean RMSE':>10}  "
                f"{'Angles (degrees)':}\n")
        f.write("  " + "-" * 80 + "\n")
        for s_idx, snr in enumerate(SNR_LEVELS_DB):
            row = f"{snr:>8}  {mean_rmse_per_snr[s_idx]:>9.2f}°  "
            row += "  ".join(
                f"{rmse_table[s_idx, a_idx]:.2f}°"
                for a_idx in range(len(TRUE_ANGLES_DEG))
            )
            f.write(row + "\n")

    print(f"\n  Saved: {out_path}")


def print_confusion_matrix(cm, snr_db):
    """Print direction bin confusion matrix to console and file."""
    print("\n" + "=" * 60)
    print(f"  TABLE 2 — Direction Bin Confusion Matrix @ SNR={snr_db} dB")
    print(f"  Bins: {BIN_NAMES}")
    print("=" * 60)

    # Normalise rows to percentages
    row_sums = cm.sum(axis=1, keepdims=True).astype(float)
    cm_pct   = np.where(row_sums > 0, cm / row_sums * 100, 0)

    col_w = 12
    header = f"  {'True \\ Pred':>12}" + "".join(
        f"{name:>{col_w}}" for name in BIN_NAMES
    )
    print(header)
    print("  " + "-" * (len(header)))

    for i, true_name in enumerate(BIN_NAMES):
        row = f"  {true_name:>12}"
        for j in range(N_BINS):
            cell = f"{cm[i,j]}({cm_pct[i,j]:.0f}%)"
            row += f"{cell:>{col_w}}"
        print(row)

    print("=" * 60)

    # Overall bin accuracy
    correct = np.trace(cm)
    total   = cm.sum()
    bin_acc = correct / total * 100
    print(f"\n  Direction bin accuracy : {correct}/{total} = {bin_acc:.1f}%")

    # Save to file
    out_path = os.path.join(OUT_DIR, "doa_confusion_matrix.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Direction Bin Confusion Matrix @ SNR={snr_db} dB\n")
        f.write(f"Microphone separation : d = {D*100:.0f} cm\n")
        f.write(f"Trials per angle      : {N_TRIALS}\n\n")
        f.write(f"{'True \\ Pred':>12}")
        for name in BIN_NAMES:
            f.write(f"{name:>{col_w}}")
        f.write("\n" + "-" * (12 + col_w * N_BINS) + "\n")
        for i, true_name in enumerate(BIN_NAMES):
            f.write(f"{true_name:>12}")
            for j in range(N_BINS):
                cell = f"{cm[i,j]}({cm_pct[i,j]:.0f}%)"
                f.write(f"{cell:>{col_w}}")
            f.write("\n")
        f.write(f"\nDirection bin accuracy: {bin_acc:.1f}%\n")

    print(f"  Saved: {out_path}")
    return bin_acc


def set_publication_style():
    """Configure matplotlib for academic, publication-quality plots."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "text.usetex": False,  # Safe default on all platforms
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4,
        "ytick.major.size": 4,
        "xtick.minor.size": 2,
        "ytick.minor.size": 2,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.linewidth": 0.6,
        "grid.color": "#E2E8F0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#E2E8F0",
        "legend.fancybox": False,
    })


def plot_rmse_vs_snr_a(mean_rmse_per_snr):
    """
    FIGURE 4(a): Angular RMSE vs SNR
    Single column width (3.5" x 2.8") optimized for IEEE two-column paper.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Single deep-blue academic line with circular markers
    ax.plot(
        SNR_LEVELS_DB, mean_rmse_per_snr,
        color="#1E3A8A", marker="o", markersize=4.5, linewidth=1.2,
        label="Mean RMSE"
    )

    # Dashed threshold lines at 5° and 10°
    ax.axhline(y=5, color="#10B981", linestyle="--", linewidth=0.8, alpha=0.8, label="5° threshold")
    ax.axhline(y=10, color="#F59E0B", linestyle="--", linewidth=0.8, alpha=0.8, label="10° threshold")

    # Labels and Title
    ax.set_xlabel("SNR (dB)", fontsize=10)
    ax.set_ylabel("Angular RMSE (°)", fontsize=10)
    ax.set_title("Angular RMSE vs SNR", fontsize=10, fontweight="bold", pad=6)

    # Scale settings
    ax.set_xticks(SNR_LEVELS_DB)
    ax.set_xlim(-6, 21)
    ax.set_ylim(0, 55)

    # Tick and label typography
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.grid(True)

    # Legend - Medium size
    ax.legend(fontsize=8, loc="upper right")

    # Tight margin optimized for IEEE column alignment
    plt.tight_layout(pad=0.1)

    # Export paths
    out_png = os.path.join(OUT_DIR, "doa_rmse_vs_snr_a.png")
    out_pdf = os.path.join(OUT_DIR, "doa_rmse_vs_snr_a.pdf")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"  Saved Figure 4(a): {out_png} & {out_pdf}")
    return out_png


def plot_rmse_vs_snr_b(rmse_table):
    """
    FIGURE 4(b): RMSE Across True Angles
    Single column width (3.5" x 2.8") optimized for IEEE two-column paper.
    """
    set_publication_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    # Selected angles and indices corresponding to [-90, -60, -30, 0, +30, +60, +90]
    target_angles = [-90, -60, -30, 0, 30, 60, 90]
    indices = [0, 2, 4, 6, 8, 10, 12]

    # Distinguishable but subtle color palette
    colors = [
        "#991B1B",  # -90°: Deep Red
        "#EA580C",  # -60°: Rust Orange
        "#D97706",  # -30°: Amber/Gold
        "#059669",  # 0°: Emerald Green
        "#0284C7",  # +30°: Sky Blue
        "#4F46E5",  # +60°: Indigo
        "#6D28D9"   # +90°: Purple
    ]

    for i, (a_idx, angle) in enumerate(zip(indices, target_angles)):
        label = f"{angle:+}°" if angle != 0 else "0°"
        ax.plot(
            SNR_LEVELS_DB, rmse_table[:, a_idx],
            color=colors[i], marker="o", markersize=3.5, linewidth=1.0,
            label=label
        )

    # Labels and Title
    ax.set_xlabel("SNR (dB)", fontsize=10)
    ax.set_ylabel("Angular RMSE (°)", fontsize=10)
    ax.set_title("RMSE Across True Angles", fontsize=10, fontweight="bold", pad=6)

    # Scale settings
    ax.set_xticks(SNR_LEVELS_DB)
    ax.set_xlim(-6, 21)
    ax.set_ylim(0, 80)

    # Tick and label typography
    ax.tick_params(axis="both", which="major", labelsize=9)
    ax.grid(True)

    # Compact Legend inside plot to save space
    ax.legend(fontsize=7, loc="upper right", ncol=2, title="True Angle", title_fontsize=7)

    # Tight margin optimized for IEEE column alignment
    plt.tight_layout(pad=0.1)

    # Export paths
    out_png = os.path.join(OUT_DIR, "doa_rmse_vs_snr_b.png")
    out_pdf = os.path.join(OUT_DIR, "doa_rmse_vs_snr_b.pdf")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"  Saved Figure 4(b): {out_png} & {out_pdf}")
    return out_png



def print_final_summary(mean_rmse_per_snr, bin_acc):
    """Print paper-ready summary to console and file."""
    summary = []
    summary.append("=" * 60)
    summary.append("  DOA EVALUATION SUMMARY — FOR RESEARCH PAPER")
    summary.append("=" * 60)
    summary.append(f"  Hardware parameters:")
    summary.append(f"    Microphone separation : {D*100:.0f} cm")
    summary.append(f"    Sampling frequency    : {FS:,} Hz")
    summary.append(f"    Speed of sound        : {V} m/s")
    summary.append(f"    Max TDOA              : {MAX_TDOA_SAMPLES} samples")
    summary.append(f"    Angle range           : -90° to +90°")
    summary.append("")
    summary.append(f"  Angular RMSE at each SNR level:")
    for snr, rmse in zip(SNR_LEVELS_DB, mean_rmse_per_snr):
        bar = "#" * int(rmse)
        summary.append(f"    SNR {snr:>3} dB -> "
                       f"{rmse:>5.2f}°  {bar}")
    summary.append("")
    summary.append(f"  Direction bin accuracy @ SNR=10 dB : "
                   f"{bin_acc:.1f}%")
    summary.append("")
    summary.append("  Key findings for paper:")
    low_snr_rmse  = mean_rmse_per_snr[SNR_LEVELS_DB == -5][0]
    high_snr_rmse = mean_rmse_per_snr[SNR_LEVELS_DB == 20][0]
    mid_snr_rmse  = mean_rmse_per_snr[SNR_LEVELS_DB == 10][0]
    summary.append(f"    - At SNR=-5 dB (worst): RMSE = {low_snr_rmse:.2f}°")
    summary.append(f"    - At SNR=10 dB (typical): RMSE = {mid_snr_rmse:.2f}°")
    summary.append(f"    - At SNR=20 dB (best): RMSE = {high_snr_rmse:.2f}°")
    summary.append(f"    - Direction accuracy (10 dB): {bin_acc:.1f}%")
    summary.append("=" * 60)

    for line in summary:
        print(line)

    out_path = os.path.join(OUT_DIR, "doa_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(summary))
    print(f"\n  Saved: {out_path}")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    import time
    t0 = time.time()

    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  GCC-PHAT DOA SIMULATION")
    print("=" * 60)
    print(f"  Microphone separation : d = {D*100:.0f} cm")
    print(f"  Sampling frequency    : fs = {FS:,} Hz")
    print(f"  Speed of sound        : v = {V} m/s")
    print(f"  Max TDOA              : {MAX_TDOA_SAMPLES} samples "
          f"({MAX_TDOA_SAMPLES/FS*1000:.2f} ms)")
    print(f"  Trials per condition  : {N_TRIALS}")
    print(f"  Output directory      : {os.path.abspath(OUT_DIR)}")
    np.random.seed(42)

    # ── Simulation 1: RMSE vs SNR ──────────────────────────
    rmse_table, mae_table, mean_rmse_per_snr = run_rmse_simulation()
    print_rmse_table(rmse_table, mean_rmse_per_snr)

    # ── Simulation 2: Confusion matrix @ SNR=10 dB ────────
    cm       = run_confusion_simulation(snr_db=10)
    bin_acc  = print_confusion_matrix(cm, snr_db=10)

    # ── Plot ──────────────────────────────────────────────
    print("\n  Generating plots...")
    plot_rmse_vs_snr_a(mean_rmse_per_snr)
    plot_rmse_vs_snr_b(rmse_table)

    # ── Final summary ─────────────────────────────────────
    print_final_summary(mean_rmse_per_snr, bin_acc)

    elapsed = time.time() - t0
    print(f"\n  Total simulation time: {elapsed:.1f}s")
    print(f"\n  Output files:")
    print(f"    results/doa_rmse_table.txt")
    print(f"    results/doa_confusion_matrix.txt")
    print(f"    results/doa_rmse_vs_snr_a.png")
    print(f"    results/doa_rmse_vs_snr_a.pdf")
    print(f"    results/doa_rmse_vs_snr_b.png")
    print(f"    results/doa_rmse_vs_snr_b.pdf")
    print(f"    results/doa_summary.txt")


if __name__ == "__main__":
    main()
