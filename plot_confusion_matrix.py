import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def main():
    # Configure matplotlib for academic, publication-quality plots
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "text.usetex": False,  # Safe default on all platforms
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.linewidth": 0.6,
    })

    # Dimensions matching IEEE column width (3.5" width, slightly taller for labels)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Confusion Matrix Data:
    # Row 0: True Horn (610 total) -> 604 TP, 6 FN
    # Row 1: True Noise (7923 total) -> 146 FP, 7777 TN
    cm = np.array([
        [604, 6],
        [146, 7777]
    ])

    # Class Labels
    labels = ["Horn", "Noise"]

    # Calculate row-wise percentages for normalized shading (Recall / Specificity)
    # This prevents the massive count difference (7,777 vs 604) from washing out the Horn cell.
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_pct = cm / row_sums

    # Premium slate-blue sequential colormap for maximum visual aesthetics
    colors = ["#FFFFFF", "#EFF6FF", "#BFDBFE", "#3B82F6", "#1E3A8A"]
    cmap = LinearSegmentedColormap.from_list("custom_blue", colors, N=256)

    # Plot the matrix
    im = ax.imshow(cm_pct, cmap=cmap, vmin=0, vmax=1)

    # Customize axis lines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#475569")

    # Set ticks and sizes
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_yticklabels(labels, fontsize=10)

    # Set labels & Title with journal requirements
    ax.set_xlabel("Predicted Class", fontsize=10, fontweight="bold", labelpad=4)
    ax.set_ylabel("True Class", fontsize=10, fontweight="bold", labelpad=4)
    ax.set_title("Confusion Matrix (Fold 5)", fontsize=10, fontweight="bold", pad=8)

    # Overlay numbers and percentages
    # If the cell's normalized percentage value is high, use white text; otherwise dark slate.
    thresh = 0.5
    for i in range(len(labels)):
        for j in range(len(labels)):
            count = cm[i, j]
            pct = cm_pct[i, j] * 100
            text_color = "#FFFFFF" if cm_pct[i, j] > thresh else "#0F172A"
            
            # Format display text: "Count \n (Percentage)"
            text = f"{count:,}\n({pct:.2f}%)"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color=text_color,
                fontsize=9.5,
                fontweight="bold"
            )

    # Adjust margins tightly
    plt.tight_layout(pad=0.1)

    # Ensure output directory exists
    OUT_DIR = "results"
    os.makedirs(OUT_DIR, exist_ok=True)

    # Save vector PDF and high-resolution PNG
    out_png = os.path.join(OUT_DIR, "fold5_confusion_matrix.png")
    out_pdf = os.path.join(OUT_DIR, "fold5_confusion_matrix.pdf")

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()

    print(f"Successfully generated Confusion Matrix:")
    print(f"  PNG: {out_png}")
    print(f"  PDF: {out_pdf}")

if __name__ == "__main__":
    main()
