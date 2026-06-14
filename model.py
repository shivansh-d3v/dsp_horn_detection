import torch
import torch.nn as nn

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

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)   # (B, 4096)
        x = self.classifier(x)
        return x                      # (B, 1) raw logit
