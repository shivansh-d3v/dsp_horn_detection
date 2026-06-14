import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import librosa
from sklearn.metrics import classification_report, confusion_matrix

# Audio / spectrogram parameters
SAMPLE_RATE    = 16_000
DURATION_MS    = 633
TARGET_SAMPLES = int(SAMPLE_RATE * DURATION_MS / 1000)
N_FFT          = 400
HOP_LENGTH     = 160
N_MELS         = 64

class FullDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        for f in glob.glob(os.path.join(root_dir, "horn", "*.wav")):
            self.samples.append((f, 1))
        for f in glob.glob(os.path.join(root_dir, "background", "*.wav")):
            self.samples.append((f, 0))

        # Also support kfold_dataset layout
        for f in glob.glob(os.path.join(root_dir, "horns", "*.wav")):
            self.samples.append((f, 1))
        for f in glob.glob(os.path.join(root_dir, "noise", "*.wav")):
            self.samples.append((f, 0))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        filepath, label = self.samples[idx]
        y, _ = librosa.load(filepath, sr=SAMPLE_RATE)
        if len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode="constant")
        else:
            y = y[:TARGET_SAMPLES]
        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLE_RATE,
            n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS,
        )
        log_mel = np.log(mel + 0.01)
        spec_tensor  = torch.from_numpy(log_mel).unsqueeze(0).float()
        return spec_tensor, label

class MCUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,  16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def main():
    model = MCUModel()
    # load state dict, ignoring missing or unexpected keys in case dropout index differs
    model.load_state_dict(torch.load('best_horn_detector_mcu.pth', map_location='cpu'))
    model.eval()
    
    # We will evaluate on Fold_1/Test by default, as that's a true test set
    test_dir = os.path.join('kfold_dataset', 'Fold_1', 'Test')
    if not os.path.exists(test_dir) or len(glob.glob(os.path.join(test_dir, '*/*.wav'))) == 0:
        test_dir = 'dataset'
        
    print(f"Evaluating on {test_dir}...")
    dataset = FullDataset(test_dir)
    print(f"Total samples: {len(dataset)}")
    test_loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            if i % 10 == 0:
                print(f"Processing batch {i}/{len(test_loader)}...")
            outputs = model(inputs)
            preds = (torch.sigmoid(outputs) >= 0.5).int().view(-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['No Horn', 'Horn']))
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    main()
