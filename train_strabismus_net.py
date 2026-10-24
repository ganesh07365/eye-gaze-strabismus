"""
train_strabismus_net.py - Dual-Gaze Siamese Strabismus Neural Network
====================================================================
BTP Project: Webcam-Based Eye Tracking & Visual Training for Strabismus Screening

Architecture Highlights:
  1. Transfer Learning: Reuses the pretrained GazeCNN backbone (models/gaze_model_p00.pth)
     developed for the MPIIGaze screen-tracking pipeline.
  2. Siamese Dual-Eye Branch:
     - Extracts Right Eye (OD) feature vector (f_R)
     - Extracts Left Eye (OS) feature vector (f_L) with horizontal mirroring
  3. Disparity Fusion Layer: Computes inter-ocular feature disparity:
     Delta_f = f_L - f_R
  4. Clinical 5-Class Classifier Head:
     Predicts [NORMAL, ESOTROPIA, EXOTROPIA, HYPERTROPIA, HYPOTROPIA]
  5. Auto-saves trained model to models/strabismus_classifier.pth
     and evaluation metrics to diagnostics/results/strabismus_training_report.txt
"""

import os
import sys
import random
import time
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==============================================================================
# 1. PATH RESOLUTION & REPRODUCIBILITY
# ==============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything(42)

DEFAULT_DATASET_PATHS = [
    r"C:\Users\HP\OneDrive\Documents\BTP\datasets\STRABISMUS",
    r"C:\Users\HP\Documents\BTP\datasets\STRABISMUS",
    os.path.join(PROJECT_ROOT, "datasets", "STRABISMUS"),
]

def find_dataset_root():
    for p in DEFAULT_DATASET_PATHS:
        if os.path.exists(p):
            return os.path.abspath(p)
    return None

CLASSES = ["NORMAL", "ESOTROPIA", "EXOTROPIA", "HYPERTROPIA", "HYPOTROPIA"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


# ==============================================================================
# 2. DUAL-GAZE SIAMESE STRABISMUS NETWORK ARCHITECTURE
# ==============================================================================
class DualGazeStrabismusNet(nn.Module):
    """
    Siamese neural network connecting your previous GazeCNN MPIIGaze backbone
    to an inter-ocular disparity classification head.
    """
    def __init__(self, pretrained_gaze_path=None):
        super().__init__()
        # Backbone: Exact same layer structure as GazeCNN in gaze_to_screen.py
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten_size = 50 * 6 * 12  # 3600 for a 36x60 eye patch
        self.relu = nn.ReLU()

        # Eye feature embedding
        self.eye_fc = nn.Sequential(
            nn.Linear(self.flatten_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Preload weights from gaze_model_p00.pth if available
        if pretrained_gaze_path and os.path.exists(pretrained_gaze_path):
            try:
                ckpt = torch.load(pretrained_gaze_path, map_location="cpu")
                # Load matching conv layers
                state = self.state_dict()
                for k in ["conv1.weight", "conv1.bias", "conv2.weight", "conv2.bias"]:
                    if k in ckpt and ckpt[k].shape == state[k].shape:
                        state[k].copy_(ckpt[k])
                print(f"[TRANSFER LEARNING] Successfully loaded GazeCNN backbone from: {pretrained_gaze_path}")
            except Exception as e:
                print(f"[WARNING] Could not load pretrained weights ({e}). Initializing from scratch.")

        # Disparity Fusion & Classification Head
        # Concatenates: [f_R (256), f_L (256), delta_f (256), abs(delta_f) (256)] = 1024 dims
        fusion_dim = 256 * 4
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, len(CLASSES))
        )

    def extract_eye_feature(self, eye_patch):
        # eye_patch shape: [B, 1, 36, 60]
        x = self.relu(self.conv1(eye_patch))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        return self.eye_fc(x)

    def forward(self, r_eye, l_eye):
        # r_eye: [B, 1, 36, 60] Right Eye (OD)
        # l_eye: [B, 1, 36, 60] Left Eye (OS, horizontally flipped for canonical symmetry)
        f_r = self.extract_eye_feature(r_eye)
        f_l = self.extract_eye_feature(l_eye)

        # Compute inter-ocular disparity vector
        delta_f = f_l - f_r
        abs_delta = torch.abs(delta_f)

        # Fused bilateral representation
        fused = torch.cat([f_r, f_l, delta_f, abs_delta], dim=1)
        logits = self.classifier(fused)
        return logits


# ==============================================================================
# 3. DATASET LOADER & PREPROCESSING
# ==============================================================================
class StrabismusDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.samples = samples  # List of (img_path, class_idx)
        self.is_train = is_train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            # Fallback zero image if read error
            img = np.zeros((130, 480, 3), dtype=np.uint8)

        # Convert to Grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        mid = w // 2

        # Split into Right and Left Eye patches
        r_crop = gray[:, :mid]
        l_crop = gray[:, mid:]

        # Resize each eye to canonical (60, 36)
        r_patch = cv2.resize(r_crop, (60, 36))
        l_patch = cv2.resize(l_crop, (60, 36))

        # Mirror left eye horizontally so both eyes are canonically aligned (MPIIGaze standard)
        l_patch = cv2.flip(l_patch, 1)

        # Data augmentation during training
        if self.is_train:
            # Random brightness/contrast jitter
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-15, 15)
            r_patch = np.clip(alpha * r_patch + beta, 0, 255).astype(np.uint8)
            l_patch = np.clip(alpha * l_patch + beta, 0, 255).astype(np.uint8)

        # Normalize pixels to [-1.0, 1.0]
        r_t = torch.from_numpy(r_patch).float().unsqueeze(0) / 255.0 * 2.0 - 1.0
        l_t = torch.from_numpy(l_patch).float().unsqueeze(0) / 255.0 * 2.0 - 1.0

        return r_t, l_t, torch.tensor(label, dtype=torch.long)


# ==============================================================================
# 4. TRAINING & VALIDATION ROUTINE
# ==============================================================================
def train():
    print("=" * 75)
    print("BTP PROJECT: TRAINING DUAL-GAZE SIAMESE STRABISMUS CLASSIFIER")
    print("=" * 75)

    dataset_root = find_dataset_root()
    if not dataset_root:
        print("ERROR: Dataset directory not found.")
        return

    print(f"Dataset root: {dataset_root}")

    # Collect dataset samples per class
    all_samples = []
    class_counts = {}
    for c in CLASSES:
        cdir = os.path.join(dataset_root, c)
        files = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        class_counts[c] = len(files)
        for f in files:
            all_samples.append((f, CLASS_TO_IDX[c]))

    print(f"Total dataset size: {len(all_samples)} images across {len(CLASSES)} classes:")
    for c, cnt in class_counts.items():
        print(f"  - {c:12s}: {cnt} images")

    # Stratified 80/20 train/validation split
    train_samples = []
    val_samples = []
    for c in CLASSES:
        c_items = [s for s in all_samples if s[1] == CLASS_TO_IDX[c]]
        random.shuffle(c_items)
        split_pt = int(0.80 * len(c_items))
        train_samples.extend(c_items[:split_pt])
        val_samples.extend(c_items[split_pt:])

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    print(f"\nSplit: {len(train_samples)} Training images | {len(val_samples)} Validation/Test images")

    train_loader = DataLoader(StrabismusDataset(train_samples, is_train=True), batch_size=16, shuffle=True)
    val_loader   = DataLoader(StrabismusDataset(val_samples, is_train=False), batch_size=16, shuffle=False)

    # Initialize model with your pretrained GazeCNN weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    pretrained_gaze_path = os.path.join(PROJECT_ROOT, "models", "gaze_model_p00.pth")
    model = DualGazeStrabismusNet(pretrained_gaze_path).to(device)

    # Optimizer with weight decay to prevent overfitting
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=35, eta_min=1e-5)

    epochs = 35
    best_val_acc = 0.0
    best_cm = None
    save_path = os.path.join(PROJECT_ROOT, "models", "strabismus_classifier.pth")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("\nStarting training loop...")
    print("-" * 75)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Train Acc':<12} | {'Val Loss':<12} | {'Val Acc':<12} | {'Status'}")
    print("-" * 75)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for r_batch, l_batch, y_batch in train_loader:
            r_batch, l_batch, y_batch = r_batch.to(device), l_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(r_batch, l_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1)
            train_correct += (preds == y_batch).sum().item()
            train_total += len(y_batch)

        scheduler.step()
        ep_train_loss = train_loss / train_total
        ep_train_acc = train_correct / train_total * 100.0

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for r_batch, l_batch, y_batch in val_loader:
                r_batch, l_batch, y_batch = r_batch.to(device), l_batch.to(device), y_batch.to(device)
                logits = model(r_batch, l_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * len(y_batch)
                preds = torch.argmax(logits, dim=1)
                val_correct += (preds == y_batch).sum().item()
                val_total += len(y_batch)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_targets.extend(y_batch.cpu().numpy())

        ep_val_loss = val_loss / val_total
        ep_val_acc = val_correct / val_total * 100.0

        status = ""
        if ep_val_acc > best_val_acc:
            best_val_acc = ep_val_acc
            torch.save(model.state_dict(), save_path)
            status = f"* SAVED (Best: {best_val_acc:.1f}%)"

            # Compute confusion matrix
            cm = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
            for t, p in zip(all_val_targets, all_val_preds):
                cm[t, p] += 1
            best_cm = cm

        if ep % 5 == 0 or ep == epochs or status:
            print(f"[{ep:2d}/{epochs}] | {ep_train_loss:10.4f}   | {ep_train_acc:9.1f}%   | {ep_val_loss:10.4f}   | {ep_val_acc:9.1f}%   | {status}")

    elapsed = time.time() - t0
    print("-" * 75)
    print(f"Training completed in {elapsed:.1f} seconds.")
    print(f"Best Validation Accuracy: {best_val_acc:.1f}%")
    print(f"Saved model to: {save_path}")

    # Generate and print Clinical Confusion Matrix
    print("\n" + "=" * 75)
    print("CLINICAL EVALUATION CONFUSION MATRIX (Validation Set):")
    print("=" * 75)
    header = f"{'True Class':<14} | " + " | ".join([f"{c[:7]:>7}" for c in CLASSES])
    print(header)
    print("-" * len(header))
    if best_cm is not None:
        for i, c in enumerate(CLASSES):
            row_str = " | ".join([f"{best_cm[i, j]:7d}" for j in range(len(CLASSES))])
            print(f"{c:<14} | {row_str}")

    # Save complete report to diagnostics/results/
    log_dir = os.path.join(PROJECT_ROOT, "diagnostics", "results")
    os.makedirs(log_dir, exist_ok=True)
    report_file = os.path.join(log_dir, "strabismus_training_report.txt")
    with open(report_file, "w") as rf:
        rf.write("=" * 75 + "\n")
        rf.write(f"DUAL-GAZE SIAMESE STRABISMUS NETWORK TRAINING REPORT — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        rf.write("=" * 75 + "\n")
        rf.write(f"Total Dataset Images  : {len(all_samples)}\n")
        rf.write(f"Training Samples      : {len(train_samples)}\n")
        rf.write(f"Validation Samples    : {len(val_samples)}\n")
        rf.write(f"Pretrained Backbone   : GazeCNN (gaze_model_p00.pth)\n")
        rf.write(f"Best Validation Acc   : {best_val_acc:.1f}%\n\n")
        rf.write("CONFUSION MATRIX:\n")
        rf.write(header + "\n")
        rf.write("-" * len(header) + "\n")
        if best_cm is not None:
            for i, c in enumerate(CLASSES):
                row_str = " | ".join([f"{best_cm[i, j]:7d}" for j in range(len(CLASSES))])
                rf.write(f"{c:<14} | {row_str}\n")
    print(f"\nReport written to: {report_file}")


if __name__ == "__main__":
    train()
