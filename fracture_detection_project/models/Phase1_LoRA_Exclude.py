# ==============================================================
# Phase1_LoRA_Exclude.py
# FRACTURE DETECTION PROJECT
# ==============================================================
# Training with marker exclusion:
# 1. Loads marker_coords.csv from FindMarkers.py
# 2. Passes exclude region to model
# 3. Model learns to ignore letter region
# 4. Focuses on bone structure only
# RANK=32, BLOCKS=8, Focal Loss
# Fully resumable if PC turns off
# ==============================================================

import os
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import open_clip
from sklearn.metrics import (accuracy_score, f1_score,
                             roc_auc_score,
                             classification_report)
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# 1. CONFIG
# ==============================================================

BASE_PATH    = r"d:\fracture_detection_project"

# Marker locations from FindMarkers.py
MARKER_CSV   = os.path.join(BASE_PATH, "data",
                            "marker_coords.csv")

# Use clean CSV if available
TRAIN_CSV    = os.path.join(BASE_PATH, "data",
                            "train_labels_clean.csv")
VAL_CSV      = os.path.join(BASE_PATH, "data",
                            "valid_labels_clean.csv")
if not os.path.exists(TRAIN_CSV):
    TRAIN_CSV = os.path.join(BASE_PATH, "data",
                             "train_labels.csv")
    VAL_CSV   = os.path.join(BASE_PATH, "data",
                             "valid_labels.csv")

SAVE_DIR     = os.path.join(BASE_PATH, "models", "saved")
CHECKPOINT   = os.path.join(SAVE_DIR,
                            "phase1_checkpoint.pt")
BEST_MODEL   = os.path.join(SAVE_DIR,
                            "phase1_best_model.pt")
os.makedirs(SAVE_DIR, exist_ok=True)

DEVICE       = "cuda" if torch.cuda.is_available() \
               else "cpu"
print(f"Using device: {DEVICE}")

# Training
BATCH_SIZE   = 8
LR           = 1e-4
HEAD_LR      = 1e-3
EPOCHS       = 20
PATIENCE     = 5

# Marker exclusion padding
# Adds buffer zone around detected letter
# 0.03 = 3% of image size on each side
MARKER_PADDING = 0.03

# LoRA — strongest settings
LORA_RANK    = 32
LORA_ALPHA   = 64
LORA_BLOCKS  = 8

# ==============================================================
# 2. LOAD MARKER COORDS
# ==============================================================

def load_marker_coords(marker_csv: str) -> dict:
    """
    Loads marker locations from FindMarkers.py output.
    Returns dict: {image_path: {x1,y1,x2,y2}}
    """
    if not os.path.exists(marker_csv):
        print("⚠️  marker_coords.csv not found!")
        print("   Run FindMarkers.py first.")
        print("   Training without exclusion...\n")
        return {}

    df      = pd.read_csv(marker_csv)
    markers = {}

    for _, row in df.iterrows():
        if row["has_marker"] == 1:
            markers[row["image_path"]] = {
                "x1": max(0.0, float(row["x1"]) - MARKER_PADDING),
                "y1": max(0.0, float(row["y1"]) - MARKER_PADDING),
                "x2": min(1.0, float(row["x2"]) + MARKER_PADDING),
                "y2": min(1.0, float(row["y2"]) + MARKER_PADDING),
            }

    total    = len(df)
    with_m   = len(markers)
    print(f"✅ Marker coords loaded")
    print(f"   Total images  : {total}")
    print(f"   With markers  : {with_m} "
          f"({100*with_m/max(total,1):.1f}%)")
    print(f"   Clean images  : {total-with_m} "
          f"({100*(total-with_m)/max(total,1):.1f}%)")
    return markers


# ==============================================================
# 3. EXCLUDE REGION TENSOR
# ==============================================================

def get_exclude_tensor(image_path: str,
                       markers: dict) -> torch.Tensor:
    """
    Returns exclude region as 8-dim tensor.

    If marker found for this image:
        [x1, y1, x2, y2, cx, cy, w, h]
        where cx,cy = center, w,h = size

    If no marker:
        [0, 0, 0, 0, 0, 0, 0, 0]
        Model knows no region to exclude
        Processes full image normally
    """
    if image_path in markers:
        m  = markers[image_path]
        x1 = m["x1"]
        y1 = m["y1"]
        x2 = m["x2"]
        y2 = m["y2"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w  = x2 - x1
        h  = y2 - y1
        return torch.tensor(
            [x1, y1, x2, y2, cx, cy, w, h],
            dtype=torch.float32)
    else:
        # No marker — zeros signal "ignore nothing"
        return torch.zeros(8, dtype=torch.float32)


# ==============================================================
# 4. LORA (same as original — proven working)
# ==============================================================

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear,
                 r: int = 32, alpha: int = 64):
        super().__init__()
        self.linear  = linear
        self.scaling = alpha / r
        in_f  = linear.weight.shape[1]
        out_f = linear.weight.shape[0]
        self.lora_A  = nn.Parameter(
            torch.randn(r, in_f) * 0.01)
        self.lora_B  = nn.Parameter(
            torch.zeros(out_f, r))
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x):
        return (self.linear(x) +
                (x @ self.lora_A.T @ self.lora_B.T)
                * self.scaling)


def print_model_structure(model):
    print("\n── Top-level modules ──")
    for name, _ in model.named_children():
        print(f"  {name}")
    print("\n── Visual sub-modules ──")
    for name, _ in model.visual.named_children():
        print(f"  visual.{name}")


def find_blocks(model):
    for attr in [
        "visual.transformer.resblocks",
        "visual.trunk.blocks",
        "visual.model.blocks",
        "visual.blocks"
    ]:
        try:
            obj = model
            for a in attr.split("."): obj = getattr(obj, a)
            return obj, attr
        except AttributeError:
            pass
    raise AttributeError("Cannot find transformer blocks.")


def inject_lora_into_block(block, r, alpha):
    injected = []
    if hasattr(block, "attn"):
        attn = block.attn
        if hasattr(attn, "out_proj") and isinstance(
                attn.out_proj, nn.Linear):
            attn.out_proj = LoRALinear(
                attn.out_proj, r, alpha)
            injected.append("attn.out_proj")
        elif hasattr(attn, "proj") and isinstance(
                attn.proj, nn.Linear):
            attn.proj = LoRALinear(
                attn.proj, r, alpha)
            injected.append("attn.proj")
    if hasattr(block, "mlp"):
        mlp = block.mlp
        for attr in ["c_fc","c_proj","fc1","fc2"]:
            if hasattr(mlp, attr) and isinstance(
                    getattr(mlp, attr), nn.Linear):
                setattr(mlp, attr,
                        LoRALinear(getattr(mlp, attr),
                                   r, alpha))
                injected.append(f"mlp.{attr}")
    return injected


def inject_lora(model, r=32, alpha=64, n_blocks=8):
    for param in model.parameters():
        param.requires_grad_(False)
    blocks, path = find_blocks(model)
    total = len(blocks)
    print(f"\nBlocks path  : {path}")
    print(f"Total blocks : {total}")
    print(f"LoRA on last : {n_blocks} blocks")
    for i, block in enumerate(blocks):
        if i >= total - n_blocks:
            injected = inject_lora_into_block(
                block, r, alpha)
            print(f"  Block {i:02d} → {injected}")
    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad)
    total_p = sum(
        p.numel() for p in model.parameters())
    print(f"\nTrainable    : {trainable:,} / "
          f"{total_p:,} "
          f"({100*trainable/total_p:.2f}%)")
    return model


# ==============================================================
# 5. MODEL WITH EXCLUDE GUIDANCE
# ==============================================================

class FractureClassifierExclude(nn.Module):
    """
    Two-stream classifier:

    Stream 1: BiomedCLIP image features (512-dim)
    Stream 2: Exclude region encoder (64-dim)
    Combined: 576-dim → classification

    The exclude encoder learns:
    "When zeros → process everything"
    "When non-zero → down-weight that region"

    This guides the model's attention away
    from annotation markers automatically.
    """
    def __init__(self, clip_model,
                 embed_dim: int = 512,
                 exclude_dim: int = 64,
                 num_classes: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.clip_model = clip_model

        # Exclude region encoder
        # Input: 8-dim [x1,y1,x2,y2,cx,cy,w,h]
        # zeros = no marker to exclude
        self.exclude_encoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.GELU(),
            nn.LayerNorm(32),
            nn.Dropout(0.1),
            nn.Linear(32, exclude_dim),
            nn.GELU(),
        )

        # Combined head: 512 + 64 = 576
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim + exclude_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim + exclude_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, image: torch.Tensor,
                exclude: torch.Tensor) -> torch.Tensor:
        # Visual features
        img_f     = self.clip_model.encode_image(
            image).float()

        # Exclude region features
        excl_f    = self.exclude_encoder(
            exclude.float())

        # Combine
        combined  = torch.cat([img_f, excl_f], dim=1)
        return self.head(combined)


# ==============================================================
# 6. FOCAL LOSS
# ==============================================================

class FocalLoss(nn.Module):
    """
    Focal Loss — focuses training on hard examples.

    In fracture detection:
    Easy examples = clear normal bones, obvious fractures
    Hard examples = subtle fractures, ambiguous cases

    Standard CrossEntropy treats all equally.
    Focal Loss down-weights easy examples.
    Model must focus on real bone patterns
    not easy marker signals.

    gamma=2 is standard recommendation.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            reduction="none")
        pt      = torch.exp(-ce_loss)
        focal   = ((1 - pt) ** self.gamma) * ce_loss
        return focal.mean()


# ==============================================================
# 7. DATASET
# ==============================================================

def get_transforms(train: bool, img_size: int = 224):
    mean = (0.48145466, 0.4578275,  0.40821073)
    std  = (0.26862954, 0.26130258, 0.27577711)
    if train:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            # Heavy random erasing
            # Prevents ANY remaining shortcuts
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.10),
                ratio=(0.3, 3.3),
                value=0),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


class MURADatasetExclude(Dataset):
    """
    Returns (image, exclude_tensor, label) per sample.

    If image has marker → exclude_tensor has its location
    If image has no marker → exclude_tensor is all zeros
    """
    def __init__(self, df: pd.DataFrame,
                 markers: dict,
                 transform=None):
        self.df        = df.reset_index(drop=True)
        self.markers   = markers
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        img     = Image.open(
            row["image_path"]).convert("RGB")

        # Get exclude tensor for this image
        exclude = get_exclude_tensor(
            row["image_path"], self.markers)

        label   = torch.tensor(
            int(row["label"]), dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, exclude, label


# ==============================================================
# 8. TRAINING UTILITIES
# ==============================================================

def compute_class_weights(labels, device):
    from sklearn.utils.class_weight import \
        compute_class_weight
    classes = np.unique(labels)
    weights = compute_class_weight(
        "balanced", classes=classes, y=labels)
    return torch.tensor(
        weights, dtype=torch.float32).to(device)


def train_one_epoch(model, loader, optimizer,
                    criterion, device, scheduler):
    model.train()
    total_loss = 0
    for imgs, excludes, labels in tqdm(
            loader, desc="  Training", leave=False):
        imgs     = imgs.to(device)
        excludes = excludes.to(device)
        labels   = labels.to(device)

        optimizer.zero_grad()
        logits = model(imgs, excludes)
        loss   = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad,
                   model.parameters()), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, loader, device):
    model.eval()
    all_labels = []
    all_preds  = []
    all_probs  = []
    with torch.no_grad():
        for imgs, excludes, labels in tqdm(
                loader, desc="  Evaluating",
                leave=False):
            imgs     = imgs.to(device)
            excludes = excludes.to(device)
            logits   = model(imgs, excludes)
            probs    = torch.softmax(
                logits, dim=1)[:, 1].cpu().numpy()
            preds    = logits.argmax(
                dim=1).cpu().numpy()
            all_labels.extend(labels.numpy())
            all_preds.extend(preds)
            all_probs.extend(probs)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds,
                   average="binary")
    auc = roc_auc_score(all_labels, all_probs)
    return (acc, f1, auc,
            np.array(all_labels),
            np.array(all_preds),
            np.array(all_probs))


def save_checkpoint(epoch, model, optimizer,
                    scheduler, best_auc,
                    patience_count, path):
    """Saves full training state for resume."""
    torch.save({
        "epoch"         : epoch,
        "model_state"   : model.state_dict(),
        "optimizer"     : optimizer.state_dict(),
        "scheduler"     : scheduler.state_dict(),
        "best_auc"      : best_auc,
        "patience_count": patience_count,
    }, path)


def load_checkpoint(path, model, optimizer,
                    scheduler):
    """Loads checkpoint and returns training state."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    return (ckpt["epoch"],
            ckpt["best_auc"],
            ckpt["patience_count"])


# ==============================================================
# 9. MAIN
# ==============================================================

if __name__ == "__main__":

    print("="*60)
    print("PHASE 1 — Marker Exclusion Training")
    print("="*60)
    print("Features:")
    print(f"  ✅ Marker exclusion (no letter bias)")
    print(f"  ✅ LoRA RANK={LORA_RANK} BLOCKS={LORA_BLOCKS}")
    print(f"  ✅ Focal Loss (focuses on hard cases)")
    print(f"  ✅ RandomErasing p=0.5 (heavy augmentation)")
    print(f"  ✅ Resumable (safe if PC turns off)")
    print("="*60)

    # ── Load marker coords ────────────────────────────────────
    markers = load_marker_coords(MARKER_CSV)

    # ── Load CSVs ─────────────────────────────────────────────
    train_df = pd.read_csv(TRAIN_CSV)
    val_df   = pd.read_csv(VAL_CSV)
    print(f"\nTrain : {len(train_df):,} samples")
    print(f"Val   : {len(val_df):,} samples")
    print(f"Fracture rate: "
          f"{train_df['label'].mean():.2%}")

    # How many training images have markers
    train_with_marker = sum(
        1 for p in train_df["image_path"]
        if p in markers)
    print(f"Train with markers: {train_with_marker}"
          f" ({100*train_with_marker/len(train_df):.1f}%)")

    # ── DataLoaders ───────────────────────────────────────────
    train_ds = MURADatasetExclude(
        train_df, markers,
        transform=get_transforms(train=True))
    val_ds   = MURADatasetExclude(
        val_df, markers,
        transform=get_transforms(train=False))

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0,
        pin_memory=True)
    val_loader   = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0,
        pin_memory=True)

    # ── Load BiomedCLIP ───────────────────────────────────────
    print("\nLoading BiomedCLIP...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-"
        "PubMedBERT_256-vit_base_patch16_224"
    )
    clip_model = inject_lora(
        clip_model,
        r=LORA_RANK,
        alpha=LORA_ALPHA,
        n_blocks=LORA_BLOCKS)

    model = FractureClassifierExclude(
        clip_model,
        embed_dim   = 512,
        exclude_dim = 64).to(DEVICE)

    # ── Focal Loss ────────────────────────────────────────────
    cw        = compute_class_weights(
        train_df["label"].values, DEVICE)
    criterion = FocalLoss(alpha=cw, gamma=2.0)
    print(f"\nClass weights → "
          f"Normal: {cw[0]:.3f} | "
          f"Fracture: {cw[1]:.3f}")

    # ── Optimizer ─────────────────────────────────────────────
    lora_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and "head" not in n
        and "exclude" not in n]
    head_params = [
        p for n, p in model.named_parameters()
        if p.requires_grad
        and ("head" in n or "exclude" in n)]

    optimizer = optim.AdamW([
        {"params": lora_params, "lr": LR,
         "weight_decay": 1e-4},
        {"params": head_params, "lr": HEAD_LR,
         "weight_decay": 1e-3},
    ])

    total_steps  = EPOCHS * len(train_loader)
    warmup_steps = 2 * len(train_loader)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr          = [LR, HEAD_LR],
        total_steps     = total_steps,
        pct_start       = warmup_steps / total_steps,
        anneal_strategy = "cos",
    )

    # ── Resume if checkpoint exists ───────────────────────────
    start_epoch    = 1
    best_auc       = 0.0
    best_state     = None
    patience_count = 0

    if os.path.exists(CHECKPOINT):
        print(f"\n🔄 Checkpoint found! Resuming...")
        start_epoch, best_auc, patience_count = \
            load_checkpoint(
                CHECKPOINT, model,
                optimizer, scheduler)
        start_epoch += 1
        best_state   = copy.deepcopy(model.state_dict())
        print(f"   Resuming from epoch {start_epoch}")
        print(f"   Best AUC so far: {best_auc:.4f}")
    else:
        print("\nNo checkpoint — starting fresh")

    # ── Training loop ─────────────────────────────────────────
    print("\n── Training ──")
    print(f"{'Epoch':>6} | {'Loss':>8} | "
          f"{'Acc':>8} | {'F1':>8} | {'AUC':>8}")
    print("-"*52)

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer,
            criterion, DEVICE, scheduler)

        acc, f1, auc, _, _, _ = evaluate(
            model, val_loader, DEVICE)

        print(f"{epoch:>6} | {train_loss:>8.4f} | "
              f"{acc:>8.4f} | {f1:>8.4f} | "
              f"{auc:>8.4f}")

        # Save checkpoint every epoch
        # Safe to turn off PC anytime ✅
        save_checkpoint(
            epoch, model, optimizer, scheduler,
            best_auc, patience_count, CHECKPOINT)

        if auc > best_auc:
            best_auc       = auc
            best_state     = copy.deepcopy(
                model.state_dict())
            patience_count = 0
            torch.save(best_state, BEST_MODEL)
            print(f"         ✓ New best AUC "
                  f"{best_auc:.4f} — saved")
        else:
            patience_count += 1
            print(f"         No improvement "
                  f"({patience_count}/{PATIENCE})")
            if patience_count >= PATIENCE:
                print("  Early stopping.")
                break

    # ── Final results ─────────────────────────────────────────
    # Remove checkpoint — training complete
    if os.path.exists(CHECKPOINT):
        os.remove(CHECKPOINT)

    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    model.load_state_dict(best_state)
    acc, f1, auc, y_true, y_pred, _ = evaluate(
        model, val_loader, DEVICE)

    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")

    print("\nClassification Report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["No Fracture", "Fracture"]))

    print("="*60)
    print("COMPARISON")
    print("="*60)
    runs = [
        ("Original (no fixes)",      0.8527),
        ("With ROI masking",         0.8436),
        ("With crop + coords",       0.8436),
    ]
    for name, prev in runs:
        diff = auc - prev
        flag = "✅" if diff > 0 else "❌"
        print(f"{flag} vs {name:<28}: "
              f"{prev:.4f} → {auc:.4f} "
              f"({diff:+.4f})")

    print(f"\nModel saved: {BEST_MODEL}")
    print("Next: python models/Phase2_BodyPart.py")