# ==============================================================
# Phase3_ThresholdTuning.py
# FRACTURE DETECTION PROJECT
# ==============================================================
# Tunes optimal decision threshold per body part
# Updated for FractureClassifierExclude model
# Uses marker exclusion from marker_coords.csv
# ==============================================================

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import open_clip
from sklearn.metrics import (roc_auc_score,
                             accuracy_score,
                             f1_score,
                             recall_score,
                             precision_score,
                             roc_curve)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# 1. CONFIG
# ==============================================================

BASE_PATH    = r"d:\fracture_detection_project"

VAL_CSV      = os.path.join(BASE_PATH, "data",
                            "valid_labels_clean.csv")
if not os.path.exists(VAL_CSV):
    VAL_CSV  = os.path.join(BASE_PATH, "data",
                             "valid_labels.csv")

MARKER_CSV   = os.path.join(BASE_PATH, "data",
                            "marker_coords.csv")
PHASE1_MODEL = os.path.join(BASE_PATH, "models",
                            "saved",
                            "phase1_best_model.pt")
SAVE_DIR     = os.path.join(BASE_PATH, "models", "saved")
RESULTS_DIR  = os.path.join(BASE_PATH, "results")
THRESH_FILE  = os.path.join(SAVE_DIR,
                            "optimal_thresholds.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE         = "cuda" if torch.cuda.is_available() \
                 else "cpu"
MARKER_PADDING = 0.03

BODY_PARTS = [
    "XR_ELBOW",   "XR_FINGER", "XR_FOREARM",
    "XR_HAND",    "XR_HUMERUS","XR_SHOULDER","XR_WRIST"
]

# ==============================================================
# 2. MARKER COORDS
# ==============================================================

def load_marker_coords():
    if not os.path.exists(MARKER_CSV):
        return {}
    df      = pd.read_csv(MARKER_CSV)
    markers = {}
    for _, row in df.iterrows():
        if row["has_marker"] == 1:
            markers[row["image_path"]] = {
                "x1": max(0.0, float(row["x1"])-MARKER_PADDING),
                "y1": max(0.0, float(row["y1"])-MARKER_PADDING),
                "x2": min(1.0, float(row["x2"])+MARKER_PADDING),
                "y2": min(1.0, float(row["y2"])+MARKER_PADDING),
            }
    print(f"✅ Markers: {len(markers)}")
    return markers


def get_exclude_tensor(image_path, markers):
    if image_path in markers:
        m  = markers[image_path]
        x1,y1,x2,y2 = m["x1"],m["y1"],m["x2"],m["y2"]
        return torch.tensor(
            [x1,y1,x2,y2,(x1+x2)/2,(y1+y2)/2,
             x2-x1,y2-y1], dtype=torch.float32)
    return torch.zeros(8, dtype=torch.float32)


# ==============================================================
# 3. LORA
# ==============================================================

class LoRALinear(nn.Module):
    def __init__(self, linear, r=32, alpha=64):
        super().__init__()
        self.linear  = linear
        self.scaling = alpha / r
        in_f  = linear.weight.shape[1]
        out_f = linear.weight.shape[0]
        self.lora_A = nn.Parameter(torch.randn(r,in_f)*0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f,r))
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)
    def forward(self, x):
        return (self.linear(x)+
                (x@self.lora_A.T@self.lora_B.T)*self.scaling)


def find_blocks(model):
    for attr in ["visual.transformer.resblocks",
                 "visual.trunk.blocks",
                 "visual.model.blocks",
                 "visual.blocks"]:
        try:
            obj = model
            for a in attr.split("."): obj=getattr(obj,a)
            return obj, attr
        except AttributeError: pass
    raise AttributeError("Cannot find blocks.")


def inject_lora_block(block, r, alpha):
    if hasattr(block,"attn"):
        a = block.attn
        if hasattr(a,"out_proj") and isinstance(a.out_proj,nn.Linear):
            a.out_proj=LoRALinear(a.out_proj,r,alpha)
        elif hasattr(a,"proj") and isinstance(a.proj,nn.Linear):
            a.proj=LoRALinear(a.proj,r,alpha)
    if hasattr(block,"mlp"):
        m=block.mlp
        for attr in ["c_fc","c_proj","fc1","fc2"]:
            if hasattr(m,attr) and isinstance(getattr(m,attr),nn.Linear):
                setattr(m,attr,LoRALinear(getattr(m,attr),r,alpha))


def inject_lora(model, r=32, alpha=64, n=8):
    for p in model.parameters(): p.requires_grad_(False)
    blocks, path = find_blocks(model)
    total = len(blocks)
    for i,b in enumerate(blocks):
        if i >= total-n: inject_lora_block(b,r,alpha)
    return model


# ==============================================================
# 4. MODELS
# ==============================================================

class FractureClassifierExclude(nn.Module):
    def __init__(self, clip_model, embed_dim=512,
                 exclude_dim=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.clip_model = clip_model
        self.exclude_encoder = nn.Sequential(
            nn.Linear(8,32), nn.GELU(),
            nn.LayerNorm(32), nn.Dropout(0.1),
            nn.Linear(32,exclude_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim+exclude_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim+exclude_dim,256),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256,num_classes))

    def encode(self, x, exclude):
        img_f  = self.clip_model.encode_image(x).float()
        excl_f = self.exclude_encoder(exclude.float())
        return torch.cat([img_f,excl_f],dim=1)


class BodyPartHead(nn.Module):
    def __init__(self, embed_dim=576, num_classes=2, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim,128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128,num_classes))
    def forward(self, x): return self.head(x)


# ==============================================================
# 5. DATASET
# ==============================================================

class ValDataset(Dataset):
    def __init__(self, df, markers, transform=None):
        self.df=df.reset_index(drop=True)
        self.markers=markers
        self.transform=transform
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row     = self.df.iloc[idx]
        img     = Image.open(row["image_path"]).convert("RGB")
        exclude = get_exclude_tensor(row["image_path"],self.markers)
        label   = torch.tensor(int(row["label"]),dtype=torch.long)
        if self.transform: img=self.transform(img)
        return img, exclude, label


def get_transform():
    return transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466,0.4578275,0.40821073),
            (0.26862954,0.26130258,0.27577711))])


# ==============================================================
# 6. THRESHOLD TUNING
# ==============================================================

def find_optimal_threshold(labels, probs):
    thresholds  = np.arange(0.1, 0.9, 0.01)
    best_thresh = 0.5
    best_f1     = 0.0
    for t in thresholds:
        preds = (probs >= t).astype(int)
        try:
            score = f1_score(labels, preds,
                             zero_division=0)
            if score > best_f1:
                best_f1     = score
                best_thresh = t
        except Exception:
            continue
    return round(float(best_thresh), 3), best_f1


# ==============================================================
# 7. ROC PLOT
# ==============================================================

def plot_roc_curves(all_results, save_path):
    fig, ax = plt.subplots(figsize=(10,8))
    fig.patch.set_facecolor('#0A0E1A')
    ax.set_facecolor('#0F1524')
    colors = ['#00C8FF','#00FF88','#FF4466',
               '#FFD700','#FF8C00','#A78BFA','#FF69B4']
    for (bp,data),color in zip(all_results.items(),colors):
        fpr,tpr,_ = roc_curve(data["labels"],data["probs"])
        ax.plot(fpr,tpr,color=color,lw=2,
                label=f'{bp.replace("XR_","")} '
                      f'(AUC={data["auc"]:.3f})')
    ax.plot([0,1],[0,1],'w--',alpha=0.3,lw=1)
    ax.set_xlabel('False Positive Rate',color='#8AABCC')
    ax.set_ylabel('True Positive Rate',color='#8AABCC')
    ax.set_title('ROC Curves — With Marker Exclusion',
                 color='white',fontsize=12,fontweight='bold')
    ax.tick_params(colors='#8AABCC')
    for s in ax.spines.values(): s.set_color('#1E2D4A')
    ax.legend(facecolor='#0F1524',labelcolor='white',
               fontsize=9,edgecolor='#1E2D4A')
    plt.tight_layout()
    plt.savefig(save_path,dpi=120,bbox_inches='tight')
    plt.close()
    print(f"ROC saved: {save_path}")


# ==============================================================
# 8. MAIN
# ==============================================================

if __name__ == "__main__":

    print("="*60)
    print("PHASE 3 — Threshold Tuning")
    print("="*60)

    markers = load_marker_coords()
    val_df  = pd.read_csv(VAL_CSV)
    print(f"Val: {len(val_df):,}")

    print("\nLoading encoder...")
    clip_model,_,_ = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-"
        "PubMedBERT_256-vit_base_patch16_224")
    clip_model = inject_lora(clip_model,r=32,alpha=64,n=8)
    encoder    = FractureClassifierExclude(
        clip_model).to(DEVICE)
    encoder.load_state_dict(
        torch.load(PHASE1_MODEL,map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad_(False)
    print("✅ Encoder ready")

    tf                 = get_transform()
    optimal_thresholds = {}
    all_results        = {}

    prev_thresholds = {
        "XR_ELBOW"   :0.512, "XR_FINGER"  :0.351,
        "XR_FOREARM" :0.625, "XR_HAND"    :0.447,
        "XR_HUMERUS" :0.633, "XR_SHOULDER":0.294,
        "XR_WRIST"   :0.480,
    }

    for bp in BODY_PARTS:
        print(f"\n── {bp}")

        head_path = os.path.join(SAVE_DIR,f"head_v2_{bp}.pt")
        if not os.path.exists(head_path):
            print(f"  ⚠️ Not found — skip")
            optimal_thresholds[bp] = 0.5
            continue

        head = BodyPartHead(embed_dim=576).to(DEVICE)
        head.load_state_dict(
            torch.load(head_path,map_location=DEVICE))
        head.eval()

        bp_val = val_df[val_df["body_part"]==bp]
        if len(bp_val) == 0:
            optimal_thresholds[bp] = 0.5
            continue

        ds = ValDataset(bp_val, markers, transform=tf)
        ld = DataLoader(ds,batch_size=32,
                        shuffle=False,num_workers=0)

        all_probs, all_labels = [], []
        with torch.no_grad():
            for imgs,excludes,labels in tqdm(
                    ld,desc="  Predicting",leave=False):
                imgs=imgs.to(DEVICE)
                excludes=excludes.to(DEVICE)
                feats  = encoder.encode(imgs,excludes)
                logits = head(feats)
                probs  = torch.softmax(
                    logits,1)[:,1].cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        labels_arr = np.array(all_labels)
        probs_arr  = np.array(all_probs)
        auc        = roc_auc_score(labels_arr,probs_arr)

        opt_thresh, opt_f1 = find_optimal_threshold(
            labels_arr, probs_arr)

        # Compare default vs optimal
        preds_def = (probs_arr>=0.5).astype(int)
        preds_opt = (probs_arr>=opt_thresh).astype(int)

        rec_def = recall_score(labels_arr,preds_def,
                               zero_division=0)
        rec_opt = recall_score(labels_arr,preds_opt,
                               zero_division=0)
        f1_def  = f1_score(labels_arr,preds_def,
                           zero_division=0)

        prev = prev_thresholds.get(bp,0.5)
        print(f"  AUC         : {auc:.4f}")
        print(f"  Prev thresh : {prev:.3f} → "
              f"New: {opt_thresh:.3f}")
        print(f"  Recall      : "
              f"{rec_def:.3f} → {rec_opt:.3f} "
              f"({rec_opt-rec_def:+.3f})")
        print(f"  F1          : "
              f"{f1_def:.3f} → {opt_f1:.3f} "
              f"({opt_f1-f1_def:+.3f})")

        optimal_thresholds[bp] = opt_thresh
        all_results[bp] = {
            "labels"   : labels_arr,
            "probs"    : probs_arr,
            "auc"      : auc,
            "threshold": opt_thresh,
        }

    # Save thresholds
    with open(THRESH_FILE,"w") as f:
        json.dump(optimal_thresholds,f,indent=4)
    print(f"\n✅ Thresholds saved: {THRESH_FILE}")

    # ROC curves
    plot_roc_curves(
        all_results,
        os.path.join(RESULTS_DIR,
                     "phase3_roc_curves.png"))

    # Summary
    print("\n"+"="*60)
    print("PHASE 3 COMPLETE")
    print("="*60)
    print(f"\n{'Body Part':<15} {'Threshold':>10} "
          f"{'AUC':>8}")
    print("-"*36)
    aucs = []
    for bp in BODY_PARTS:
        if bp not in all_results: continue
        t   = optimal_thresholds[bp]
        auc = all_results[bp]["auc"]
        aucs.append(auc)
        print(f"  {bp:<13} {t:>10.3f} {auc:>8.4f}")
    if aucs:
        print("-"*36)
        print(f"  {'MEAN':<13} "
              f"{'':>10} {np.mean(aucs):>8.4f}")

    print(f"\nThresholds  : {THRESH_FILE}")
    print("Next        : streamlit run app.py")