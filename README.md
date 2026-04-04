# 🩻 FractureAI — Musculoskeletal Fracture Detection

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00C8FF?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-00FF88?style=for-the-badge)

**A clinical-grade AI system for bone fracture detection in musculoskeletal X-rays**

*BiomedCLIP · LoRA Fine-tuning · Marker Exclusion · Qwen3-VL · Streamlit*

[Demo](#-application) · [Architecture](#️-architecture) · [Results](#-results) · [Installation](#️-installation) · [Usage](#-usage)

</div>

---

## 📌 Overview

FractureAI is an end-to-end deep learning system that detects bone fractures across 7 musculoskeletal body parts using the Stanford MURA dataset. It combines a fine-tuned biomedical vision model with a local LLM for structured medical reporting — all running on consumer hardware with no API costs.

The project addresses a real clinical challenge: **shortcut learning**, where models learn to classify based on annotation markers (R/L letters placed by radiologists) rather than actual bone pathology. FractureAI solves this through a novel **marker exclusion mechanism** — detecting annotation letters via OpenCV and teaching the model to explicitly ignore those regions.

```
Upload X-Ray → Detect Markers → BiomedCLIP + LoRA → Specialist Head
     → Attention Rollout → Qwen3-VL Report → Treatment Guide
```

---

## ✨ Key Features

- 🎯 **Mean AUC 0.8669** across 7 body part specialists (+62% from baseline 0.53)
- 🔍 **Marker Exclusion** — detects and suppresses R/L annotation bias (96.2% detection rate)
- 🧠 **LoRA Fine-tuning** — only ~2% of parameters trained, fits 6GB VRAM
- 👁️ **Attention Rollout** — native ViT explainability (Grad-CAM fails on transformers)
- 🤖 **Qwen3-VL Local** — structured medical reports via Ollama, zero rate limits
- 📊 **Per-Body-Part Thresholds** — F1-optimized decision boundaries per anatomy
- 🏥 **Clinical UI** — dark medical theme with treatment guide and medical chatbot

---

## 🏆 Results

<div align="center">

| Body Part | AUC | Threshold |
|:---------:|:---:|:---------:|
| XR_ELBOW | 0.8876 | 0.400 |
| XR_FINGER | 0.8571 | 0.490 |
| XR_FOREARM | 0.8870 | 0.340 |
| XR_HAND | 0.8242 | 0.460 |
| XR_HUMERUS | 0.8783 | 0.370 |
| XR_SHOULDER | 0.8335 | 0.520 |
| XR_WRIST | **0.9007** | 0.340 |
| **MEAN** | **0.8669** | — |

</div>

### Performance Journey

```
Frozen Embeddings  →  Phase 1 LoRA  →  Phase 2 Heads  →  Phase 3 Thresholds
    AUC: 0.53            0.8555           0.8620              0.8669
                                                           (+62% total gain)
```

---

## 🏗️ Architecture

### Two-Stream Classifier (576-dim)

```
                    ┌─────────────────────────────────┐
                    │           Input X-Ray            │
                    └───────────────┬─────────────────┘
                                    │
              ┌─────────────────────┴──────────────────────┐
              │                                            │
    ┌─────────▼──────────┐                    ┌───────────▼────────────┐
    │   BiomedCLIP        │                    │   OpenCV Marker        │
    │   ViT-B/16 + LoRA   │                    │   Detection            │
    │   (r=32, α=64, n=8) │                    │   (96.2% accuracy)     │
    └─────────┬──────────┘                    └───────────┬────────────┘
              │                                            │
         512-dim                               8-dim exclude tensor
         image features                        [x1,y1,x2,y2,cx,cy,w,h]
              │                                            │
              │                              ┌────────────▼────────────┐
              │                              │   Exclude Encoder        │
              │                              │   8 → 32 → 64 dim        │
              │                              └────────────┬────────────┘
              │                                           │
    ┌──────────▼──────────────────────────────────────▼──┐
    │          Concatenation: 512 + 64 = 576-dim          │
    └───────────────────────────┬─────────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │   Body Part Head       │
                    │   576 → 128 → 2        │
                    │   (7 specialists)      │
                    └───────────┬───────────┘
                                │
                    ┌───────────▼───────────┐
                    │  Fracture Probability  │
                    │  + Optimal Threshold   │
                    └───────────────────────┘
```

### Why This Architecture Works

| Problem | Solution |
|---------|----------|
| Shortcut learning on R/L markers | Exclude encoder suppresses marker region |
| Class imbalance | Focal Loss (γ=2.0) + class weights |
| VRAM constraints (6GB) | LoRA — only 2% of parameters trained |
| Grad-CAM blank on ViT | Attention Rollout — native transformer explainability |
| Rate limits on cloud LLMs | Qwen3-VL via Ollama — fully local |

---

## 📁 Project Structure

```
fracture_detection_project/
│
├── data/
│   ├── train_labels.csv              ← original labels
│   ├── valid_labels.csv
│   ├── train_labels_clean.csv        ← after CleanAnnotations.py
│   ├── valid_labels_clean.csv
│   ├── marker_coords.csv             ← from FindMarkers.py (96.2% found)
│   └── clean/                        ← preprocessed images
│       ├── train/
│       └── val/
│
├── models/
│   ├── saved/
│   │   ├── phase1_best_model.pt      ← 757MB trained encoder
│   │   ├── head_v2_XR_ELBOW.pt      ← 298KB × 7 specialist heads
│   │   ├── head_v2_XR_FINGER.pt
│   │   ├── head_v2_XR_FOREARM.pt
│   │   ├── head_v2_XR_HAND.pt
│   │   ├── head_v2_XR_HUMERUS.pt
│   │   ├── head_v2_XR_SHOULDER.pt
│   │   ├── head_v2_XR_WRIST.pt
│   │   └── optimal_thresholds.json
│   │
│   ├── Phase1_LoRA_Exclude.py        ← trains encoder with marker exclusion
│   ├── Phase2_BodyPart.py            ← trains 7 specialist heads
│   ├── Phase3_ThresholdTuning.py     ← finds optimal threshold per body part
│   └── Phase7_Qwen.py               ← generates gallery with Qwen3-VL
│
├── preprocessing/
│   ├── FindMarkers.py                ← OpenCV marker detection
│   └── CleanAnnotations.py          ← corner crop + bone edge enhancement
│
├── results/
│   ├── gemma/                        ← Qwen-generated gallery reports
│   │   └── XR_*/
│   │       ├── fracture_report.png
│   │       └── no_fracture_report.png
│   └── phase3_v2_roc_curves.png
│
└── app.py                            ← FractureAI Streamlit application
```

---

## ⚙️ Installation

### Prerequisites
- Python 3.10+
- CUDA GPU recommended (6GB+ VRAM)
- [Ollama](https://ollama.ai) for local Qwen inference

### 1. Clone and install dependencies

```bash
git clone https://github.com/yourusername/fractureai.git
cd fractureai

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/Mac

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install open_clip_torch streamlit pandas scikit-learn
pip install opencv-python pillow matplotlib tqdm ollama
```

### 2. Download MURA Dataset

Get Stanford MURA v1.1 from [stanfordmlgroup.github.io](https://stanfordmlgroup.github.io/competitions/mura/)

```
MURA-v1.1/
├── train/
│   └── XR_ELBOW/, XR_FINGER/, ...
└── valid/
    └── XR_ELBOW/, XR_FINGER/, ...
```

### 3. Set up Qwen local model

```bash
# Install Ollama from https://ollama.ai
ollama pull qwen2.5vl:7b          # smaller, faster
# or
ollama pull qwen3-vl:235b-cloud   # larger, better quality
```

---

## 🚀 Usage

### Full Pipeline (run in order)

```bash
# Step 1 — Detect annotation markers in all images
python preprocessing/FindMarkers.py

# Step 2 — Clean images (remove markers, enhance bone edges)
python preprocessing/CleanAnnotations.py

# Step 3 — Train encoder with LoRA + marker exclusion
python models/Phase1_LoRA_Exclude.py

# Step 4 — Train 7 body part specialist heads
python models/Phase2_BodyPart.py

# Step 5 — Tune decision thresholds per body part
python models/Phase3_ThresholdTuning.py

# Step 6 — Generate gallery with Qwen3-VL
# Terminal 1:
ollama serve
# Terminal 2:
python models/Phase7_Qwen.py

# Step 7 — Launch the app
streamlit run app.py
```

### Quick Start (models already trained)

```bash
ollama serve          # Terminal 1 — keep open
streamlit run app.py  # Terminal 2
```

---

## 🖥️ Application

The Streamlit app has 3 tabs:

### Tab 1 — Live Analysis
Upload any musculoskeletal X-ray and receive:
- Binary fracture prediction with probability score
- Attention Rollout heatmap (where AI focused)
- Overlay visualization (α=0.3 for bone visibility)
- Qwen3-VL structured medical report:
  - Fracture type, location, severity
  - Clinical findings and recommendation
  - Marker bias detection flag
- Personalized treatment and recovery guide

### Tab 2 — Sample Gallery
Pre-computed Qwen3-VL reports on MURA validation samples.
One fracture + one normal case per body part (14 total).

### Tab 3 — Project Intel
- Training journey chart (AUC progression)
- Per-body-part AUC bar chart
- Pipeline architecture diagram
- Training configuration details

---

## 🔬 Technical Deep Dive

### The Shortcut Learning Problem

MURA X-rays contain radiologist annotation letters (R, L, RY) placed in image corners to indicate laterality. Early experiments revealed the model exploiting these as prediction shortcuts — the model was achieving reasonable accuracy by recognizing annotation patterns rather than actual bone pathology.

**Evidence:** XR_FINGER AUC was near-random (0.52) despite normal training curves. Attention maps confirmed focus on corner letters rather than bone structures.

### The Fix — Exclude Encoder

```python
# FindMarkers.py detects letters using OpenCV
# Returns normalized bounding box coordinates
exclude = [x1, y1, x2, y2, cx, cy, w, h]

# FractureClassifierExclude uses a learned encoder
# to interpret and suppress the marker region
excl_features = exclude_encoder(exclude)  # 8 → 64 dim
combined = concat([img_features, excl_features])  # 576 dim
```

The model learns: when exclude tensor is non-zero, down-weight that spatial region. Images without markers receive a zero tensor — processed normally.

**Result:** XR_FINGER improved from ~0.52 → 0.8571

### Attention Rollout

Standard Grad-CAM fails on Vision Transformers (no convolutional layers). Attention Rollout propagates attention weights through all transformer layers:

```python
R = eye(seq_len)
for attn in attention_layers:
    A = attn + eye          # residual connection
    A = A / A.sum(dim=-1)   # normalize rows
    R = A @ R               # propagate rollout
cls_attn = R[0, 1:]         # CLS token attention to patches
```

### Qwen Multi-Image Analysis

Each prediction sends 4 images to Qwen3-VL:

| Image | Purpose |
|-------|---------|
| Original X-ray | Raw scan context, marker visibility |
| Clean enhanced | Bone structure analysis (markers removed) |
| Attention heatmap | Where BiomedCLIP focused |
| Overlay (α=0.3) | Whether AI focused on marker vs bone |

Qwen returns `marker_bias_detected: true/false` — flagging when BiomedCLIP may have used a shortcut.

---

## 📊 Dataset

**Stanford MURA v1.1** — Musculoskeletal Radiographs

| Split | Images | Studies |
|-------|--------|---------|
| Train | 36,808 | 11,184 |
| Validation | 3,197 | 1,199 |
| **Total** | **40,005** | **12,383** |

7 body parts: Elbow, Finger, Forearm, Hand, Humerus, Shoulder, Wrist

---

## 🛠️ Training Configuration

### Phase 1 — LoRA Fine-tuning

```
Base Model    : BiomedCLIP ViT-B/16
LoRA Rank     : 32  |  Alpha   : 64  |  Scaling : 2.0
LoRA Blocks   : Last 8 of 12 transformer blocks
Exclude Dim   : 64 (8 → 32 → 64)
Batch Size    : 8
LR (LoRA)     : 1e-4  |  LR (Head) : 1e-3
Scheduler     : OneCycleLR + cosine annealing
Optimizer     : AdamW (weight decay 1e-4)
Loss          : Focal Loss (γ=2.0) + class weights
Augmentation  : HFlip, VFlip, Rotation±15°, ColorJitter, RandomErasing (p=0.5)
Epochs        : 20 (early stopping, patience=5)
Best AUC      : 0.8555
```

### Phase 2 — Body Part Heads

```
Architecture  : 576 → LayerNorm → Dropout → 128 → GELU → Dropout → 2
Training      : Frozen encoder, heads only
Per body part : Independent training
```

### Phase 3 — Threshold Tuning

```
Search range  : 0.10 to 0.90 (step 0.01)
Metric        : F1 Score maximization
```

---

## 🧩 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
open_clip_torch>=2.20.0
streamlit>=1.30.0
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
tqdm>=4.65.0
ollama>=0.1.0
```

---

## ⚠️ Medical Disclaimer

This system is intended for **research and educational purposes only**. All results must be verified by a licensed radiologist before any clinical decision is made. This system is **not FDA approved** and should not be used as a substitute for professional medical diagnosis.

---

## 🗺️ Roadmap

- [ ] Resume Phase 1 training (checkpoint exists — may improve beyond 0.8669)
- [ ] Reduce marker bias in attention maps (more training epochs needed)
- [ ] DICOM support for clinical X-ray formats
- [ ] Multi-fracture detection per image
- [ ] Confidence calibration (Platt scaling)
- [ ] FastAPI REST endpoint

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- **Stanford ML Group** — MURA dataset
- **Microsoft Research** — BiomedCLIP model
- **Hugging Face** — model hosting
- **Qwen Team (Alibaba)** — Qwen3-VL vision-language model
- **Ollama** — local LLM inference runtime

---

<div align="center">

**Built with PyTorch · BiomedCLIP · LoRA · Qwen3-VL · Streamlit**

*If this project helped you, consider giving it a ⭐*

</div>
