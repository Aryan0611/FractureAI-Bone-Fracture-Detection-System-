# ==============================================================
# app.py — FractureAI Clinical Intelligence System
# ==============================================================
# Run: streamlit run app.py
# Updated: Qwen3-VL local via Ollama
#          FractureClassifierExclude v2 (576-dim)
#          Marker exclusion support
#          alpha=0.3 overlay
# ==============================================================

import os, io, json, base64
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import open_clip
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import streamlit as st
from ollama import chat
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="FractureAI | Clinical Intelligence",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@300;400;500;600;700&family=Exo+2:wght@300;400;600;700&display=swap');
*, *::before, *::after { box-sizing: border-box; margin: 0; }
html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: block !important; visibility: visible !important; }
[data-testid="stSidebarCollapseButton"] { display: block !important; visibility: visible !important; }
button[kind="headerNoPadding"] { display: block !important; visibility: visible !important; }
section[data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }
.stApp {
    background: #020B14;
    background-image:
        linear-gradient(rgba(0,200,255,0.015) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,255,0.015) 1px, transparent 1px);
    background-size: 40px 40px;
    color: #C8E6F0;
}
[data-testid="stSidebar"] {
    background: #030E1A !important;
    border-right: 1px solid rgba(0,200,255,0.12) !important;
}
[data-testid="stSidebar"] > div { padding-top: 0 !important; }
.block-container { padding: 1rem 2rem 2rem 2rem !important; }
.scanline-overlay {
    position: fixed; top: 0; left: 0;
    width: 100%; height: 100%;
    background: repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.03) 2px,rgba(0,0,0,0.03) 4px);
    pointer-events: none; z-index: 9999;
}
.sys-header { border-bottom: 1px solid rgba(0,200,255,0.2); padding: 16px 0 20px 0; margin-bottom: 24px; position: relative; }
.sys-header::after { content: ''; position: absolute; bottom: -1px; left: 0; width: 200px; height: 2px; background: linear-gradient(90deg, #00C8FF, transparent); }
.sys-title { font-family: 'Share Tech Mono', monospace; font-size: 1.9rem; color: #00C8FF; letter-spacing: 4px; text-transform: uppercase; line-height: 1; }
.sys-subtitle { font-family: 'Rajdhani', sans-serif; font-size: 0.85rem; color: rgba(0,200,255,0.5); letter-spacing: 3px; text-transform: uppercase; margin-top: 4px; }
.sys-status { display: inline-flex; align-items: center; gap: 6px; background: rgba(0,200,255,0.06); border: 1px solid rgba(0,200,255,0.2); padding: 3px 10px; border-radius: 2px; font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: #00FF88; letter-spacing: 1px; margin-top: 10px; }
.status-dot { width: 6px; height: 6px; border-radius: 50%; background: #00FF88; animation: pulse-dot 2s ease-in-out infinite; }
@keyframes pulse-dot { 0%, 100% { opacity: 1; } 50% { opacity: 0.3; } }
.panel { background: rgba(0,20,40,0.7); border: 1px solid rgba(0,200,255,0.12); border-radius: 4px; padding: 20px; position: relative; overflow: hidden; }
.panel::before { content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%; background: linear-gradient(180deg, #00C8FF 0%, rgba(0,200,255,0.1) 100%); }
.panel-title { font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: rgba(0,200,255,0.6); letter-spacing: 3px; text-transform: uppercase; margin-bottom: 14px; padding-left: 8px; }
.result-main { background: rgba(0,15,30,0.9); border: 1px solid rgba(0,200,255,0.15); border-radius: 4px; padding: 24px 28px; position: relative; overflow: hidden; }
.result-fracture { border-color: rgba(255,60,60,0.4); }
.result-fracture::before { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 3px; background: linear-gradient(90deg, #FF3C3C, rgba(255,60,60,0.1)); }
.result-normal { border-color: rgba(0,255,136,0.4); }
.result-normal::before { content: ''; position: absolute; top: 0; left: 0; width: 100%; height: 3px; background: linear-gradient(90deg, #00FF88, rgba(0,255,136,0.1)); }
.result-label { font-family: 'Share Tech Mono', monospace; font-size: 2rem; letter-spacing: 3px; text-transform: uppercase; line-height: 1; }
.result-prob { font-family: 'Share Tech Mono', monospace; font-size: 3.5rem; line-height: 1; letter-spacing: -1px; }
.conf-track { height: 6px; background: rgba(0,200,255,0.08); border-radius: 0; margin: 10px 0; position: relative; overflow: visible; }
.conf-fill { height: 100%; border-radius: 0; position: relative; transition: width 1s cubic-bezier(0.23, 1, 0.32, 1); }
.conf-fill::after { content: ''; position: absolute; right: -1px; top: -3px; width: 2px; height: 12px; background: white; }
.metric-card { background: rgba(0,20,40,0.8); border: 1px solid rgba(0,200,255,0.1); border-radius: 3px; padding: 14px 16px; text-align: center; position: relative; }
.metric-card::after { content: ''; position: absolute; bottom: 0; left: 10%; right: 10%; height: 1px; background: linear-gradient(90deg, transparent, rgba(0,200,255,0.3), transparent); }
.metric-val { font-family: 'Share Tech Mono', monospace; font-size: 1.7rem; color: #00C8FF; line-height: 1; }
.metric-lbl { font-family: 'Rajdhani', sans-serif; font-size: 0.7rem; color: rgba(0,200,255,0.4); letter-spacing: 2px; text-transform: uppercase; margin-top: 5px; }
.qwen-panel { background: rgba(0,10,25,0.95); border: 1px solid rgba(0,200,255,0.15); border-radius: 4px; overflow: hidden; }
.qwen-header { background: rgba(0,200,255,0.06); border-bottom: 1px solid rgba(0,200,255,0.12); padding: 10px 16px; display: flex; justify-content: space-between; align-items: center; }
.qwen-header-title { font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; color: rgba(0,200,255,0.7); letter-spacing: 2px; text-transform: uppercase; }
.qwen-row { padding: 9px 16px; border-bottom: 1px solid rgba(0,200,255,0.06); display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
.qwen-row:last-child { border-bottom: none; }
.qwen-key { font-family: 'Rajdhani', sans-serif; font-size: 0.78rem; color: rgba(0,200,255,0.4); letter-spacing: 1.5px; text-transform: uppercase; white-space: nowrap; min-width: 130px; }
.qwen-val { font-family: 'Exo 2', sans-serif; font-size: 0.85rem; color: #C8E6F0; text-align: right; line-height: 1.4; }
.badge { display: inline-flex; align-items: center; gap: 5px; padding: 3px 10px; border-radius: 2px; font-family: 'Share Tech Mono', monospace; font-size: 0.68rem; letter-spacing: 1px; text-transform: uppercase; }
.badge-agree { background: rgba(0,255,136,0.08); border: 1px solid rgba(0,255,136,0.25); color: #00FF88; }
.badge-disagree { background: rgba(255,165,0,0.08); border: 1px solid rgba(255,165,0,0.25); color: #FFA500; }
.disclaimer-box { background: rgba(255,165,0,0.04); border: 1px solid rgba(255,165,0,0.2); border-left: 3px solid rgba(255,165,0,0.5); padding: 10px 14px; border-radius: 2px; margin-top: 14px; }
.disclaimer-box p { font-family: 'Rajdhani', sans-serif; font-size: 0.8rem; color: rgba(255,165,0,0.8); letter-spacing: 0.5px; margin: 0; }
[data-testid="stFileUploader"] { background: rgba(0,20,40,0.5) !important; border: 1px dashed rgba(0,200,255,0.2) !important; border-radius: 4px !important; }
.stButton > button { background: rgba(0,200,255,0.08) !important; border: 1px solid rgba(0,200,255,0.35) !important; color: #00C8FF !important; font-family: 'Share Tech Mono', monospace !important; letter-spacing: 2px !important; text-transform: uppercase !important; border-radius: 2px !important; transition: all 0.2s ease !important; }
.stButton > button:hover { background: rgba(0,200,255,0.15) !important; border-color: rgba(0,200,255,0.6) !important; box-shadow: 0 0 15px rgba(0,200,255,0.15) !important; }
[data-testid="stSelectbox"] > div > div { background: rgba(0,20,40,0.8) !important; border: 1px solid rgba(0,200,255,0.2) !important; border-radius: 2px !important; color: #C8E6F0 !important; }
[data-testid="stTabs"] [role="tablist"] { background: transparent !important; border-bottom: 1px solid rgba(0,200,255,0.12) !important; gap: 0 !important; }
[data-testid="stTabs"] button { font-family: 'Share Tech Mono', monospace !important; font-size: 0.72rem !important; letter-spacing: 2px !important; text-transform: uppercase !important; color: rgba(0,200,255,0.4) !important; background: transparent !important; border: none !important; border-bottom: 2px solid transparent !important; padding: 10px 20px !important; border-radius: 0 !important; }
[data-testid="stTabs"] button[aria-selected="true"] { color: #00C8FF !important; border-bottom: 2px solid #00C8FF !important; background: rgba(0,200,255,0.05) !important; }
.cyber-divider { height: 1px; background: linear-gradient(90deg, transparent, rgba(0,200,255,0.3), transparent); margin: 20px 0; }
.empty-state { text-align: center; padding: 80px 20px; border: 1px dashed rgba(0,200,255,0.1); border-radius: 4px; }
.empty-xray { font-size: 5rem; filter: opacity(0.4); display: block; }
.empty-text { font-family: 'Share Tech Mono', monospace; font-size: 0.85rem; color: rgba(0,200,255,0.3); letter-spacing: 2px; text-transform: uppercase; margin-top: 16px; }
.side-metric { background: rgba(0,20,40,0.6); border: 1px solid rgba(0,200,255,0.1); border-radius: 3px; padding: 10px 14px; margin-bottom: 8px; }
.side-metric-val { font-family: 'Share Tech Mono', monospace; font-size: 1.3rem; color: #00C8FF; }
.side-metric-lbl { font-family: 'Rajdhani', sans-serif; font-size: 0.7rem; color: rgba(0,200,255,0.35); letter-spacing: 2px; text-transform: uppercase; }
.pipeline-box { background: rgba(0,10,25,0.9); border: 1px solid rgba(0,200,255,0.1); border-radius: 4px; padding: 20px 24px; font-family: 'Share Tech Mono', monospace; font-size: 0.82rem; color: rgba(0,200,255,0.6); line-height: 2; }
.pipe-step { color: #00C8FF; display: block; padding: 4px 0; border-left: 2px solid rgba(0,200,255,0.2); padding-left: 12px; margin-left: 20px; }
.pipe-arrow { color: rgba(0,200,255,0.25); margin-left: 28px; display: block; font-size: 0.7rem; }
.tech-card { background: rgba(0,20,40,0.7); border: 1px solid rgba(0,200,255,0.1); border-radius: 4px; padding: 16px; text-align: center; }
.tech-icon { font-size: 1.8rem; display: block; }
.tech-name { font-family: 'Rajdhani', sans-serif; font-size: 0.95rem; font-weight: 600; color: #C8E6F0; margin-top: 8px; letter-spacing: 1px; }
.tech-desc { font-size: 0.72rem; color: rgba(0,200,255,0.4); letter-spacing: 1px; text-transform: uppercase; margin-top: 3px; }
hr { border-color: rgba(0,200,255,0.1) !important; }
.chat-panel { background: rgba(2,11,20,0.98); border: 1px solid rgba(0,200,255,0.2); border-radius: 8px 8px 0 0; overflow: hidden; }
.chat-header { background: rgba(0,200,255,0.08); border-bottom: 1px solid rgba(0,200,255,0.15); padding: 12px 16px; font-family: 'Share Tech Mono', monospace; font-size: 0.72rem; color: #00C8FF; letter-spacing: 2px; text-transform: uppercase; }
.chat-msg-user { background: rgba(0,200,255,0.08); border: 1px solid rgba(0,200,255,0.15); border-radius: 8px 8px 0 8px; padding: 8px 12px; margin: 6px 0 6px 20%; font-size: 0.85rem; color: #C8E6F0; }
.chat-msg-bot { background: rgba(0,15,30,0.9); border: 1px solid rgba(0,200,255,0.1); border-radius: 8px 8px 8px 0; padding: 8px 12px; margin: 6px 20% 6px 0; font-size: 0.85rem; color: #C8E6F0; line-height: 1.5; }
.chat-msg-label { font-family: 'Share Tech Mono', monospace; font-size: 0.6rem; color: rgba(0,200,255,0.4); letter-spacing: 1px; text-transform: uppercase; margin-bottom: 2px; }
.treatment-card { background: rgba(0,15,30,0.9); border: 1px solid rgba(0,200,255,0.12); border-radius: 4px; padding: 16px 20px; margin-bottom: 10px; position: relative; }
.treatment-card::before { content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%; background: linear-gradient(180deg, #00C8FF 0%, rgba(0,200,255,0.1) 100%); }
.treatment-icon { font-size: 1.4rem; margin-bottom: 6px; display: block; }
.treatment-title { font-family: 'Share Tech Mono', monospace; font-size: 0.7rem; color: #00C8FF; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 8px; }
.treatment-item { font-family: 'Exo 2', sans-serif; font-size: 0.82rem; color: rgba(200,230,240,0.8); padding: 3px 0; border-bottom: 1px solid rgba(0,200,255,0.05); line-height: 1.5; }
.treatment-item:last-child { border-bottom: none; }
.treatment-item::before { content: '\25B8 '; color: rgba(0,200,255,0.4); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="scanline-overlay"></div>', unsafe_allow_html=True)

# ==============================================================
# CONFIG
# ==============================================================
BASE_PATH    = r"d:\fracture_detection_project"
SAVE_DIR     = os.path.join(BASE_PATH, "models", "saved")
PHASE1_MODEL = os.path.join(SAVE_DIR, "phase1_best_model.pt")
THRESH_FILE  = os.path.join(SAVE_DIR, "optimal_thresholds.json")
MARKER_CSV   = os.path.join(BASE_PATH, "data", "marker_coords.csv")
GRADCAM_DIR  = os.path.join(BASE_PATH, "results", "gradcam")
GEMMA_DIR    = os.path.join(BASE_PATH, "results", "gemma")

# ── Qwen local via Ollama ─────────────────────────────────────
QWEN_MODEL    = "qwen3-vl:235b-cloud"
QWEN_FALLBACK = "qwen2.5vl:7b"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MARKER_PADDING = 0.03
CROP_MARGIN    = 0.12

BODY_PARTS = ["XR_ELBOW","XR_FINGER","XR_FOREARM","XR_HAND","XR_HUMERUS","XR_SHOULDER","XR_WRIST"]
BP_LABELS  = {"XR_ELBOW":"Elbow","XR_FINGER":"Finger","XR_FOREARM":"Forearm","XR_HAND":"Hand","XR_HUMERUS":"Humerus","XR_SHOULDER":"Shoulder","XR_WRIST":"Wrist"}

# ── Updated AUC values from Phase 3 v2 ───────────────────────
BP_STATS = {
    "XR_ELBOW"   :{"acc":0.85,"f1":0.84,"auc":0.8876,"recall":0.79},
    "XR_FINGER"  :{"acc":0.78,"f1":0.78,"auc":0.8571,"recall":0.82},
    "XR_FOREARM" :{"acc":0.84,"f1":0.81,"auc":0.8870,"recall":0.73},
    "XR_HAND"    :{"acc":0.77,"f1":0.72,"auc":0.8242,"recall":0.74},
    "XR_HUMERUS" :{"acc":0.82,"f1":0.81,"auc":0.8783,"recall":0.80},
    "XR_SHOULDER":{"acc":0.74,"f1":0.75,"auc":0.8335,"recall":0.81},
    "XR_WRIST"   :{"acc":0.83,"f1":0.80,"auc":0.9007,"recall":0.75},
}

# ==============================================================
# MARKER COORDS
# ==============================================================
def load_marker_coords() -> dict:
    if not os.path.exists(MARKER_CSV):
        return {}
    df      = pd.read_csv(MARKER_CSV)
    markers = {}
    for _, row in df.iterrows():
        if row["has_marker"] == 1:
            markers[row["image_path"]] = {
                "x1": max(0.0, float(row["x1"]) - MARKER_PADDING),
                "y1": max(0.0, float(row["y1"]) - MARKER_PADDING),
                "x2": min(1.0, float(row["x2"]) + MARKER_PADDING),
                "y2": min(1.0, float(row["y2"]) + MARKER_PADDING),
            }
    return markers


def get_exclude_tensor(image_path, markers):
    if image_path in markers:
        m  = markers[image_path]
        x1, y1, x2, y2 = m["x1"], m["y1"], m["x2"], m["y2"]
        return torch.tensor(
            [x1, y1, x2, y2,
             (x1+x2)/2, (y1+y2)/2,
             x2-x1, y2-y1],
            dtype=torch.float32)
    return torch.zeros(8, dtype=torch.float32)


# ==============================================================
# MODEL CLASSES — v2 (576-dim with exclude encoder)
# ==============================================================
class LoRALinear(nn.Module):
    def __init__(self, linear, r=32, alpha=64):
        super().__init__()
        self.linear  = linear
        self.scaling = alpha / r
        in_f  = linear.weight.shape[1]
        out_f = linear.weight.shape[0]
        self.lora_A = nn.Parameter(torch.randn(r, in_f)*0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x):
        return self.linear(x) + (x @ self.lora_A.T @ self.lora_B.T) * self.scaling


def find_blocks(model):
    for attr in ["visual.transformer.resblocks","visual.trunk.blocks","visual.model.blocks","visual.blocks"]:
        try:
            obj = model
            for a in attr.split("."): obj = getattr(obj, a)
            return obj
        except AttributeError: pass
    raise AttributeError("Cannot find transformer blocks")


def inject_lora_block(block, r, alpha):
    if hasattr(block, "attn"):
        a = block.attn
        if hasattr(a, "out_proj") and isinstance(a.out_proj, nn.Linear):
            a.out_proj = LoRALinear(a.out_proj, r, alpha)
        elif hasattr(a, "proj") and isinstance(a.proj, nn.Linear):
            a.proj = LoRALinear(a.proj, r, alpha)
    if hasattr(block, "mlp"):
        m = block.mlp
        for attr in ["c_fc","c_proj","fc1","fc2"]:
            if hasattr(m, attr) and isinstance(getattr(m, attr), nn.Linear):
                setattr(m, attr, LoRALinear(getattr(m, attr), r, alpha))


def inject_lora(model, r=32, alpha=64, n=8):
    for p in model.parameters(): p.requires_grad_(False)
    blocks = find_blocks(model)
    total  = len(blocks)
    for i, b in enumerate(blocks):
        if i >= total - n: inject_lora_block(b, r, alpha)
    return model


class FractureClassifierExclude(nn.Module):
    """v2 model — 576-dim (512 image + 64 exclude)"""
    def __init__(self, clip_model, embed_dim=512,
                 exclude_dim=64, num_classes=2, dropout=0.3):
        super().__init__()
        self.clip_model = clip_model
        self.exclude_encoder = nn.Sequential(
            nn.Linear(8, 32), nn.GELU(),
            nn.LayerNorm(32), nn.Dropout(0.1),
            nn.Linear(32, exclude_dim), nn.GELU())
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim + exclude_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim + exclude_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes))

    def forward(self, x, exclude):
        img_f  = self.clip_model.encode_image(x).float()
        excl_f = self.exclude_encoder(exclude.float())
        return self.head(torch.cat([img_f, excl_f], 1))

    def encode(self, x, exclude):
        img_f  = self.clip_model.encode_image(x).float()
        excl_f = self.exclude_encoder(exclude.float())
        return torch.cat([img_f, excl_f], 1)


class BodyPartHead(nn.Module):
    """v2 head — 576-dim input"""
    def __init__(self, embed_dim=576, nc=2, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Dropout(dropout),
            nn.Linear(embed_dim, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, nc))

    def forward(self, x): return self.head(x)


# ==============================================================
# ATTENTION ROLLOUT
# ==============================================================
class AttentionRollout:
    def __init__(self, clip_model):
        self.clip_model = clip_model

    def generate(self, img_tensor):
        visual = self.clip_model.visual
        img_tensor = img_tensor.to(DEVICE)
        try: blocks = visual.trunk.blocks
        except:
            try: blocks = visual.transformer.resblocks
            except: blocks = visual.blocks

        attn_list, hooks = [], []

        def make_hook():
            def hook(module, inp, out):
                if isinstance(inp, tuple) and len(inp) > 0:
                    x = inp[0]
                    B, N, C = x.shape
                    if hasattr(module, 'qkv'):
                        nh  = getattr(module, 'num_heads', 12)
                        qkv = module.qkv(x).reshape(B, N, 3, nh, -1).permute(2, 0, 3, 1, 4)
                        q, k, v = qkv.unbind(0)
                        attn = (q @ k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
                        attn = attn.softmax(-1)
                        attn_list.append(attn.mean(1).detach().cpu())
            return hook

        for b in blocks:
            if hasattr(b, 'attn'):
                hooks.append(b.attn.register_forward_hook(make_hook()))

        with torch.no_grad(): visual(img_tensor)
        for h in hooks: h.remove()

        if not attn_list: return np.ones((224, 224)) * 0.3

        sl = attn_list[0].shape[-1]
        R  = torch.eye(sl)
        for a in attn_list:
            a = a[0] + torch.eye(sl)
            a = a / a.sum(-1, keepdim=True)
            R = torch.mm(a, R)

        ca = R[0, 1:]; n = ca.shape[0]; g = int(n**0.5)
        m  = ca[:g*g].reshape(g, g).numpy()
        m -= m.min()
        if m.max() > 0: m /= m.max()
        return cv2.resize(m, (224, 224), interpolation=cv2.INTER_LINEAR)


# ==============================================================
# LOAD MODELS
# ==============================================================
@st.cache_resource(show_spinner=False)
def load_models():
    with open(THRESH_FILE) as f: thresholds = json.load(f)

    # Load marker coords for exclude tensors
    markers = load_marker_coords()

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

    # v2 LoRA settings — must match Phase 1
    clip_model = inject_lora(clip_model, r=32, alpha=64, n=8)

    enc = FractureClassifierExclude(clip_model).to(DEVICE)
    enc.load_state_dict(torch.load(PHASE1_MODEL, map_location=DEVICE))
    enc.eval()

    # Load v2 heads (576-dim) — try v2 first, fallback to old
    heads = {}
    for bp in BODY_PARTS:
        for fn in [f"head_v2_{bp}.pt", f"head_{bp}.pt"]:
            p = os.path.join(SAVE_DIR, fn)
            if os.path.exists(p):
                try:
                    h = BodyPartHead(576).to(DEVICE)
                    h.load_state_dict(torch.load(p, map_location=DEVICE))
                    h.eval()
                    heads[bp] = h
                    break
                except Exception:
                    continue

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073),
            (0.26862954, 0.26130258, 0.27577711))])

    return enc, heads, thresholds, tf, AttentionRollout(enc.clip_model), markers


# ==============================================================
# HELPERS
# ==============================================================
def predict(enc, head, tf, img, markers, img_path, threshold):
    """Predict with exclude tensor from marker coords."""
    t       = tf(img).unsqueeze(0).to(DEVICE)
    exclude = get_exclude_tensor(
        img_path, markers).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats = enc.encode(t, exclude)
        p     = torch.softmax(head(feats), 1)
        return float(p[0, 1].detach().cpu())


def gen_heatmap(rollout, tf, img):
    return rollout.generate(tf(img).unsqueeze(0))


def overlay_heatmap(img, hmap, alpha=0.3):
    """alpha=0.3 — lighter overlay, bone more visible."""
    r = img.resize((224, 224)).convert("RGB")
    n = np.array(r)
    h = (cm.jet(hmap)[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray((n * (1 - alpha) + h * alpha).astype(np.uint8))


def get_clean_image(img_source) -> Image.Image:
    """Accepts file path (str) or PIL Image directly."""
    try:
        if isinstance(img_source, str):
            if not img_source or not os.path.exists(img_source):
                return img_source if isinstance(img_source, Image.Image) else Image.new("RGB", (224,224))
            img = cv2.imread(img_source, cv2.IMREAD_GRAYSCALE)
        else:
            # PIL Image passed directly
            img = cv2.cvtColor(np.array(img_source), cv2.COLOR_RGB2GRAY)
        if img is None:
            return Image.new("RGB", (224,224))
        h, w    = img.shape
        mh      = max(int(h * CROP_MARGIN), 5)
        mw      = max(int(w * CROP_MARGIN), 5)
        cropped = img[mh:h-mh, mw:w-mw]
        cropped = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enh     = clahe.apply(cropped)
        ot, _   = cv2.threshold(enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bt      = max(20, int(ot * 0.3))
        _, bm   = cv2.threshold(enh, bt, 255, cv2.THRESH_BINARY)
        bm      = cv2.morphologyEx(bm, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        bl      = cv2.GaussianBlur(enh, (3,3), 0)
        ed      = cv2.Canny(bl, 15, 50)
        em      = cv2.bitwise_and(ed, ed, mask=bm)
        ed2     = cv2.dilate(em, np.ones((2,2), np.uint8), iterations=1)
        result  = cv2.addWeighted(enh, 0.85, ed2, 0.15, 0)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_GRAY2RGB))
    except:
        return img_source if isinstance(img_source, Image.Image) else Image.new("RGB", (224,224))


def pil_b64(img):
    b = io.BytesIO()
    img.resize((224, 224)).save(b, format="PNG")
    return base64.b64encode(b.getvalue()).decode()


# ==============================================================
# QWEN LOCAL CALL
# ==============================================================
def call_qwen(img, hmap, bp, prob, threshold,
              img_path="", model=None):
    if model is None:
        model = QWEN_MODEL

    clean_img   = get_clean_image(img)
    """
    Sends 4 images to Qwen local via Ollama:
    1. Original X-ray
    2. Clean enhanced (markers removed)
    3. Attention heatmap
    4. Overlay (heatmap on original)
    """
    if model is None:
        model = QWEN_MODEL

    clean_img   = get_clean_image("")  # fallback
    heat_pil    = Image.fromarray(
        (cm.jet(hmap)[:,:,:3]*255).astype(np.uint8))
    overlay_pil = overlay_heatmap(img, hmap, alpha=0.3)

    bpn  = bp.replace('XR_', '')
    pred = "FRACTURE DETECTED" if prob >= threshold \
           else "NO FRACTURE"

    prompt = f"""You are an expert radiologist AI.
Body part: {bpn}
BiomedCLIP: {pred} ({prob*100:.1f}%)
Threshold : {threshold:.2f}

4 images provided:
1. Original X-ray (raw, may have R/L markers)
2. Clean enhanced (markers removed, bone edges bright)
3. Attention heatmap (red = AI focus)
4. Overlay (heatmap on original — shows if AI focused on marker or bone)

Use image 2 for bone analysis.
Use image 4 to judge attention_quality:
  bone-focused   = red on bone/cortex
  marker-focused = red on R/L text
  mixed          = both

Return ONLY valid JSON:
{{"fracture_detected":true,"confidence":0.0,
"status":"confirmed or inconclusive or normal",
"fracture_type":"desc or null","location":"location or null",
"severity":"mild or null",
"attention_quality":"bone-focused or marker-focused or mixed",
"marker_bias_detected":false,
"clinical_findings":"one sentence",
"recommendation":"one sentence",
"treatment_plan":"brief plan",
"disclaimer":"AI-generated. Radiologist confirmation required."}}"""

    try:
        r = chat(
            model=model,
            messages=[{
                "role"   : "user",
                "content": prompt,
                "images" : [
                    pil_b64(img),        # 1. original
                    pil_b64(clean_img),  # 2. clean
                    pil_b64(heat_pil),   # 3. heatmap
                    pil_b64(overlay_pil) # 4. overlay
                ]
            }]
        )
        raw = r.message.content.strip()
        import re
        raw = re.sub(r'<think>.*?</think>', '',
                     raw, flags=re.DOTALL).strip()
        for p in ["```json", "```"]:
            if raw.startswith(p): raw = raw[len(p):]
        if raw.endswith("```"): raw = raw[:-3]
        result = json.loads(raw.strip())
        if not result.get("fracture_detected", False):
            result.update({
                "fracture_type": None,
                "location"     : None,
                "severity"     : None})
        result["confidence"] = max(0.0, min(1.0,
            float(result.get("confidence", 0))))
        return result

    except Exception as e:
        # Try fallback
        if model != QWEN_FALLBACK:
            return call_qwen(img, hmap, bp, prob,
                             threshold, QWEN_FALLBACK)
        return None


def call_chatbot(question, context, history):
    ctx_str = ""
    if context:
        bp   = context.get("body_part","?").replace("XR_","")
        pred = context.get("prediction", False)
        prob = context.get("prob", 0)
        ctx_str = (
            f"PATIENT CONTEXT:\n"
            f"- Body part: {bp}\n"
            f"- Fracture: {'YES' if pred else 'NO'}\n"
            f"- Probability: {prob*100:.1f}%\n"
            f"- Findings: {context.get('qwen_findings','N/A')}\n")

    system_msg = (
        "You are a helpful medical AI assistant "
        "specializing in bone fractures.\n" + ctx_str +
        "Answer in 3-5 sentences. "
        "Always remind users to consult a real doctor.")

    messages = [{"role": "system", "content": system_msg}]
    for msg in history[-6:]:
        messages.append({"role": msg["role"],
                         "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    try:
        r = chat(model=QWEN_MODEL, messages=messages)
        import re
        raw = r.message.content.strip()
        raw = re.sub(r'<think>.*?</think>', '',
                     raw, flags=re.DOTALL).strip()
        return raw
    except Exception as e:
        try:
            r = chat(model=QWEN_FALLBACK, messages=messages)
            return r.message.content.strip()
        except:
            return f"Qwen unavailable: {str(e)[:60]}"


# ==============================================================
# TREATMENT DATA
# ==============================================================
TREATMENT_DATA = {
    "fracture": {
        "first_aid": [
            "Immobilize the area — do NOT move or realign the bone",
            "Apply ice wrapped in cloth for 15-20 mins to reduce swelling",
            "Elevate the injured limb above heart level if possible",
            "Call emergency services or visit ER immediately",
            "Take prescribed pain relief — avoid ibuprofen in first 24h",
        ],
        "foods": [
            "Milk, yogurt, cheese — calcium for bone repair",
            "Salmon, tuna, sardines — Vitamin D and Omega-3",
            "Broccoli, kale, spinach — calcium and Vitamin K",
            "Eggs — Vitamin D and protein for healing",
            "Citrus fruits — Vitamin C for collagen synthesis",
            "Almonds, seeds — magnesium for bone strength",
            "AVOID: alcohol, caffeine excess, smoking — all slow healing",
        ],
        "recovery": [
            "Week 1-2: Complete rest, ice therapy, pain management",
            "Week 2-6: Immobilization (cast/splint), gentle finger movement",
            "Week 6-12: Physical therapy begins, strength rebuilding",
            "Month 3-6: Progressive weight bearing, full ROM exercises",
            "Month 6+: Return to full activity with doctor clearance",
        ],
        "avoid": [
            "Weight bearing on injured limb without doctor clearance",
            "High-impact activities: running, jumping, contact sports",
            "Smoking — reduces bone healing speed significantly",
            "Anti-inflammatory drugs (ibuprofen) in first 72 hours",
            "Ignoring pain signals — pain is your body talking",
        ],
        "see_doctor": [
            "URGENT: Bone visibly deformed or skin broken",
            "URGENT: Numbness, tingling or loss of sensation",
            "Severe swelling that worsens after 48 hours",
            "Fever above 38 degrees C — may indicate infection",
            "Cast becomes too tight, wet, or causes skin irritation",
        ],
        "pain": [
            "Ice therapy: 20 mins on, 20 mins off in first 48 hours",
            "Elevation above heart level reduces swelling and pain",
            "Paracetamol (acetaminophen) for mild-moderate pain",
            "Prescribed opioids only as directed by doctor",
            "Adequate sleep — most bone healing occurs during sleep",
        ],
    },
    "normal": {
        "first_aid": [
            "No fracture detected — monitor for any worsening symptoms",
            "If pain persists beyond 48 hours, seek medical evaluation",
            "Apply ice if there is any swelling or bruising",
            "Rest the area and avoid strenuous activity for 24-48 hours",
        ],
        "foods": [
            "Maintain calcium intake: 1000-1200mg daily for bone health",
            "Vitamin D: sunlight exposure 15 mins daily or supplements",
            "Adequate protein for bone matrix maintenance",
            "Antioxidant-rich foods: berries, leafy greens, nuts",
        ],
        "recovery": [
            "No fracture recovery needed",
            "Continue regular low-impact exercise for bone health",
            "Consider annual bone density check if over 50",
            "Maintain healthy weight to reduce joint stress",
        ],
        "avoid": [
            "High-impact sports without proper warmup and technique",
            "Excessive caffeine and alcohol — deplete calcium",
            "Prolonged inactivity — weakens bones over time",
        ],
        "see_doctor": [
            "If pain persists beyond 1 week despite rest",
            "Annual check-up if you have osteoporosis risk factors",
            "Any sudden onset joint pain or swelling",
        ],
        "pain": [
            "OTC pain relief: Paracetamol or Ibuprofen for soreness",
            "Gentle stretching and range-of-motion exercises",
            "Warm compress for chronic muscle tension around joint",
        ],
    }
}


def render_treatment_guide(body_part, prediction, prob):
    key     = "fracture" if prediction else "normal"
    data    = TREATMENT_DATA[key]
    bp_name = body_part.replace("XR_", "")
    color   = "#FF3C3C" if prediction else "#00FF88"
    label   = "FRACTURE" if prediction else "NO FRACTURE"
    st.markdown(
        f'''<div style="font-family:'Share Tech Mono',monospace;
        font-size:0.65rem;color:rgba(0,200,255,0.4);
        letter-spacing:3px;text-transform:uppercase;margin-bottom:16px">
        <span style="color:{color}">{label}</span> —
        {bp_name.upper()} TREATMENT AND RECOVERY PROTOCOL
        </div>''', unsafe_allow_html=True)
    sections = [
        ("🚑","FIRST AID",data["first_aid"],"#FF3C3C"),
        ("🥗","BONE HEALING NUTRITION",data["foods"],"#00C8FF"),
        ("📅","RECOVERY TIMELINE",data["recovery"],"#FFD700"),
        ("🚫","WHAT TO AVOID",data["avoid"],"#FF8C00"),
        ("🏥","SEE DOCTOR WHEN",data["see_doctor"],"#FF3C3C"),
        ("💊","PAIN MANAGEMENT",data["pain"],"#A78BFA"),
    ]
    for i in range(0, len(sections), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j >= len(sections): break
            icon, title, items, col_color = sections[i+j]
            items_html = "".join(
                f'<div class="treatment-item">{it}</div>'
                for it in items)
            with col:
                st.markdown(
                    f'''<div class="treatment-card">
                    <span class="treatment-icon">{icon}</span>
                    <div class="treatment-title"
                         style="color:{col_color}">{title}</div>
                    {items_html}</div>''',
                    unsafe_allow_html=True)


def call_chatbot_render(context):
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    st.markdown('<div class="cyber-divider"></div>',
                unsafe_allow_html=True)
    with st.expander(
            "💬  MEDICAL CHATBOT — Ask Qwen Anything",
            expanded=False):
        st.markdown(
            '''<div class="chat-header">
            🤖 FRACTUREAI MEDICAL ASSISTANT &nbsp;·&nbsp;
            <span style="color:rgba(0,200,255,0.4);font-size:0.6rem">
            QWEN · CONTEXT-AWARE</span></div>''',
            unsafe_allow_html=True)
        if context:
            bp   = context.get("body_part","?").replace("XR_","")
            pred = context.get("prediction", False)
            prob = context.get("prob", 0)
            st.markdown(
                f'''<div style="background:rgba(0,200,255,0.04);
                border:1px solid rgba(0,200,255,0.1);padding:8px 12px;
                border-radius:3px;font-family:'Share Tech Mono',monospace;
                font-size:0.65rem;color:rgba(0,200,255,0.5);
                letter-spacing:1px;margin:8px 0">
                CONTEXT: {bp.upper()} ·
                {"FRACTURE" if pred else "NO FRACTURE"} ·
                PROB {prob*100:.1f}%</div>''',
                unsafe_allow_html=True)
        quick_qs = [
            "How long to heal?","What foods help bone healing?",
            "When can I exercise?","What painkillers are safe?",
            "Signs of complications?","Recovery exercises?",
        ]
        st.markdown(
            '<div style="font-family:monospace;font-size:0.6rem;'
            'color:rgba(0,200,255,0.35);letter-spacing:1px;'
            'margin:8px 0 4px 0">QUICK QUESTIONS:</div>',
            unsafe_allow_html=True)
        qcols = st.columns(3)
        for i, q in enumerate(quick_qs):
            with qcols[i % 3]:
                if st.button(q, key=f"qq_{i}",
                             use_container_width=True):
                    st.session_state["chat_history"].append(
                        {"role":"user","content":q})
                    with st.spinner("QUERYING QWEN..."):
                        ans = call_chatbot(
                            q, context,
                            st.session_state["chat_history"])
                    st.session_state["chat_history"].append(
                        {"role":"assistant","content":ans})
                    st.rerun()
        history = st.session_state.get("chat_history", [])
        if history:
            for msg in history:
                if msg["role"] == "user":
                    st.markdown(
                        f'''<div class="chat-msg-label">YOU</div>
                        <div class="chat-msg-user">{msg["content"]}</div>''',
                        unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'''<div class="chat-msg-label">FRACTUREAI</div>
                        <div class="chat-msg-bot">{msg["content"]}</div>''',
                        unsafe_allow_html=True)
        else:
            st.markdown(
                '''<div style="text-align:center;padding:16px;
                font-family:'Share Tech Mono',monospace;font-size:0.75rem;
                color:rgba(0,200,255,0.25);letter-spacing:1px">
                ASK ANYTHING ABOUT FRACTURES, RECOVERY,<br>
                NUTRITION OR YOUR SCAN RESULTS</div>''',
                unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns([5, 1])
        with c1:
            user_input = st.text_input(
                "Question", key="chat_input",
                placeholder="e.g. How long will recovery take?",
                label_visibility="collapsed")
        with c2:
            send_btn = st.button("SEND", key="chat_send",
                                 use_container_width=True)
        if send_btn and user_input and user_input.strip():
            st.session_state["chat_history"].append(
                {"role":"user","content":user_input})
            with st.spinner("QUERYING QWEN..."):
                answer = call_chatbot(
                    user_input, context,
                    st.session_state["chat_history"])
            st.session_state["chat_history"].append(
                {"role":"assistant","content":answer})
            st.rerun()
        if history:
            if st.button("CLEAR CHAT", key="clear_chat",
                         use_container_width=True):
                st.session_state["chat_history"] = []
                st.rerun()


# ==============================================================
# CHARTS
# ==============================================================
def journey_chart():
    fig, ax = plt.subplots(figsize=(11, 4))
    fig.patch.set_facecolor('#020B14')
    ax.set_facecolor('#030E1A')
    phases = ["Frozen\nEmbeddings","Phase 1\nLoRA","Phase 2\nBody Parts","Phase 3\nThresholds"]
    auc    = [0.53, 0.8555, 0.8620, 0.8669]
    acc    = [0.53, 0.7945, 0.7991, 0.7991]
    rec    = [0.50, 0.6800, 0.7100, 0.7744]
    x = np.arange(len(phases)); w = 0.25
    for vals, col, offset, lbl in [
        (auc,"#00C8FF",-w,"ROC-AUC"),
        (acc,"#00FF88", 0,"Accuracy"),
        (rec,"#FF4466",+w,"Recall"),
    ]:
        bars = ax.bar(x+offset, vals, w, color=col,
                      alpha=0.75, label=lbl, edgecolor='none')
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x()+bar.get_width()/2, h+0.01,
                    f'{h:.2f}', ha='center', va='bottom',
                    fontsize=7.5, color='white',
                    fontfamily='monospace')
    ax.plot([-w,3-w],[auc[0],auc[-1]], color='#FFD700',
            lw=1.5, linestyle='--', alpha=0.6,
            label=f'+{(auc[-1]-auc[0])*100:.0f}% AUC')
    ax.set_xticks(x)
    ax.set_xticklabels(phases, fontsize=8.5,
                       fontfamily='monospace')
    for t in ax.get_xticklabels(): t.set_color('#00C8FF80')
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score', color='#00C8FF66',
                  fontsize=9, fontfamily='monospace')
    ax.tick_params(colors='#00C8FF4D', labelsize=8)
    for spine in ax.spines.values():
        spine.set_color('#00C8FF1A')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(facecolor='#030E1A', labelcolor='white',
              fontsize=8, framealpha=0.8,
              edgecolor='#00C8FF33')
    ax.set_title('Model Performance Across Training Phases',
                 color='#00C8FF', fontsize=11,
                 fontfamily='monospace', pad=12)
    ax.yaxis.grid(True, color='#00C8FF0D',
                  linestyle='--')
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig


def auc_chart():
    fig, ax = plt.subplots(figsize=(11, 3.5))
    fig.patch.set_facecolor('#020B14')
    ax.set_facecolor('#030E1A')
    bps  = [bp.replace('XR_','') for bp in BODY_PARTS]
    aucs = [BP_STATS[bp]['auc'] for bp in BODY_PARTS]
    cols = ['#00FF88' if a>=0.88 else
            '#00C8FF' if a>=0.83 else
            '#FF4466' for a in aucs]
    bars = ax.bar(bps, aucs, color=cols,
                  alpha=0.75, edgecolor='none', width=0.55)
    for bar, a in zip(bars, aucs):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.004, f'{a:.3f}',
                ha='center', va='bottom', color='white',
                fontsize=8.5, fontfamily='monospace')
    ax.axhline(0.85, color='#FFD70066', ls='--',
               lw=1, label='0.85 target')
    ax.set_ylim(0.70, 1.02)
    ax.set_ylabel('ROC-AUC', color='#00C8FF66',
                  fontsize=9, fontfamily='monospace')
    ax.tick_params(colors='#00C8FF66', labelsize=9)
    ax.set_xticklabels(bps, fontfamily='monospace')
    for t in ax.get_xticklabels(): t.set_color('#00C8FF66')
    for spine in ax.spines.values():
        spine.set_color('#00C8FF1A')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, color='#00C8FF0D',
                  linestyle='--')
    ax.set_axisbelow(True)
    ax.legend(facecolor='#030E1A', labelcolor='white',
              fontsize=8, edgecolor='#00C8FF33')
    ax.set_title('ROC-AUC by Body Part Specialist',
                 color='#00C8FF', fontsize=11,
                 fontfamily='monospace', pad=10)
    plt.tight_layout()
    return fig


# ==============================================================
# MAIN
# ==============================================================
def main():
    c_title, c_status = st.columns([3, 1])
    with c_title:
        st.markdown("""
        <div class="sys-header">
            <div class="sys-title">🩻 FractureAI</div>
            <div class="sys-subtitle">Clinical Bone Fracture Intelligence System</div>
            <div class="sys-status">
                <div class="status-dot"></div>
                SYSTEM ONLINE · BIOMEDCLIP + LORA-32 ACTIVE
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c_status:
        st.markdown(f"""
        <div style="text-align:right;padding-top:8px">
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.35);letter-spacing:2px;text-transform:uppercase;margin-bottom:4px">COMPUTE UNIT</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.95rem;color:#00C8FF">{DEVICE.upper()}</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.35);letter-spacing:2px;margin-top:8px;text-transform:uppercase">MODEL ARCHITECTURE</div>
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:rgba(0,200,255,0.6)">ViT-B/16 + LoRA-32</div>
        </div>
        """, unsafe_allow_html=True)

    with st.spinner("INITIALIZING NEURAL SUBSYSTEMS..."):
        enc, heads, thresholds, tf, rollout, markers = load_models()

    with st.sidebar:
        st.markdown("""
        <div style="padding:16px 0 8px 0;border-bottom:1px solid rgba(0,200,255,0.1);margin-bottom:16px">
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.4);letter-spacing:3px;text-transform:uppercase">INPUT PARAMETERS</div>
        </div>
        """, unsafe_allow_html=True)
        uploaded    = st.file_uploader("UPLOAD X-RAY",
                         type=["png","jpg","jpeg"],
                         label_visibility="visible")
        body_part   = st.selectbox("BODY PART", BODY_PARTS,
                         format_func=lambda x: BP_LABELS[x],
                         label_visibility="visible")
        use_qwen    = st.toggle("QWEN MEDICAL REPORT",
                         value=True)
        analyze_btn = st.button("🔬 ANALYZE",
                         use_container_width=True,
                         key="analyze_btn", type="primary")

        st.markdown("""
        <div style="margin-top:20px;font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.4);letter-spacing:3px;text-transform:uppercase;border-bottom:1px solid rgba(0,200,255,0.1);padding-bottom:8px;margin-bottom:12px">SPECIALIST METRICS</div>
        """, unsafe_allow_html=True)
        stats = BP_STATS[body_part]
        for label, val, fmt in [
            ("ROC-AUC",  stats["auc"],    "{:.4f}"),
            ("Accuracy", stats["acc"],    "{:.2%}"),
            ("F1 Score", stats["f1"],     "{:.4f}"),
            ("Recall",   stats["recall"], "{:.2%}"),
        ]:
            st.markdown(f"""
            <div class="side-metric">
                <div class="side-metric-val">{fmt.format(val)}</div>
                <div class="side-metric-lbl">{label}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="margin-top:20px;padding-top:12px;border-top:1px solid rgba(0,200,255,0.1)">
            <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:rgba(0,200,255,0.3);letter-spacing:2px;line-height:1.8">
                DATASET: MURA v1.1<br>TRAIN SET: 36,808 imgs<br>VAL SET: 3,197 imgs<br>BODY PARTS: 7<br>BASE MODEL: ViT-B/16<br>LORA RANK: 32<br>LORA BLOCKS: 8/12<br>MEAN AUC: 0.8669
            </div>
        </div>
        """, unsafe_allow_html=True)

    diag_ctx = st.session_state.get("diagnosis_context", {})
    tab1, tab2, tab3 = st.tabs([
        "⬡  LIVE ANALYSIS",
        "⬡  SAMPLE GALLERY",
        "⬡  PROJECT INTEL"])

    # ── TAB 1 ────────────────────────────────────────────────
    with tab1:
        if uploaded is None:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-xray">🩻</span>
                <p class="empty-text">AWAITING X-RAY INPUT</p>
                <p style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:rgba(0,200,255,0.2);margin-top:8px;letter-spacing:1px">
                    UPLOAD IMAGE → SELECT BODY PART → ANALYZE
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Save uploaded file temporarily for marker lookup
            tmp_path = os.path.join(
                BASE_PATH, "temp_upload.png")
            with open(tmp_path, "wb") as f:
                f.write(uploaded.read())
            img       = Image.open(tmp_path).convert("RGB")
            threshold = thresholds.get(body_part, 0.5)

            if analyze_btn or st.session_state.get("ran_once"):
                st.session_state["ran_once"] = True

                with st.spinner("RUNNING NEURAL INFERENCE..."):
                    head = heads.get(body_part)
                    if not head:
                        st.error(f"No specialist for {body_part}")
                        return
                    prob       = predict(enc, head, tf, img,
                                         markers, tmp_path,
                                         threshold)
                    prediction = prob >= threshold
                    hmap       = gen_heatmap(rollout, tf, img)
                    overlay    = overlay_heatmap(img, hmap)

                pred_text  = "FRACTURE DETECTED" if prediction else "NO FRACTURE"
                color      = "#FF3C3C" if prediction else "#00FF88"
                card_class = "result-fracture" if prediction else "result-normal"
                icon       = "⚠" if prediction else "✓"
                bar_color  = "#FF3C3C" if prediction else "#00FF88"
                pct        = int(prob * 100)
                risk_color = "#FF3C3C" if prob > 0.7 else "#FFA500" if prob > 0.4 else "#00FF88"

                st.markdown(f"""
                <div class="result-main {card_class}">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.4);letter-spacing:3px;text-transform:uppercase;margin-bottom:6px">DIAGNOSTIC RESULT</div>
                            <div class="result-label" style="color:{color}">{icon} {pred_text}</div>
                            <div style="font-family:'Share Tech Mono',monospace;font-size:0.72rem;color:rgba(0,200,255,0.4);margin-top:6px;letter-spacing:1px">
                                {BP_LABELS[body_part].upper()} &nbsp;·&nbsp; THRESHOLD: {threshold:.2f} &nbsp;·&nbsp; AUC: {BP_STATS[body_part]['auc']:.3f}
                            </div>
                        </div>
                        <div style="text-align:right">
                            <div style="font-family:'Share Tech Mono',monospace;font-size:0.62rem;color:rgba(0,200,255,0.4);letter-spacing:2px;text-transform:uppercase;margin-bottom:2px">FRACTURE PROB</div>
                            <div class="result-prob" style="color:{risk_color}">{prob*100:.1f}%</div>
                        </div>
                    </div>
                    <div class="conf-track" style="margin-top:14px">
                        <div class="conf-fill" style="width:{pct}%;background:linear-gradient(90deg,{bar_color}aa,{bar_color})"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
                st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.4);letter-spacing:3px;text-transform:uppercase;margin-bottom:12px">VISUAL ANALYSIS</div>""", unsafe_allow_html=True)

                c1, c2, c3 = st.columns(3)
                heat_pil = Image.fromarray(
                    (cm.jet(hmap)[:,:,:3]*255
                     ).astype(np.uint8))
                with c1: st.image(img.resize((224,224)),
                                  caption="RAW INPUT",
                                  use_container_width=True)
                with c2: st.image(heat_pil,
                                  caption="ATTENTION ROLLOUT",
                                  use_container_width=True)
                with c3: st.image(overlay,
                                  caption="OVERLAY FUSION (α=0.3)",
                                  use_container_width=True)

                if use_qwen:
                    st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
                    st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.4);letter-spacing:3px;text-transform:uppercase;margin-bottom:12px">QWEN — MEDICAL REASONING ENGINE</div>""", unsafe_allow_html=True)

                    with st.spinner("QUERYING QWEN..."):
                        qwen = call_qwen(img, hmap,
                                         body_part, prob,
                                         threshold)

                    if qwen:
                        gd   = qwen.get("fracture_detected", False)
                        gc   = qwen.get("confidence", 0)
                        gs   = qwen.get("status", "unknown")
                        mb   = qwen.get("marker_bias_detected", False)
                        gcol = "#FF3C3C" if gd else \
                               "#FFA500" if gs=="inconclusive" \
                               else "#00FF88"
                        agree     = (gd == prediction)
                        badge_cls = "badge-agree" if agree else "badge-disagree"
                        badge_txt = "MODELS AGREE" if agree else "MODELS DISAGREE"

                        st.markdown(f"""
                        <div class="qwen-panel">
                            <div class="qwen-header">
                                <span class="qwen-header-title">QWEN STRUCTURED MEDICAL REPORT</span>
                                <span class="badge {badge_cls}">{badge_txt}</span>
                            </div>
                            <div class="qwen-row"><span class="qwen-key">FRACTURE DETECTED</span><span class="qwen-val" style="color:{gcol};font-weight:600">{'YES ⚠' if gd else 'NO ✓'}</span></div>
                            <div class="qwen-row"><span class="qwen-key">CONFIDENCE</span><span class="qwen-val" style="color:{gcol}">{gc*100:.1f}%</span></div>
                            <div class="qwen-row"><span class="qwen-key">STATUS</span><span class="qwen-val" style="color:{gcol}">{gs.upper()}</span></div>
                            <div class="qwen-row"><span class="qwen-key">FRACTURE TYPE</span><span class="qwen-val">{qwen.get('fracture_type','—') or '—'}</span></div>
                            <div class="qwen-row"><span class="qwen-key">LOCATION</span><span class="qwen-val">{qwen.get('location','—') or '—'}</span></div>
                            <div class="qwen-row"><span class="qwen-key">SEVERITY</span><span class="qwen-val">{qwen.get('severity','—') or '—'}</span></div>
                            <div class="qwen-row"><span class="qwen-key">ATTENTION QUALITY</span><span class="qwen-val">{qwen.get('attention_quality','—')}</span></div>
                            <div class="qwen-row"><span class="qwen-key">MARKER BIAS</span><span class="qwen-val" style="color:{'#FF3C3C' if mb else '#00FF88'}">{'YES ⚠' if mb else 'NO ✓'}</span></div>
                            <div class="qwen-row"><span class="qwen-key">CLINICAL FINDINGS</span><span class="qwen-val">{qwen.get('clinical_findings','—')}</span></div>
                            <div class="qwen-row"><span class="qwen-key">RECOMMENDATION</span><span class="qwen-val">{qwen.get('recommendation','—')}</span></div>
                            <div class="qwen-row"><span class="qwen-key">TREATMENT PLAN</span><span class="qwen-val">{qwen.get('treatment_plan','—')}</span></div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""<div style="background:rgba(255,165,0,0.04);border:1px solid rgba(255,165,0,0.2);padding:14px 16px;border-radius:3px;font-family:'Share Tech Mono',monospace;font-size:0.8rem;color:rgba(255,165,0,0.7);letter-spacing:1px">⚠ QWEN UNAVAILABLE — RUN 'ollama serve' IN TERMINAL — BIOMED CLIP RESULT REMAINS VALID</div>""", unsafe_allow_html=True)

                st.markdown("""<div class="disclaimer-box"><p>⚠ MEDICAL DISCLAIMER — This AI system is intended for research and educational purposes only. All results must be verified by a licensed radiologist before any clinical decision is made. Not FDA approved.</p></div>""", unsafe_allow_html=True)
                render_treatment_guide(body_part, prediction, prob)

                st.session_state["diagnosis_context"] = {
                    "body_part"    : body_part,
                    "prediction"   : prediction,
                    "prob"         : prob,
                    "threshold"    : threshold,
                    "qwen_findings": (qwen or {}).get(
                        "clinical_findings", ""),
                }

            else:
                c1, c2 = st.columns([1, 2])
                with c1:
                    st.image(img, caption="INPUT IMAGE",
                             use_container_width=True)
                with c2:
                    st.markdown(f"""
                    <div class="panel" style="margin-top:0">
                        <div class="panel-title">SCAN PARAMETERS</div>
                        <div style="padding-left:8px">
                            <div style="margin-bottom:14px">
                                <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.4);letter-spacing:2px;text-transform:uppercase">TARGET ANATOMY</div>
                                <div style="font-family:'Rajdhani',sans-serif;font-size:1.4rem;font-weight:600;color:#00C8FF;margin-top:2px">{BP_LABELS[body_part]}</div>
                            </div>
                            <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px">
                                <div><div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,200,255,0.35);letter-spacing:1px">DECISION THRESHOLD</div><div style="font-family:'Share Tech Mono',monospace;font-size:1.1rem;color:#C8E6F0">{threshold:.2f}</div></div>
                                <div><div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,200,255,0.35);letter-spacing:1px">SPECIALIST AUC</div><div style="font-family:'Share Tech Mono',monospace;font-size:1.1rem;color:#C8E6F0">{BP_STATS[body_part]['auc']:.4f}</div></div>
                                <div><div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,200,255,0.35);letter-spacing:1px">QWEN REPORT</div><div style="font-family:'Share Tech Mono',monospace;font-size:1.1rem;color:#00FF88">{'ON' if use_qwen else 'OFF'}</div></div>
                                <div><div style="font-family:'Share Tech Mono',monospace;font-size:0.6rem;color:rgba(0,200,255,0.35);letter-spacing:1px">COMPUTE</div><div style="font-family:'Share Tech Mono',monospace;font-size:1.1rem;color:#C8E6F0">{DEVICE.upper()}</div></div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.info("▶ Click **ANALYZE** in sidebar")

    # ── TAB 2 ────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.4);letter-spacing:3px;text-transform:uppercase;border-bottom:1px solid rgba(0,200,255,0.1);padding-bottom:8px;margin-bottom:16px">PRE-COMPUTED VALIDATION SAMPLES</div>
        <p style="font-family:'Exo 2',sans-serif;font-size:0.85rem;color:rgba(0,200,255,0.5);margin-bottom:16px">Real predictions from MURA validation set — generated by Qwen3-VL.</p>
        """, unsafe_allow_html=True)
        sel  = st.selectbox("FILTER BODY PART",
                            ["All"]+BODY_PARTS,
                            format_func=lambda x:
                            "All Body Parts" if x=="All"
                            else BP_LABELS[x])
        show = BODY_PARTS if sel=="All" else [sel]
        for bp in show:
            paths, labels = [], []
            for p, l in [
                (os.path.join(GEMMA_DIR, bp,
                              "fracture_report.png"),
                 "QWEN: FRACTURE"),
                (os.path.join(GEMMA_DIR, bp,
                              "no_fracture_report.png"),
                 "QWEN: NORMAL"),
            ]:
                if os.path.exists(p):
                    paths.append(p)
                    labels.append(l)
            if not paths: continue
            auc_val = BP_STATS[bp]['auc']
            auc_col = "#00FF88" if auc_val>=0.88 else \
                      "#00C8FF" if auc_val>=0.83 else \
                      "#FF4466"
            st.markdown(
                f"""<div style="display:flex;align-items:center;gap:12px;margin:16px 0 8px 0">
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.8rem;color:#C8E6F0;letter-spacing:2px">{BP_LABELS[bp].upper()}</div>
                <div style="font-family:'Share Tech Mono',monospace;font-size:0.7rem;color:{auc_col}">AUC {auc_val:.4f}</div>
                </div>""", unsafe_allow_html=True)
            cols  = st.columns(min(len(paths), 2))
            for i, col in enumerate(cols):
                if i < len(paths):
                    with col:
                        st.image(Image.open(paths[i]),
                                 caption=labels[i],
                                 use_container_width=True)
            st.markdown('<div class="cyber-divider"></div>',
                        unsafe_allow_html=True)

    # ── TAB 3 ────────────────────────────────────────────────
    with tab3:
        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.4);letter-spacing:3px;text-transform:uppercase;border-bottom:1px solid rgba(0,200,255,0.1);padding-bottom:8px;margin-bottom:16px">SYSTEM PERFORMANCE METRICS</div>""", unsafe_allow_html=True)

        mc = st.columns(4)
        for col, (val, lbl, color) in zip(mc, [
            ("0.8669","MEAN AUC","#00C8FF"),
            ("0.9007","BEST AUC (WRIST)","#00FF88"),
            ("0.8242","LOWEST AUC (HAND)","#FF4466"),
            ("+62%","AUC GAIN","#FFD700"),
        ]):
            with col:
                st.markdown(
                    f"""<div class="metric-card">
                    <div class="metric-val" style="color:{color}">{val}</div>
                    <div class="metric-lbl">{lbl}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        fig1 = journey_chart()
        st.pyplot(fig1, use_container_width=True)
        plt.close()

        st.markdown("<br>", unsafe_allow_html=True)
        fig2 = auc_chart()
        st.pyplot(fig2, use_container_width=True)
        plt.close()

        st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.4);letter-spacing:3px;text-transform:uppercase;margin-bottom:12px">PIPELINE ARCHITECTURE</div>
        <div class="pipeline-box">
            <span>INPUT: Musculoskeletal X-Ray (PNG/JPG)</span>
            <span class="pipe-arrow">▼</span>
            <span class="pipe-step">BiomedCLIP ViT-B/16 (fine-tuned via LoRA-32)</span>
            <span class="pipe-arrow">▼</span>
            <span class="pipe-step">512-dim Image + 64-dim Exclude = 576-dim Features</span>
            <span class="pipe-arrow">▼</span>
            <span class="pipe-step">Body-Part Specialist Head (7 parallel heads)</span>
            <span class="pipe-arrow">▼</span>
            <span class="pipe-step">Fracture Probability + Optimal Threshold</span>
            <span class="pipe-arrow">▼</span>
            <span class="pipe-step">Attention Rollout → Spatial Heatmap (α=0.3)</span>
            <span class="pipe-arrow">▼</span>
            <span class="pipe-step">Qwen3-VL (local) → Structured Medical Report</span>
            <span class="pipe-arrow">▼</span>
            <span>OUTPUT: fracture, confidence, location, severity, treatment</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.4);letter-spacing:3px;text-transform:uppercase;margin-bottom:12px">TECHNOLOGY STACK</div>""", unsafe_allow_html=True)

        tc = st.columns(5)
        for col, (icon, name, desc) in zip(tc, [
            ("🧠","BiomedCLIP","Vision Encoder"),
            ("⚡","LoRA-32","PEFT Fine-tuning"),
            ("👁","Attn Rollout","Explainability"),
            ("🤖","Qwen3-VL","Med Reasoning"),
            ("🦴","MURA v1.1","7 Body Parts"),
        ]):
            with col:
                st.markdown(
                    f"""<div class="tech-card">
                    <span class="tech-icon">{icon}</span>
                    <div class="tech-name">{name}</div>
                    <div class="tech-desc">{desc}</div>
                    </div>""", unsafe_allow_html=True)

        st.markdown('<div class="cyber-divider"></div>', unsafe_allow_html=True)
        st.markdown("""<div style="font-family:'Share Tech Mono',monospace;font-size:0.65rem;color:rgba(0,200,255,0.4);letter-spacing:3px;text-transform:uppercase;margin-bottom:12px">TRAINING CONFIGURATION</div>""", unsafe_allow_html=True)

        dc = st.columns(2)
        with dc[0]:
            st.markdown("""<div class="panel"><div class="panel-title">PHASE 1 — LORA FINE-TUNING</div><div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:rgba(0,200,255,0.55);line-height:2;padding-left:8px">BASE MODEL &nbsp;: BiomedCLIP ViT-B/16<br>LORA RANK &nbsp;&nbsp;: 32<br>LORA ALPHA &nbsp;: 64 (scaling=2.0)<br>LORA BLOCKS &nbsp;: 8/12<br>EXCLUDE DIM &nbsp;: 64 (marker exclusion)<br>BATCH SIZE &nbsp;: 8<br>LEARNING RATE : 1e-4 (LoRA), 1e-3 (head)<br>FOCAL LOSS &nbsp;: gamma=2.0<br>BEST AUC &nbsp;&nbsp;&nbsp;: 0.8555</div></div>""", unsafe_allow_html=True)
        with dc[1]:
            st.markdown("""<div class="panel"><div class="panel-title">PHASE 2-3 — SPECIALISTS + THRESHOLDS</div><div style="font-family:'Share Tech Mono',monospace;font-size:0.75rem;color:rgba(0,200,255,0.55);line-height:2;padding-left:8px">HEADS &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: 7 specialists (v2)<br>ARCHITECTURE : 576→128→2<br>BEST HEAD &nbsp;&nbsp;: XR_WRIST (0.9007)<br>MEAN AUC &nbsp;&nbsp;&nbsp;: 0.8669<br>THRESHOLD &nbsp;&nbsp;: Tuned per body part<br>EXPLAINABILITY: Attention Rollout<br>LLM BACKEND &nbsp;: Qwen3-VL (local/Ollama)<br>DATASET &nbsp;&nbsp;&nbsp;&nbsp;: MURA v1.1 (Stanford)<br>TOTAL IMGS &nbsp;: 40,005</div></div>""", unsafe_allow_html=True)

    st.markdown("---")
    call_chatbot_render(diag_ctx)


if __name__ == "__main__":
    main()