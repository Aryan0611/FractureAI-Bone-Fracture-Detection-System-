# ==============================================================
# Phase7_Qwen.py
# FRACTURE DETECTION PROJECT — v3
# ==============================================================
# Medical reasoning using Qwen3-VL locally via Ollama
# Sends 4 images to Qwen per prediction:
#   1. Original X-ray         (raw scan, may have markers)
#   2. Clean enhanced X-ray   (markers removed, bone edges bright)
#   3. Attention heatmap      (where BiomedCLIP focused)
#   4. Overlay                (heatmap blended on original — shows
#                              if AI focused on marker or bone)
# Gets:  structured JSON medical report
# No rate limits — runs fully local on your GPU
# ==============================================================

import os
import io
import json
import base64
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import open_clip
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from ollama import chat
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

SAVE_DIR     = os.path.join(BASE_PATH, "models", "saved")
RESULTS_DIR  = os.path.join(BASE_PATH, "results", "gemma")
MARKER_CSV   = os.path.join(BASE_PATH, "data",
                            "marker_coords.csv")
CLEAN_DIR    = os.path.join(BASE_PATH, "data",
                            "clean")

PHASE1_MODEL = os.path.join(SAVE_DIR,
                            "phase1_best_model.pt")
THRESH_FILE  = os.path.join(SAVE_DIR,
                            "optimal_thresholds.json")
os.makedirs(RESULTS_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Qwen local config ─────────────────────────────────────────
QWEN_MODEL     = "qwen3-vl:235b-cloud"  # cloud version
# If cloud unavailable fall back to local:
QWEN_FALLBACK  = "qwen2.5vl:7b"

N_SAMPLES      = 1      # 1 fracture + 1 normal per body part
MARKER_PADDING = 0.03
CROP_MARGIN    = 0.12   # must match CleanAnnotations.py

BODY_PARTS = [
    "XR_ELBOW",   "XR_FINGER", "XR_FOREARM",
    "XR_HAND",    "XR_HUMERUS","XR_SHOULDER","XR_WRIST"
]

# ==============================================================
# 2. MARKER COORDS
# ==============================================================

def load_marker_coords() -> dict:
    if not os.path.exists(MARKER_CSV):
        print("⚠️  marker_coords.csv not found")
        return {}
    df      = pd.read_csv(MARKER_CSV)
    markers = {}
    for _, row in df.iterrows():
        if row["has_marker"] == 1:
            markers[row["image_path"]] = {
                "x1": max(0.0, float(row["x1"])
                          - MARKER_PADDING),
                "y1": max(0.0, float(row["y1"])
                          - MARKER_PADDING),
                "x2": min(1.0, float(row["x2"])
                          + MARKER_PADDING),
                "y2": min(1.0, float(row["y2"])
                          + MARKER_PADDING),
            }
    print(f"✅ Markers loaded: {len(markers)} images")
    return markers


def get_exclude_tensor(image_path, markers):
    if image_path in markers:
        m  = markers[image_path]
        x1, y1, x2, y2 = (m["x1"], m["y1"],
                           m["x2"], m["y2"])
        return torch.tensor(
            [x1, y1, x2, y2,
             (x1+x2)/2, (y1+y2)/2,
             x2-x1, y2-y1],
            dtype=torch.float32)
    return torch.zeros(8, dtype=torch.float32)


# ==============================================================
# 3. CLEAN IMAGE (same pipeline as CleanAnnotations.py)
# ==============================================================

def get_clean_image(image_path: str) -> Image.Image:
    """
    Applies the same preprocessing as CleanAnnotations.py:
    1. Corner crop — removes annotation markers
    2. Bone-masked edge enhancement

    This gives Qwen a cleaner image to reason about
    compared to the raw X-ray with R/L markers.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return Image.open(image_path).convert("RGB")

    h, w = img.shape

    # Step 1 — Corner crop
    mh      = max(int(h * CROP_MARGIN), 5)
    mw      = max(int(w * CROP_MARGIN), 5)
    cropped = img[mh:h-mh, mw:w-mw]
    cropped = cv2.resize(cropped, (w, h),
                         interpolation=cv2.INTER_LINEAR)

    # Step 2 — CLAHE
    clahe    = cv2.createCLAHE(clipLimit=2.0,
                                tileGridSize=(8, 8))
    enhanced = clahe.apply(cropped)

    # Step 3 — Bone mask
    otsu_t, _ = cv2.threshold(
        enhanced, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bt = max(20, int(otsu_t * 0.3))
    _, bmask = cv2.threshold(
        enhanced, bt, 255, cv2.THRESH_BINARY)
    bmask = cv2.morphologyEx(
        bmask, cv2.MORPH_CLOSE,
        np.ones((5, 5), np.uint8))

    # Step 4 — Masked edges
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges   = cv2.Canny(blurred, 15, 50)
    edges_m = cv2.bitwise_and(edges, edges, mask=bmask)
    edges_d = cv2.dilate(
        edges_m, np.ones((2, 2), np.uint8),
        iterations=1)

    # Step 5 — Blend
    result = cv2.addWeighted(enhanced, 0.85,
                             edges_d, 0.15, 0)

    # Convert to RGB PIL for sending to Qwen
    rgb = cv2.cvtColor(result, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(rgb)


# ==============================================================
# 4. LORA + MODELS
# ==============================================================

class LoRALinear(nn.Module):
    def __init__(self, linear, r=32, alpha=64):
        super().__init__()
        self.linear  = linear
        self.scaling = alpha / r
        in_f  = linear.weight.shape[1]
        out_f = linear.weight.shape[0]
        self.lora_A = nn.Parameter(
            torch.randn(r, in_f) * 0.01)
        self.lora_B = nn.Parameter(
            torch.zeros(out_f, r))
        self.linear.weight.requires_grad_(False)
        if self.linear.bias is not None:
            self.linear.bias.requires_grad_(False)

    def forward(self, x):
        return (self.linear(x) +
                (x @ self.lora_A.T @ self.lora_B.T)
                * self.scaling)


def find_blocks(model):
    for attr in [
        "visual.transformer.resblocks",
        "visual.trunk.blocks",
        "visual.model.blocks",
        "visual.blocks"
    ]:
        try:
            obj = model
            for a in attr.split("."):
                obj = getattr(obj, a)
            return obj
        except AttributeError:
            pass
    raise AttributeError("Cannot find blocks.")


def inject_lora_block(block, r, alpha):
    if hasattr(block, "attn"):
        a = block.attn
        if hasattr(a, "out_proj") and isinstance(
                a.out_proj, nn.Linear):
            a.out_proj = LoRALinear(a.out_proj, r, alpha)
        elif hasattr(a, "proj") and isinstance(
                a.proj, nn.Linear):
            a.proj = LoRALinear(a.proj, r, alpha)
    if hasattr(block, "mlp"):
        m = block.mlp
        for attr in ["c_fc","c_proj","fc1","fc2"]:
            if hasattr(m, attr) and isinstance(
                    getattr(m, attr), nn.Linear):
                setattr(m, attr,
                        LoRALinear(getattr(m, attr),
                                   r, alpha))


def inject_lora(model, r=32, alpha=64, n=8):
    for p in model.parameters():
        p.requires_grad_(False)
    blocks = find_blocks(model)
    total  = len(blocks)
    for i, b in enumerate(blocks):
        if i >= total - n:
            inject_lora_block(b, r, alpha)
    return model


class FractureClassifierExclude(nn.Module):
    def __init__(self, clip_model,
                 embed_dim=512, exclude_dim=64,
                 num_classes=2, dropout=0.3):
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
    def __init__(self, embed_dim=576,
                 num_classes=2, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes))

    def forward(self, x): return self.head(x)


# ==============================================================
# 5. ATTENTION ROLLOUT
# ==============================================================

class AttentionRollout:
    def __init__(self, clip_model):
        self.clip_model = clip_model

    def generate(self, img_tensor):
        visual     = self.clip_model.visual
        img_tensor = img_tensor.to(DEVICE)
        try: blocks = visual.trunk.blocks
        except:
            try: blocks = visual.transformer.resblocks
            except: blocks = visual.blocks

        attn_list, hooks = [], []

        def make_hook():
            def hook(module, inp, out):
                if isinstance(inp, tuple) and len(inp) > 0:
                    x = inp[0]; B, N, C = x.shape
                    if hasattr(module, 'qkv'):
                        nh  = getattr(module, 'num_heads', 12)
                        qkv = module.qkv(x).reshape(
                            B, N, 3, nh, -1).permute(2,0,3,1,4)
                        q, k, v = qkv.unbind(0)
                        attn = ((q @ k.transpose(-2,-1)) *
                                (q.shape[-1] ** -0.5))
                        attn = attn.softmax(-1)
                        attn_list.append(
                            attn.mean(1).detach().cpu())
            return hook

        for b in blocks:
            if hasattr(b, 'attn'):
                hooks.append(
                    b.attn.register_forward_hook(
                        make_hook()))

        with torch.no_grad(): visual(img_tensor)
        for h in hooks: h.remove()

        if not attn_list:
            return np.ones((224, 224)) * 0.3

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
        return cv2.resize(m, (224, 224),
                          interpolation=cv2.INTER_LINEAR)


# ==============================================================
# 6. IMAGE UTILITIES
# ==============================================================

def pil_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.resize((224, 224)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def heatmap_to_pil(heatmap: np.ndarray) -> Image.Image:
    return Image.fromarray(
        (cm.jet(heatmap)[:, :, :3] * 255
         ).astype(np.uint8))


def make_overlay(original_pil: Image.Image,
                 heatmap: np.ndarray,
                 alpha: float = 0.3) -> Image.Image:
    """
    Blends attention heatmap onto original X-ray.

    This is the most informative image for Qwen:
    - If RED is on the annotation letter (R/L)
      → BiomedCLIP was cheating on markers
    - If RED is on the bone cortex
      → BiomedCLIP was looking at the right place

    Qwen can see this directly and comment on it
    in attention_quality field.
    """
    img_np = np.array(
        original_pil.resize((224, 224)).convert("RGB"))
    heat_c = (cm.jet(heatmap)[:, :, :3] * 255
              ).astype(np.uint8)
    overlay = (img_np * (1 - alpha) +
               heat_c * alpha).astype(np.uint8)
    return Image.fromarray(overlay)


# ==============================================================
# 7. QWEN LOCAL CALL (3 images)
# ==============================================================

def call_qwen(body_part, prob, threshold,
              original_pil, clean_pil,
              heatmap_pil, overlay_pil,
              model=None) -> dict:
    """
    Sends 4 images to Qwen:
    1. Original X-ray      — raw scan with possible R/L markers
    2. Clean enhanced      — markers removed, bone edges bright
    3. Attention heatmap   — pure heatmap (red = AI focus)
    4. Overlay             — heatmap ON original X-ray
                             most informative: shows EXACTLY
                             if AI focused on marker or bone

    Qwen can use all 4 together to:
    - Compare original vs clean (spot markers)
    - See where BiomedCLIP focused (heatmap)
    - Verify focus is on bone not letter (overlay)
    - Analyze bone structure from clean image
    """
    if model is None:
        model = QWEN_MODEL

    bpn  = body_part.replace("XR_", "")
    pred = ("FRACTURE DETECTED"
            if prob >= threshold
            else "NO FRACTURE DETECTED")

    prompt = f"""You are an expert radiologist AI.

CONTEXT:
- Body part  : {bpn}
- BiomedCLIP : {pred} ({prob*100:.1f}% probability)
- Threshold  : {threshold:.2f}

You are given 4 images in order:
1. ORIGINAL X-RAY     — raw scan, may have annotation letters (R/L/RY)
2. CLEAN ENHANCED     — same scan but markers removed, bone edges highlighted
3. ATTENTION HEATMAP  — RED areas = where BiomedCLIP AI focused most
4. OVERLAY            — heatmap blended ON the original X-ray
                        Use this to judge: is red on the letter marker
                        or on the actual bone? This is critical.

INSTRUCTIONS:
- Use image 2 (clean) for bone structure analysis
- Use image 4 (overlay) to judge attention_quality:
    "bone-focused"   = red areas are on bone/cortex
    "marker-focused" = red areas are on R/L text labels
    "mixed"          = red on both
- If marker-focused, lower your confidence slightly
  as BiomedCLIP may have used a shortcut

Return ONLY valid JSON, no extra text:
{{
    "fracture_detected": true or false,
    "confidence": 0.0 to 1.0,
    "status": "confirmed or inconclusive or normal",
    "fracture_type": "description or null",
    "location": "specific bone location or null",
    "severity": "mild or moderate or severe or null",
    "attention_quality": "bone-focused or marker-focused or mixed",
    "marker_bias_detected": true or false,
    "clinical_findings": "one sentence describing bone structure",
    "recommendation": "clinical recommendation in one sentence",
    "treatment_plan": "brief treatment approach",
    "disclaimer": "AI-generated. Radiologist confirmation required."
}}"""

    try:
        r = chat(
            model=model,
            messages=[{
                "role"   : "user",
                "content": prompt,
                "images" : [
                    pil_to_base64(original_pil),  # 1. raw
                    pil_to_base64(clean_pil),      # 2. clean
                    pil_to_base64(heatmap_pil),    # 3. heatmap
                    pil_to_base64(overlay_pil),    # 4. overlay
                ]
            }]
        )
        raw = r.message.content.strip()

        # Clean markdown fences
        for p in ["```json", "```"]:
            if raw.startswith(p):
                raw = raw[len(p):]
        if raw.endswith("```"):
            raw = raw[:-3]

        # Strip <think>...</think> tags if present
        import re
        raw = re.sub(r'<think>.*?</think>', '',
                     raw, flags=re.DOTALL).strip()

        result = json.loads(raw.strip())

        if not result.get("fracture_detected", False):
            result.update({
                "fracture_type": None,
                "location"     : None,
                "severity"     : None})

        conf = float(result.get("confidence", 0))
        result["confidence"] = max(0.0, min(1.0, conf))
        result["model_used"] = model
        return result

    except Exception as e:
        # Try fallback model
        if model != QWEN_FALLBACK:
            print(f"  ⚠️ {model} failed: {str(e)[:60]}")
            print(f"  Trying fallback: {QWEN_FALLBACK}")
            return call_qwen(
                body_part, prob, threshold,
                original_pil, clean_pil,
                heatmap_pil, overlay_pil,
                model=QWEN_FALLBACK)

        return {
            "fracture_detected"  : None,
            "confidence"         : 0.0,
            "status"             : "error",
            "fracture_type"      : None,
            "location"           : None,
            "severity"           : None,
            "attention_quality"  : "unknown",
            "marker_bias_detected": None,
            "clinical_findings"  : f"Qwen error: "
                                    f"{str(e)[:60]}",
            "recommendation"     : "Manual review",
            "treatment_plan"     : "Consult radiologist",
            "disclaimer"         : "AI unavailable.",
            "model_used"         : "none",
        }


# ==============================================================
# 8. VISUALIZATION (4 panels now — adds clean image)
# ==============================================================

def wrap_text(text, max_chars=28):
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line + " " + w) > max_chars:
            if line: lines.append(line)
            line = w
        else:
            line += (" " if line else "") + w
    if line: lines.append(line)
    return lines


def plot_result(original_img, clean_img,
                heatmap, overlay_pil, result,
                body_part, prob, threshold,
                true_label, save_path):
    """
    5-panel visualization:
    [Original] [Clean] [Heatmap] [Overlay] [Qwen Report]
    """
    fig = plt.figure(figsize=(26, 6))

    detected = result.get("fracture_detected", False)
    status   = result.get("status", "unknown")
    color    = ("#E74C3C" if detected
                else "#F39C12" if status == "inconclusive"
                else "#2ECC71")
    marker_bias = result.get("marker_bias_detected", False)

    # Panel 1 — Original
    ax1 = fig.add_subplot(1, 5, 1)
    ax1.imshow(np.array(
        original_img.resize((224,224)).convert("RGB")))
    ax1.set_title("1. Original X-Ray\n(raw with markers)",
                  fontsize=10, fontweight="bold")
    ax1.axis("off")

    # Panel 2 — Clean
    ax2 = fig.add_subplot(1, 5, 2)
    ax2.imshow(np.array(
        clean_img.resize((224,224)).convert("RGB")))
    ax2.set_title("2. Clean Enhanced\n(markers removed, edges bright)",
                  fontsize=10, color="#2980B9",
                  fontweight="bold")
    ax2.axis("off")

    # Panel 3 — Heatmap
    ax3 = fig.add_subplot(1, 5, 3)
    im  = ax3.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    ax3.set_title("3. Attention Heatmap\n(pure — red = AI focus)",
                  fontsize=10)
    ax3.axis("off")
    plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)

    # Panel 4 — Overlay (heatmap ON original)
    ax4 = fig.add_subplot(1, 5, 4)
    ax4.imshow(np.array(
        overlay_pil.resize((224, 224))))
    attn_q = result.get("attention_quality", "unknown")
    attn_col = ("#2ECC71" if attn_q == "bone-focused"
                else "#E74C3C" if attn_q == "marker-focused"
                else "#F39C12")
    bias_note = " ⚠ MARKER BIAS" if marker_bias else ""
    ax4.set_title(
        f"4. Overlay (heatmap + original)\n"
        f"Attn: {attn_q}{bias_note}",
        fontsize=9, color=attn_col)
    ax4.axis("off")

    # Panel 5 — Qwen Report
    ax5 = fig.add_subplot(1, 5, 5)
    ax5.axis("off")
    ax5.set_facecolor("#F0F3F4")

    pred_t  = ("Fracture" if prob >= threshold
               else "No Fracture")
    true_t  = ("Fracture" if true_label == 1
               else "No Fracture")
    correct = ((detected == bool(true_label))
               if detected is not None else None)

    lines = [
        ("QWEN REPORT",       12, "bold",   "#1A252F"),
        ("="*22,               9, "normal", "#BDC3C7"),
        (f"Detected: "
         f"{'YES' if detected else 'NO'}",
                              12, "bold",   color),
        (f"Conf: "
         f"{result.get('confidence',0)*100:.1f}%",
                              10, "normal", "#2C3E50"),
        (f"Status: {status.upper()}",
                              10, "bold",
         "#E74C3C" if status=="inconclusive"
         else "#27AE60"),
        (f"True: {true_t} "
         f"{'✓' if correct else '✗' if correct is not None else '?'}",
                               9, "normal",
         "#27AE60" if correct else
         "#E74C3C" if correct is not None else "#95A5A6"),
        ("-"*22,               9, "normal", "#BDC3C7"),
        (f"Type: "
         f"{result.get('fracture_type') or 'N/A'}",
                               9, "normal", "#2C3E50"),
        (f"Loc: "
         f"{result.get('location') or 'N/A'}",
                               9, "normal", "#2C3E50"),
        (f"Sev: "
         f"{result.get('severity') or 'N/A'}",
                               9, "normal", "#2C3E50"),
        (f"Attn: {attn_q}",   9, "normal", attn_col),
        (f"Bias: "
         f"{'YES ⚠' if marker_bias else 'NO ✓'}",
                               9, "normal",
         "#E74C3C" if marker_bias else "#27AE60"),
        ("-"*22,               9, "normal", "#BDC3C7"),
    ]

    findings = result.get("clinical_findings", "")
    if findings:
        lines.append(("Findings:", 8, "bold", "#2C3E50"))
        for l in wrap_text(findings):
            lines.append((l, 7, "normal", "#555"))
        lines.append(("-"*22, 8, "normal", "#BDC3C7"))

    rec = result.get("recommendation", "")
    if rec:
        lines.append(("Rec:", 8, "bold", "#2C3E50"))
        for l in wrap_text(rec):
            lines.append((l, 7, "normal", "#555"))

    lines += [
        ("-"*22,               8, "normal", "#BDC3C7"),
        ("AI-generated.",       7, "normal", "#E67E22"),
        ("Radiologist required",7, "normal", "#E67E22"),
    ]

    y = 0.97
    for text, size, weight, col in lines:
        ax5.text(0.04, y, text,
                 transform=ax5.transAxes,
                 fontsize=size, fontweight=weight,
                 color=col, va="top",
                 fontfamily="monospace")
        y -= 0.048

    plt.suptitle(
        f"Qwen3-VL Medical Analysis — {body_part}  "
        f"| BiomedCLIP: {pred_t} ({prob*100:.1f}%)",
        fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120,
                bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}")


# ==============================================================
# 9. LOAD SYSTEM
# ==============================================================

def load_system(markers):
    print("Loading BiomedCLIP + LoRA v2...")
    with open(THRESH_FILE) as f:
        thresholds = json.load(f)

    clip_model, _, _ = open_clip.create_model_and_transforms(
        "hf-hub:microsoft/BiomedCLIP-"
        "PubMedBERT_256-vit_base_patch16_224")
    clip_model = inject_lora(
        clip_model, r=32, alpha=64, n=8)

    encoder = FractureClassifierExclude(
        clip_model).to(DEVICE)
    encoder.load_state_dict(
        torch.load(PHASE1_MODEL, map_location=DEVICE))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    print("✅ Encoder loaded")

    heads = {}
    for bp in BODY_PARTS:
        for fn in [f"head_v2_{bp}.pt",
                   f"head_{bp}.pt"]:
            path = os.path.join(SAVE_DIR, fn)
            if os.path.exists(path):
                head = BodyPartHead(576).to(DEVICE)
                try:
                    head.load_state_dict(
                        torch.load(path,
                                   map_location=DEVICE))
                    head.eval()
                    heads[bp] = head
                    print(f"  ✅ {bp} ({fn})")
                    break
                except Exception as e:
                    print(f"  ⚠️ {fn}: {e}")

    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275,  0.40821073),
            (0.26862954, 0.26130258, 0.27577711))])

    print(f"✅ {len(heads)}/7 heads loaded")
    return encoder, heads, thresholds, tf


def predict_single(encoder, head, tf,
                   img_path, markers, threshold):
    img     = Image.open(img_path).convert("RGB")
    tensor  = tf(img).unsqueeze(0).to(DEVICE)
    exclude = get_exclude_tensor(
        img_path, markers).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feats  = encoder.encode(tensor, exclude)
        logits = head(feats)
        prob   = float(torch.softmax(
            logits, 1)[0, 1].detach().cpu())
    return img, prob


# ==============================================================
# 10. MAIN
# ==============================================================

if __name__ == "__main__":

    print("="*60)
    print("PHASE 7 — Qwen3-VL Medical Reasoning [v3]")
    print("Model  : Qwen3-VL (local via Ollama)")
    print("Images : Original + Clean + Heatmap + Overlay")
    print("="*60)

    # Check Ollama is running
    print("\nChecking Ollama...")
    try:
        test = chat(
            model=QWEN_MODEL,
            messages=[{
                "role"   : "user",
                "content": "Reply with one word: ready"}])
        print(f"✅ Qwen online: "
              f"{test.message.content.strip()[:30]}")
    except Exception as e:
        print(f"❌ Ollama not running: {e}")
        print("  Fix: run 'ollama serve' in terminal")
        exit(1)

    # Load everything
    markers = load_marker_coords()
    encoder, heads, thresholds, tf = load_system(markers)
    rollout  = AttentionRollout(encoder.clip_model)
    val_df   = pd.read_csv(VAL_CSV)

    print(f"\nBody parts     : {len(BODY_PARTS)}")
    print(f"Samples/part   : {N_SAMPLES}×2 "
          f"(fracture + normal)")
    print(f"Images/call    : 4 (original, clean, heatmap, overlay)")
    print(f"Total calls    : ~{len(BODY_PARTS)*N_SAMPLES*2}")

    all_results = []

    for bp in BODY_PARTS:
        print(f"\n{'─'*50}")
        print(f"Body Part: {bp}")
        print(f"{'─'*50}")

        bp_dir    = os.path.join(RESULTS_DIR, bp)
        os.makedirs(bp_dir, exist_ok=True)

        bp_df     = val_df[val_df["body_part"] == bp]
        head      = heads.get(bp)
        threshold = thresholds.get(bp, 0.5)

        if head is None or len(bp_df) == 0:
            print("  Skipping — no data or head")
            continue

        fractures = bp_df[bp_df["label"]==1].head(N_SAMPLES)
        normals   = bp_df[bp_df["label"]==0].head(N_SAMPLES)
        samples   = pd.concat([fractures, normals])

        for _, row in samples.iterrows():
            try:
                img_path   = row["image_path"]
                true_label = int(row["label"])
                label_str  = ("fracture"
                              if true_label == 1
                              else "no_fracture")

                print(f"\n  [{label_str}]")

                # BiomedCLIP prediction
                original_img, prob = predict_single(
                    encoder, head, tf,
                    img_path, markers, threshold)

                pred_text = ("Fracture"
                             if prob >= threshold
                             else "No Fracture")
                print(f"  BiomedCLIP : {pred_text} "
                      f"({prob*100:.1f}%)")

                # Clean enhanced image for Qwen
                clean_img = get_clean_image(img_path)
                print(f"  Clean img  : generated ✅")

                # Attention heatmap + overlay
                tensor      = tf(original_img).unsqueeze(0)
                heatmap     = rollout.generate(tensor)
                heatmap_pil = heatmap_to_pil(heatmap)
                overlay_pil = make_overlay(original_img,
                                           heatmap)

                # Qwen analysis — 4 images
                print(f"  Calling Qwen (4 images)...")
                result = call_qwen(
                    body_part    = bp,
                    prob         = prob,
                    threshold    = threshold,
                    original_pil = original_img,
                    clean_pil    = clean_img,
                    heatmap_pil  = heatmap_pil,
                    overlay_pil  = overlay_pil)

                mb = result.get("marker_bias_detected",
                                False)
                print(f"  Detected  : "
                      f"{result.get('fracture_detected')}")
                print(f"  Confidence: "
                      f"{result.get('confidence',0)*100:.1f}%")
                print(f"  Status    : "
                      f"{result.get('status')}")
                print(f"  Attention : "
                      f"{result.get('attention_quality')}")
                print(f"  Mrkr bias : "
                      f"{'YES ⚠' if mb else 'NO ✓'}")
                print(f"  Finding   : "
                      f"{str(result.get('clinical_findings',''))[:60]}...")

                # Save visualization (5 panels)
                save_path = os.path.join(
                    bp_dir, f"{label_str}_report.png")
                plot_result(
                    original_img, clean_img,
                    heatmap, overlay_pil, result,
                    bp, prob, threshold,
                    true_label, save_path)

                # Save JSON
                json_path = os.path.join(
                    bp_dir, f"{label_str}_result.json")
                with open(json_path, "w") as f:
                    json.dump({
                        "body_part"  : bp,
                        "image_path" : img_path,
                        "true_label" : true_label,
                        "biomed_prob": prob,
                        "biomed_pred": pred_text,
                        "threshold"  : threshold,
                        "qwen"       : result,
                    }, f, indent=4)

                all_results.append({
                    "body_part"          : bp,
                    "true_label"         : true_label,
                    "biomed_prob"        : prob,
                    "qwen_detected"      :
                        result.get("fracture_detected"),
                    "qwen_conf"          :
                        result.get("confidence", 0),
                    "status"             :
                        result.get("status"),
                    "attn_quality"       :
                        result.get("attention_quality"),
                    "marker_bias"        :
                        result.get("marker_bias_detected"),
                    "correct_biomed"     : int(
                        (prob >= threshold)
                        == bool(true_label)),
                    "correct_qwen"       : int(
                        result.get("fracture_detected")
                        == bool(true_label))
                    if result.get("fracture_detected")
                    is not None else None,
                })

            except Exception as e:
                print(f"  ⚠️ Error: {e}")
                continue

    # ── Summary ───────────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 7 [v3] COMPLETE")
    print("="*60)

    if all_results:
        df    = pd.DataFrame(all_results)
        b_acc  = df["correct_biomed"].mean()
        q_acc  = df["correct_qwen"].dropna().mean()
        errors = (df["status"] == "error").sum()
        bone_f = (df["attn_quality"] ==
                  "bone-focused").sum()
        mrkr_f = (df["attn_quality"] ==
                  "marker-focused").sum()
        bias_c = df["marker_bias"].sum()

        print(f"\nTotal samples  : {len(df)}")
        print(f"BiomedCLIP acc : {b_acc*100:.1f}%")
        print(f"Qwen acc       : {q_acc*100:.1f}%")
        print(f"Errors         : {errors}/{len(df)}")
        print(f"Bone-focused   : {bone_f}/{len(df)}")
        print(f"Marker-focused : {mrkr_f}/{len(df)}")
        print(f"Marker bias    : {bias_c}/{len(df)} "
              f"images ⚠")

        print(f"\n{'Body Part':<15} "
              f"{'BiomedCLIP':>12} {'Qwen':>8}")
        print("-"*38)
        for bp in BODY_PARTS:
            rows = df[df["body_part"] == bp]
            if len(rows) == 0: continue
            b = rows["correct_biomed"].mean()
            q = rows["correct_qwen"].dropna().mean()
            qs = (f"{q*100:.0f}%"
                  if not np.isnan(q) else "error")
            print(f"  {bp:<13} {b*100:>10.0f}% {qs:>8}")

        csv_path = os.path.join(
            RESULTS_DIR, "qwen_results_v3.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nCSV : {csv_path}")

    print(f"\nReports : {RESULTS_DIR}")
    print("Next    : streamlit run app.py")