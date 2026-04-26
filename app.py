"""
SteelScan AI — DualPath-AFNet Surface Defect Detection
Streamlit Prototype Application
Devyani Ghildiyal | Manipal University Jaipur | 229301091
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import timm
import time
import io
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SteelScan AI | Surface Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Import fonts */
  @import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Mono:wght@400;500&display=swap');

  /* Main background */
  .stApp { background-color: #0d1117; }
  .main .block-container { padding-top: 2rem; max-width: 1200px; }

  /* Hide Streamlit default elements */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 16px !important;
  }
  [data-testid="metric-container"] label {
    color: #8b949e !important;
    font-size: 11px !important;
    font-family: 'DM Mono', monospace !important;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
  [data-testid="metric-container"] [data-testid="metric-value"] {
    color: #f0f6fc !important;
    font-size: 24px !important;
    font-weight: 700 !important;
  }

  /* Upload area */
  [data-testid="stFileUploader"] {
    background: #161b22;
    border: 1.5px dashed #30363d;
    border-radius: 12px;
    padding: 8px;
  }
  [data-testid="stFileUploader"]:hover {
    border-color: #4f9eff;
    background: rgba(79,158,255,0.04);
  }

  /* Buttons */
  .stButton > button {
    background: #4f9eff !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 12px 28px !important;
    width: 100% !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: #6aaeff !important;
    transform: translateY(-1px);
  }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background-color: #0d1117 !important;
    border-right: 1px solid #21262d;
  }
  [data-testid="stSidebar"] .block-container { padding: 1.5rem 1rem; }

  /* Progress bars */
  .stProgress > div > div {
    background: linear-gradient(90deg, #4f9eff, #2dd88a) !important;
    border-radius: 4px !important;
  }

  /* Expander */
  .streamlit-expanderHeader {
    background: #161b22 !important;
    border-radius: 8px !important;
    color: #8b949e !important;
    font-size: 13px !important;
  }

  /* Custom classes */
  .result-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
  }
  .result-header {
    font-family: 'Syne', sans-serif;
    font-size: 28px;
    font-weight: 800;
    color: #f0f6fc;
    text-transform: capitalize;
    margin-bottom: 6px;
  }
  .result-conf { color: #2dd88a; font-size: 14px; margin-bottom: 16px; }
  .section-title {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #484f58;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 12px;
  }
  .hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 36px;
    font-weight: 800;
    color: #f0f6fc;
    line-height: 1.1;
    margin-bottom: 8px;
  }
  .hero-sub { color: #8b949e; font-size: 15px; line-height: 1.6; margin-bottom: 24px; }
  .alpha-bar-cnn {
    height: 10px; border-radius: 5px 0 0 5px;
    background: #4f9eff; display: inline-block;
    vertical-align: middle; transition: width 0.5s;
  }
  .alpha-bar-trans {
    height: 10px; border-radius: 0 5px 5px 0;
    background: #7b5cf0; display: inline-block;
    vertical-align: middle; transition: width 0.5s;
  }
  .tag {
    display: inline-block;
    font-size: 11px; font-weight: 500;
    padding: 3px 10px; border-radius: 20px;
    font-family: 'DM Mono', monospace;
  }
  .tag-blue { background: rgba(79,158,255,0.12); color: #4f9eff; border: 1px solid rgba(79,158,255,0.25); }
  .tag-purple { background: rgba(123,92,240,0.12); color: #7b5cf0; border: 1px solid rgba(123,92,240,0.25); }
  .tag-green { background: rgba(45,216,138,0.12); color: #2dd88a; border: 1px solid rgba(45,216,138,0.25); }
  .tag-amber { background: rgba(245,166,35,0.12); color: #f5a623; border: 1px solid rgba(245,166,35,0.25); }
  .divider { border: none; border-top: 1px solid #21262d; margin: 20px 0; }
  .info-box {
    background: rgba(79,158,255,0.06);
    border: 1px solid rgba(79,158,255,0.18);
    border-radius: 10px; padding: 14px 16px;
    font-size: 13px; color: #8b949e; line-height: 1.6;
  }
  .warning-box {
    background: rgba(245,166,35,0.08);
    border: 1px solid rgba(245,166,35,0.2);
    border-radius: 10px; padding: 14px 16px;
    font-size: 13px; color: #f5a623; line-height: 1.6;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE (must match your training code exactly)
# ─────────────────────────────────────────────────────────────
class ChannelPyramidPooling(nn.Module):
    def __init__(self, in_channels_list, out_dim=256):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(c, out_dim // 3, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_dim // 3),
                nn.GELU()
            ) for c in in_channels_list
        ])
        self.pool  = nn.AdaptiveAvgPool2d((4, 4))
        fused_dim  = (out_dim // 3) * 3 * 4 * 4
        self.fc    = nn.Sequential(nn.Linear(fused_dim, out_dim), nn.GELU(), nn.Dropout(0.2))
        self.out_dim = out_dim

    def forward(self, features_list):
        pooled = []
        for proj, feat in zip(self.projections, features_list):
            x = proj(feat); x = self.pool(x); pooled.append(x.flatten(1))
        return self.fc(torch.cat(pooled, dim=1))


class GlobalContextTransformerBranch(nn.Module):
    def __init__(self, in_channels=320, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.d_model    = d_model
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_channels, d_model, kernel_size=1, bias=False),
            nn.BatchNorm2d(d_model), nn.GELU()
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 50, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        encoder_layer  = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, x):
        B   = x.size(0)
        x   = self.input_proj(x)
        x   = x.flatten(2).permute(0, 2, 1)
        cls = self.cls_token.expand(B, -1, -1)
        x   = torch.cat([cls, x], dim=1)
        x   = x + self.pos_embed
        x   = self.transformer(x)
        return self.norm(x)[:, 0, :]


class AdaptiveFusionGate(nn.Module):
    def __init__(self, cnn_dim, trans_dim, shared_dim=256):
        super().__init__()
        self.cnn_proj   = nn.Linear(cnn_dim, shared_dim)
        self.trans_proj = nn.Linear(trans_dim, shared_dim)
        combined_dim    = cnn_dim + trans_dim
        self.gate_mlp   = nn.Sequential(
            nn.Linear(combined_dim, combined_dim // 4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(combined_dim // 4, 1), nn.Sigmoid()
        )

    def forward(self, cnn_feat, trans_feat):
        combined = torch.cat([cnn_feat, trans_feat], dim=1)
        alpha    = self.gate_mlp(combined)
        cnn_p    = self.cnn_proj(cnn_feat)
        trans_p  = self.trans_proj(trans_feat)
        fused    = alpha * cnn_p + (1 - alpha) * trans_p
        return fused, alpha


class DualPathAFNet(nn.Module):
    def __init__(self, num_classes=6, cnn_out_dim=256, trans_d_model=128,
                 shared_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.stem = timm.create_model('mobilenetv2_100', pretrained=False, features_only=True)
        self.mscb = ChannelPyramidPooling(in_channels_list=[24, 96, 320], out_dim=cnn_out_dim)
        self.gctb = GlobalContextTransformerBranch(
            in_channels=320, d_model=trans_d_model, nhead=4, num_layers=num_layers, dropout=0.1
        )
        self.afg  = AdaptiveFusionGate(cnn_dim=cnn_out_dim, trans_dim=trans_d_model, shared_dim=shared_dim)
        classifier_in = shared_dim + trans_d_model
        self.classifier = nn.Sequential(
            nn.LayerNorm(classifier_in), nn.Dropout(dropout),
            nn.Linear(classifier_in, 256), nn.GELU(),
            nn.Dropout(dropout / 2), nn.Linear(256, num_classes)
        )

    def forward(self, x, return_alpha=False):
        feats      = self.stem(x)
        s1, s2, s3 = feats[1], feats[3], feats[4]
        cnn_feat   = self.mscb([s1, s2, s3])
        trans_feat = self.gctb(s3)
        fused, alpha = self.afg(cnn_feat, trans_feat)
        combined   = torch.cat([fused, trans_feat], dim=1)
        out        = self.classifier(combined)
        if return_alpha:
            return out, alpha
        return out


# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']

CLASS_INFO = {
    'crazing':        {'desc': 'Network of fine surface cracks, like dried mud', 'color': '#4f9eff'},
    'inclusion':      {'desc': 'Small foreign particles embedded in the steel surface', 'color': '#7b5cf0'},
    'patches':        {'desc': 'Irregular rough areas of uneven surface texture', 'color': '#2dd88a'},
    'pitted_surface': {'desc': 'Small holes or cavity-like indentations on surface', 'color': '#f5a623'},
    'rolled-in_scale':{'desc': 'Steel flakes pressed into surface during rolling process', 'color': '#ff5f5f'},
    'scratches':      {'desc': 'Linear score marks running across the surface', 'color': '#e879f9'},
}

# Known alpha values from your research (mean gate values per class)
KNOWN_ALPHA = {
    'crazing':         0.971,
    'inclusion':       0.343,
    'patches':         0.975,
    'pitted_surface':  0.821,
    'rolled-in_scale': 0.895,
    'scratches':       0.615,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

MODEL_PATH = "DualPathAFNet_v2_best.pth"   # ← put your checkpoint here


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DualPathAFNet(
        num_classes=6, cnn_out_dim=256, trans_d_model=128,
        shared_dim=256, num_layers=2, dropout=0.3
    ).to(device)

    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state)
        model_loaded = True
    else:
        model_loaded = False   # demo mode — random weights

    model.eval()
    return model, device, model_loaded


# ─────────────────────────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────────────────────────
def preprocess(image: Image.Image):
    img = image.convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    return tensor


# ─────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────
def predict(model, device, image: Image.Image):
    tensor = preprocess(image).to(device)

    if device.type == "cuda":
        # warm-up
        with torch.no_grad():
            for _ in range(3):
                model(tensor)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, alpha = model(tensor, return_alpha=True)
        torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - t0) * 1000
    else:
        t0 = time.perf_counter()
        with torch.no_grad():
            logits, alpha = model(tensor, return_alpha=True)
        latency_ms = (time.perf_counter() - t0) * 1000

    probs       = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    alpha_val   = float(alpha.squeeze().cpu().item())
    pred_idx    = int(np.argmax(probs))
    pred_class  = CLASS_NAMES[pred_idx]
    confidence  = float(probs[pred_idx])

    return pred_class, confidence, probs, alpha_val, latency_ms


# ─────────────────────────────────────────────────────────────
# ALPHA GATE CHART
# ─────────────────────────────────────────────────────────────
def make_alpha_chart(predicted_class, alpha_val):
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    classes    = list(KNOWN_ALPHA.keys())
    alphas     = list(KNOWN_ALPHA.values())
    colors     = ['#4f9eff' if a > 0.5 else '#7b5cf0' for a in alphas]
    edge_colors = ['#f0f6fc' if c == predicted_class else 'none' for c in classes]
    edge_widths = [2 if c == predicted_class else 0 for c in classes]

    bars = ax.bar(classes, alphas, color=colors, edgecolor=edge_colors,
                  linewidth=edge_widths, zorder=3, width=0.6)
    ax.axhline(0.5, color='#30363d', linestyle='--', lw=1.2, zorder=2)
    ax.axhline(alpha_val, color='#f5a623', linestyle='-', lw=1.5,
               zorder=4, alpha=0.7)

    ax.set_ylim(0, 1.05)
    ax.set_ylabel('α value', color='#8b949e', fontsize=10)
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.set_xticklabels([c.replace('_', '\n').replace('-in', '\n-in') for c in classes],
                       fontsize=8, color='#8b949e')
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')

    p1 = mpatches.Patch(color='#4f9eff', label='CNN-dominant (α > 0.5)')
    p2 = mpatches.Patch(color='#7b5cf0', label='Transformer-dominant (α < 0.5)')
    p3 = mpatches.Patch(color='#f5a623', label=f'Current prediction α={alpha_val:.3f}')
    ax.legend(handles=[p1, p2, p3], fontsize=8, facecolor='#161b22',
              edgecolor='#30363d', labelcolor='#8b949e', loc='upper right')
    ax.set_title('Adaptive Fusion Gate — per-class α values', color='#8b949e',
                 fontsize=10, pad=10)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight',
                facecolor='#161b22')
    buf.seek(0)
    plt.close()
    return buf


# ─────────────────────────────────────────────────────────────
# CONFIDENCE CHART
# ─────────────────────────────────────────────────────────────
def make_confidence_chart(probs):
    fig, ax = plt.subplots(figsize=(7, 3))
    fig.patch.set_facecolor('#161b22')
    ax.set_facecolor('#161b22')

    sorted_idx    = np.argsort(probs)[::-1]
    sorted_names  = [CLASS_NAMES[i].replace('_', ' ').replace('-in', '\n-in') for i in sorted_idx]
    sorted_probs  = [probs[i] for i in sorted_idx]
    bar_colors    = ['#4f9eff' if i == 0 else '#30363d' for i in range(len(sorted_probs))]

    bars = ax.barh(sorted_names[::-1], sorted_probs[::-1],
                   color=bar_colors[::-1], height=0.6, zorder=3)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel('Confidence', color='#8b949e', fontsize=10)
    ax.tick_params(colors='#8b949e', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    for bar, prob in zip(bars, sorted_probs[::-1]):
        ax.text(min(prob + 0.02, 1.0), bar.get_y() + bar.get_height()/2,
                f'{prob:.1%}', va='center', color='#8b949e', fontsize=9)
    ax.set_title('Classification confidence', color='#8b949e', fontsize=10, pad=10)
    ax.grid(axis='x', color='#21262d', lw=0.7, zorder=1)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor='#161b22')
    buf.seek(0)
    plt.close()
    return buf


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center;margin-bottom:24px'>
          <div style='font-family:Syne,sans-serif;font-size:20px;font-weight:800;
                      color:#f0f6fc;margin-bottom:4px'>SteelScan AI</div>
          <div style='font-size:11px;color:#484f58;font-family:DM Mono,monospace;
                      letter-spacing:0.1em'>SURFACE DEFECT DETECTION</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-title">Model</div>', unsafe_allow_html=True)
        st.markdown('<span class="tag tag-green">DualPath-AFNet v2</span>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "89.26%")
            st.metric("FPS", "131")
        with col2:
            st.metric("F1 Score", "0.890")
            st.metric("Size", "13.9 MB")

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:12px;color:#8b949e;line-height:2;'>
          <span class='tag tag-blue'>CNN Branch</span> Multi-Scale Pyramid<br>
          <span class='tag tag-purple'>Transformer</span> CLS Token + Self-Attention<br>
          <span class='tag tag-green'>AFG Gate</span> Adaptive α per sample<br>
          <span style='color:#484f58'>Backbone: MobileNetV2 (pretrained)</span><br>
          <span style='color:#484f58'>Params: 3.58M · Layers: 2 Transformer</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        st.markdown('<div class="section-title">Defect Classes</div>', unsafe_allow_html=True)
        for cls, info in CLASS_INFO.items():
            st.markdown(f"""
            <div style='display:flex;gap:8px;align-items:flex-start;
                        margin-bottom:10px;padding-bottom:10px;
                        border-bottom:1px solid #21262d'>
              <div style='width:3px;height:32px;border-radius:2px;
                          background:{info["color"]};flex-shrink:0;margin-top:2px'></div>
              <div>
                <div style='font-size:12px;font-weight:500;color:#f0f6fc;
                            text-transform:capitalize;margin-bottom:2px'>
                  {cls.replace("_", " ").replace("-in", "-in ")}
                </div>
                <div style='font-size:11px;color:#484f58;line-height:1.4'>{info["desc"]}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:11px;color:#484f58;text-align:center;line-height:1.6'>
          Devyani Ghildiyal · Reg. 229301091<br>
          Manipal University Jaipur<br>
          Guide: Dr. Anil Kumar
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    # ── Hero ──────────────────────────────────────────────────
    st.markdown("""
    <div style='margin-bottom:32px'>
      <div style='font-family:DM Mono,monospace;font-size:11px;color:#4f9eff;
                  letter-spacing:0.15em;margin-bottom:12px'>
        RESEARCH PROTOTYPE · NEU SURFACE DEFECT DATASET
      </div>
      <div class='hero-title'>Steel Surface<br><span style='color:#4f9eff'>Defect Detection</span></div>
      <div class='hero-sub'>
        Upload a steel surface image and DualPath-AFNet will classify the defect type,
        show per-class confidence scores, and reveal which branch — CNN or Transformer —
        the Adaptive Fusion Gate relied on for this specific image.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load model ────────────────────────────────────────────
    with st.spinner("Loading DualPath-AFNet..."):
        model, device, model_loaded = load_model()

    if not model_loaded:
        st.markdown("""
        <div class='warning-box'>
          ⚠️ <strong>Demo mode:</strong> Model checkpoint <code>DualPathAFNet_v2_best.pth</code>
          not found. Running with random weights — predictions will not be meaningful.
          Place your trained checkpoint in the same folder as this app and restart.
        </div>
        """, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='info-box'>
          ✅ <strong>Model loaded successfully.</strong>
          DualPath-AFNet v2 · 3.58M parameters · Running on
        """ + f"{'GPU (CUDA)' if device.type == 'cuda' else 'CPU'}</div>",
        unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # ── Upload + Results ──────────────────────────────────────
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="section-title">Upload Image</div>', unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Drop a steel surface image here",
            type=["png", "jpg", "jpeg", "bmp", "tiff"],
            label_visibility="collapsed"
        )

        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_column_width=True)

            # Image metadata
            w, h = image.size
            st.markdown(f"""
            <div style='display:flex;gap:8px;margin-top:8px;flex-wrap:wrap'>
              <span class='tag tag-blue'>{w}×{h}px</span>
              <span class='tag tag-blue'>{uploaded.type}</span>
              <span class='tag tag-blue'>{uploaded.size/1024:.1f} KB</span>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            analyse_btn = st.button("🔬  Analyse Defect", key="analyse")

        else:
            st.markdown("""
            <div style='background:#161b22;border:1.5px dashed #30363d;
                        border-radius:12px;padding:60px 24px;text-align:center'>
              <div style='font-size:32px;margin-bottom:12px;opacity:0.3'>🔬</div>
              <div style='font-size:14px;color:#484f58'>
                Upload a steel surface image to begin analysis
              </div>
              <div style='font-size:12px;color:#30363d;margin-top:6px'>
                PNG, JPG, BMP, TIFF supported
              </div>
            </div>
            """, unsafe_allow_html=True)
            analyse_btn = False

    with col_right:
        st.markdown('<div class="section-title">Analysis Results</div>', unsafe_allow_html=True)

        if uploaded and analyse_btn:
            with st.spinner("Analysing..."):
                pred_class, confidence, probs, alpha_val, latency_ms = predict(
                    model, device, image
                )

            # ── Prediction result ──────────────────────────
            cls_color = CLASS_INFO[pred_class]['color']
            cls_desc  = CLASS_INFO[pred_class]['desc']

            st.markdown(f"""
            <div class='result-card'>
              <div style='font-size:11px;color:#484f58;font-family:DM Mono,monospace;
                          letter-spacing:.1em;margin-bottom:8px'>PREDICTED CLASS</div>
              <div class='result-header' style='color:{cls_color}'>
                {pred_class.replace("_", " ").replace("-in", "-in ")}
              </div>
              <div class='result-conf'>Confidence: {confidence:.1%}</div>
              <div style='background:#0d1117;border-radius:6px;height:6px;overflow:hidden;margin-bottom:16px'>
                <div style='height:100%;width:{confidence*100:.1f}%;
                            background:linear-gradient(90deg,{cls_color},#2dd88a);
                            border-radius:6px;transition:width .8s'></div>
              </div>
              <div style='font-size:13px;color:#8b949e;line-height:1.6'>{cls_desc}</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Stats row ──────────────────────────────────
            m1, m2, m3 = st.columns(3)
            with m1: st.metric("Confidence", f"{confidence:.1%}")
            with m2: st.metric("Latency", f"{latency_ms:.1f} ms")
            with m3: st.metric("Gate α", f"{alpha_val:.3f}")

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Confidence chart ───────────────────────────
            st.markdown('<div class="section-title">All Class Scores</div>', unsafe_allow_html=True)
            conf_buf = make_confidence_chart(probs)
            st.image(conf_buf, use_column_width=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Alpha gate explanation ─────────────────────
            st.markdown('<div class="section-title">Adaptive Fusion Gate Analysis</div>',
                        unsafe_allow_html=True)

            cnn_pct   = alpha_val * 100
            trans_pct = (1 - alpha_val) * 100
            branch    = "CNN branch" if alpha_val > 0.5 else "Transformer branch"
            branch_tag = "tag-blue" if alpha_val > 0.5 else "tag-purple"

            st.markdown(f"""
            <div class='result-card'>
              <div style='display:flex;gap:8px;align-items:center;margin-bottom:12px'>
                <span class='tag {branch_tag}'>{branch} dominant</span>
                <span style='font-size:12px;color:#484f58'>α = {alpha_val:.3f}</span>
              </div>
              <div style='display:flex;height:10px;border-radius:5px;overflow:hidden;
                          margin-bottom:8px;background:#0d1117'>
                <div style='width:{cnn_pct:.1f}%;background:#4f9eff;height:100%;
                            border-radius:5px 0 0 5px'></div>
                <div style='width:{trans_pct:.1f}%;background:#7b5cf0;height:100%;
                            border-radius:0 5px 5px 0'></div>
              </div>
              <div style='display:flex;justify-content:space-between;
                          font-size:11px;margin-bottom:14px;font-family:DM Mono,monospace'>
                <span style='color:#4f9eff'>CNN {cnn_pct:.0f}%</span>
                <span style='color:#7b5cf0'>Transformer {trans_pct:.0f}%</span>
              </div>
              <div style='font-size:12px;color:#8b949e;line-height:1.7;
                          padding-top:12px;border-top:1px solid #21262d'>
                {'The model is primarily using <strong style="color:#4f9eff">local texture features</strong> from the CNN branch. This is typical for defects with strong distinctive local patterns.' if alpha_val > 0.5
                 else 'The model is primarily using <strong style="color:#7b5cf0">global spatial context</strong> from the Transformer branch. This means the defect requires understanding how features are distributed across the whole image — characteristic of inclusion defects.'}
              </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Alpha chart ────────────────────────────────
            with st.expander("View full per-class gate analysis chart"):
                alpha_buf = make_alpha_chart(pred_class, alpha_val)
                st.image(alpha_buf, use_column_width=True)
                st.markdown("""
                <div style='font-size:12px;color:#484f58;margin-top:8px;line-height:1.6'>
                  Orange line = gate value for this prediction.
                  Highlighted bar = predicted class.
                  Inclusion (α=0.343) is the only class where the Transformer dominates —
                  this is the novel finding of DualPath-AFNet.
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div style='background:#161b22;border:1px solid #21262d;
                        border-radius:16px;padding:60px 24px;text-align:center;height:100%'>
              <div style='font-size:32px;margin-bottom:12px;opacity:0.2'>📊</div>
              <div style='font-size:14px;color:#484f58;line-height:1.6'>
                Results will appear here after analysis.<br>
                Upload an image and click Analyse.
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── About section ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("About DualPath-AFNet — how it works"):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("""
            <div class='result-card'>
              <span class='tag tag-blue'>CNN Branch</span>
              <div style='margin-top:12px;font-size:13px;color:#8b949e;line-height:1.7'>
                The Multi-Scale CNN Branch (MSCB) uses MobileNetV2 features at 3 different
                resolutions (56×56, 14×14, 7×7) and pools them into a 256-dim local feature
                vector. It captures textures, edges, and fine-grain patterns.
              </div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class='result-card'>
              <span class='tag tag-purple'>Transformer Branch</span>
              <div style='margin-top:12px;font-size:13px;color:#8b949e;line-height:1.7'>
                The Global Context Transformer Branch (GCTB) treats the 7×7 feature map as
                49 tokens, prepends a CLS token, and runs 2 self-attention layers. It captures
                long-range spatial relationships across the whole image.
              </div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown("""
            <div class='result-card'>
              <span class='tag tag-green'>Adaptive Fusion Gate</span>
              <div style='margin-top:12px;font-size:13px;color:#8b949e;line-height:1.7'>
                The AFG computes a per-sample scalar α ∈ (0,1) that dynamically weights
                the two branches. Inclusion defects (α=0.343) are Transformer-dominant;
                texture defects like patches (α=0.975) are CNN-dominant.
              </div>
            </div>
            """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
