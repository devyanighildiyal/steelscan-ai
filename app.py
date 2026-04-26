"""
SteelScan — DualPath-AFNet Surface Defect Detection
Redesigned Streamlit Prototype
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

st.set_page_config(
    page_title="SteelScan — Surface Defect Detection",
    page_icon="🔩",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background-color: #0f1117; }
.main .block-container { padding-top: 2.5rem; max-width: 1100px; }
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSidebar"] {
    background-color: #090b0f !important;
    border-right: 1px solid #1e2330;
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.2rem 1rem; }

[data-testid="metric-container"] {
    background: #13161f; border: 1px solid #1e2330; border-radius: 10px; padding: 14px !important;
}
[data-testid="metric-container"] label {
    color: #5a6478 !important; font-size: 10px !important;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 0.1em; text-transform: uppercase;
}
[data-testid="metric-container"] [data-testid="metric-value"] {
    color: #e8ecf4 !important; font-size: 22px !important; font-weight: 600 !important;
}

[data-testid="stFileUploader"] {
    background: #13161f; border: 1.5px dashed #2a3040; border-radius: 12px; padding: 6px;
}

.stButton > button {
    background: #3d7aed !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important; font-weight: 600 !important;
    font-size: 14px !important; padding: 10px 24px !important; width: 100% !important;
}
.stButton > button:hover { background: #5490f5 !important; }

.streamlit-expanderHeader {
    background: #13161f !important; border: 1px solid #1e2330 !important;
    border-radius: 8px !important; color: #8892a4 !important;
    font-family: 'Inter', sans-serif !important; font-size: 13px !important; font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: #13161f !important; border: 1px solid #1e2330 !important;
    border-top: none !important; border-radius: 0 0 8px 8px !important;
}

.app-title {
    font-family: 'Inter', sans-serif; font-size: 44px; font-weight: 700;
    color: #e8ecf4; letter-spacing: -0.03em; line-height: 1.1; margin-bottom: 8px;
}
.app-subtitle {
    font-size: 13px; color: #5a6478; line-height: 1.7;
    max-width: 660px; margin-bottom: 28px; font-style: italic;
}
.section-label {
    font-family: 'JetBrains Mono', monospace; font-size: 10px; color: #3d4658;
    letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 10px;
}
.card {
    background: #13161f; border: 1px solid #1e2330;
    border-radius: 12px; padding: 20px 22px; margin-bottom: 14px;
}
.result-class {
    font-family: 'Inter', sans-serif; font-size: 30px; font-weight: 700;
    letter-spacing: -0.02em; text-transform: capitalize; margin-bottom: 4px;
}
.result-conf { font-size: 13px; color: #22c55e; margin-bottom: 14px; font-weight: 500; }
.mono { font-family: 'JetBrains Mono', monospace; font-size: 11px; }
.tag {
    display: inline-block; font-family: 'JetBrains Mono', monospace;
    font-size: 10px; font-weight: 500; padding: 3px 8px;
    border-radius: 4px; letter-spacing: 0.04em;
}
.tag-blue   { background: rgba(61,122,237,0.12); color: #5b9cf6; border: 1px solid rgba(61,122,237,0.25); }
.tag-purple { background: rgba(139,92,246,0.12); color: #a78bfa; border: 1px solid rgba(139,92,246,0.25); }
.tag-green  { background: rgba(34,197,94,0.12);  color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
.tag-gray   { background: rgba(90,100,120,0.12); color: #8892a4; border: 1px solid rgba(90,100,120,0.2); }
.info-box {
    background: rgba(61,122,237,0.07); border: 1px solid rgba(61,122,237,0.2);
    border-radius: 8px; padding: 12px 14px; font-size: 13px; color: #8892a4; line-height: 1.6;
}
.warn-box {
    background: rgba(245,158,11,0.07); border: 1px solid rgba(245,158,11,0.2);
    border-radius: 8px; padding: 12px 14px; font-size: 13px; color: #fbbf24; line-height: 1.6;
}
.divider { border: none; border-top: 1px solid #1e2330; margin: 18px 0; }
.arch-step {
    display: flex; align-items: flex-start; gap: 10px; padding: 9px 10px;
    border-radius: 7px; border: 1px solid #1e2330; margin-bottom: 6px; background: #0d1017;
}
.arch-dot { width: 7px; height: 7px; border-radius: 50%; margin-top: 4px; flex-shrink: 0; }
.arch-text { font-size: 12px; color: #8892a4; line-height: 1.5; }
.arch-text b { color: #c4cad6; font-weight: 500; }
.empty-state {
    background: #13161f; border: 1px solid #1e2330;
    border-radius: 12px; padding: 60px 24px; text-align: center;
}
</style>
""", unsafe_allow_html=True)


# ── Model architecture ─────────────────────────────────────────────────────────
class ChannelPyramidPooling(nn.Module):
    def __init__(self, in_channels_list, out_dim=256):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, out_dim//3, 1, bias=False), nn.BatchNorm2d(out_dim//3), nn.GELU())
            for c in in_channels_list
        ])
        self.pool  = nn.AdaptiveAvgPool2d((4,4))
        self.fc    = nn.Sequential(nn.Linear((out_dim//3)*3*16, out_dim), nn.GELU(), nn.Dropout(0.2))

    def forward(self, fl):
        pooled = [self.pool(p(f)).flatten(1) for p,f in zip(self.projections,fl)]
        return self.fc(torch.cat(pooled,1))


class GlobalContextTransformerBranch(nn.Module):
    def __init__(self, in_ch=320, d=128, heads=4, layers=2, drop=0.1):
        super().__init__()
        self.proj      = nn.Sequential(nn.Conv2d(in_ch,d,1,bias=False), nn.BatchNorm2d(d), nn.GELU())
        self.cls       = nn.Parameter(torch.zeros(1,1,d))
        self.pos       = nn.Parameter(torch.zeros(1,50,d))
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        enc            = nn.TransformerEncoderLayer(d, heads, d*4, drop, 'gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, layers)
        self.norm      = nn.LayerNorm(d)

    def forward(self, x):
        B = x.size(0)
        x = self.proj(x).flatten(2).permute(0,2,1)
        x = torch.cat([self.cls.expand(B,-1,-1), x], 1) + self.pos
        return self.norm(self.transformer(x))[:,0,:]


class AdaptiveFusionGate(nn.Module):
    def __init__(self, cd, td, sd=256):
        super().__init__()
        self.cp = nn.Linear(cd, sd); self.tp = nn.Linear(td, sd)
        self.gate = nn.Sequential(nn.Linear(cd+td,(cd+td)//4), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear((cd+td)//4,1), nn.Sigmoid())

    def forward(self, cf, tf):
        a = self.gate(torch.cat([cf,tf],1))
        return a*self.cp(cf)+(1-a)*self.tp(tf), a


class DualPathAFNet(nn.Module):
    def __init__(self, nc=6, cd=256, td=128, sd=256, nl=2, drop=0.3):
        super().__init__()
        self.stem = timm.create_model('mobilenetv2_100', pretrained=False, features_only=True)
        self.mscb = ChannelPyramidPooling([24,96,320], cd)
        self.gctb = GlobalContextTransformerBranch(320, td, 4, nl, 0.1)
        self.afg  = AdaptiveFusionGate(cd, td, sd)
        self.head = nn.Sequential(
            nn.LayerNorm(sd+td), nn.Dropout(drop),
            nn.Linear(sd+td,256), nn.GELU(), nn.Dropout(drop/2), nn.Linear(256,nc)
        )

    def forward(self, x, return_alpha=False):
        f = self.stem(x)
        cf = self.mscb([f[1],f[3],f[4]]); tf = self.gctb(f[4])
        fused, a = self.afg(cf, tf)
        out = self.head(torch.cat([fused,tf],1))
        return (out, a) if return_alpha else out


# ── Constants ─────────────────────────────────────────────────────────────────
CLASS_NAMES = ['crazing','inclusion','patches','pitted_surface','rolled-in_scale','scratches']
CLASS_INFO  = {
    'crazing':         {'desc':'Network of fine surface cracks, like dried mud',          'color':'#5b9cf6'},
    'inclusion':       {'desc':'Small foreign particles embedded in the steel surface',    'color':'#a78bfa'},
    'patches':         {'desc':'Irregular rough areas of uneven surface texture',          'color':'#4ade80'},
    'pitted_surface':  {'desc':'Small holes or cavity-like indentations on surface',       'color':'#fbbf24'},
    'rolled-in_scale': {'desc':'Steel flakes pressed into surface during rolling',         'color':'#f87171'},
    'scratches':       {'desc':'Linear score marks running across the surface',            'color':'#e879f9'},
}
KNOWN_ALPHA = {'crazing':0.971,'inclusion':0.343,'patches':0.975,
               'pitted_surface':0.821,'rolled-in_scale':0.895,'scratches':0.615}
MODEL_PATH  = "DualPathAFNet_v2_best.pth"


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = DualPathAFNet().to(device)
    loaded = False
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        loaded = True
    model.eval()
    return model, device, loaded


# ── Inference ─────────────────────────────────────────────────────────────────
def preprocess(img):
    img = img.convert("RGB").resize((224,224), Image.BILINEAR)
    arr = (np.array(img, dtype=np.float32)/255.0 - [0.485,0.456,0.406]) / [0.229,0.224,0.225]
    return torch.from_numpy(arr).permute(2,0,1).unsqueeze(0).float()

def predict(model, device, image):
    t = preprocess(image).to(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        logits, alpha = model(t, return_alpha=True)
    lat   = (time.perf_counter()-t0)*1000
    probs = torch.softmax(logits,1).squeeze().cpu().numpy()
    a     = float(alpha.squeeze().cpu().item())
    idx   = int(np.argmax(probs))
    return CLASS_NAMES[idx], float(probs[idx]), probs, a, lat


# ── Charts ────────────────────────────────────────────────────────────────────
BG = '#13161f'

def conf_chart(probs):
    fig, ax = plt.subplots(figsize=(6,2.8)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    idx   = np.argsort(probs)
    names = [CLASS_NAMES[i].replace('_',' ').replace('-in','\n-in') for i in idx]
    vals  = [probs[i] for i in idx]
    cols  = ['#3d7aed' if i==np.argmax(probs) else '#1e2330' for i in idx]
    ax.barh(names, vals, color=cols, height=0.55, zorder=3)
    ax.set_xlim(0, 1.08); ax.set_xlabel('Confidence', color='#5a6478', fontsize=9)
    ax.tick_params(colors='#5a6478', labelsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor('#1e2330')
    for bar, v in zip(ax.patches, vals):
        ax.text(v+0.02, bar.get_y()+bar.get_height()/2, f'{v:.1%}', va='center', color='#5a6478', fontsize=9)
    ax.grid(axis='x', color='#1e2330', lw=0.7, zorder=1)
    plt.tight_layout(); buf=io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=BG)
    buf.seek(0); plt.close(); return buf

def alpha_chart(pred_class, alpha_val):
    fig, ax = plt.subplots(figsize=(6,2.8)); fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    classes = list(KNOWN_ALPHA.keys()); alphas = list(KNOWN_ALPHA.values())
    cols  = ['#3d7aed' if a>0.5 else '#8b5cf6' for a in alphas]
    edges = ['#e8ecf4' if c==pred_class else 'none' for c in classes]
    lws   = [2 if c==pred_class else 0 for c in classes]
    ax.bar(range(len(classes)), alphas, color=cols, edgecolor=edges, linewidth=lws, zorder=3, width=0.55)
    ax.axhline(0.5, color='#2a3040', ls='--', lw=1.2, zorder=2)
    ax.axhline(alpha_val, color='#fbbf24', ls='-', lw=1.5, zorder=4, alpha=0.8)
    ax.set_xticks(range(len(classes)))
    ax.set_xticklabels([c.replace('_','\n').replace('-in','\n-in') for c in classes], fontsize=7.5, color='#5a6478')
    ax.set_ylim(0,1.08); ax.set_ylabel('α', color='#5a6478', fontsize=9)
    ax.tick_params(colors='#5a6478', labelsize=9)
    for spine in ax.spines.values(): spine.set_edgecolor('#1e2330')
    p1=mpatches.Patch(color='#3d7aed',label='CNN-dominant')
    p2=mpatches.Patch(color='#8b5cf6',label='Transformer-dominant')
    p3=mpatches.Patch(color='#fbbf24',label=f'This image α={alpha_val:.3f}')
    ax.legend(handles=[p1,p2,p3], fontsize=7.5, facecolor=BG, edgecolor='#1e2330',
              labelcolor='#8892a4', loc='upper right')
    plt.tight_layout(); buf=io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=BG)
    buf.seek(0); plt.close(); return buf


# ── Sidebar ───────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="font-family:'Inter',sans-serif;font-size:24px;font-weight:700;
                    color:#e8ecf4;letter-spacing:-0.02em;margin-bottom:2px">SteelScan</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:9px;color:#3d4658;
                    letter-spacing:0.14em;text-transform:uppercase;margin-bottom:10px">
          Research Prototype · NEU Surface Defect Dataset
        </div>
        <div style="font-size:11px;color:#3d4658;line-height:1.6;
                    margin-bottom:20px;font-style:italic;border-left:2px solid #1e2330;padding-left:10px">
          Deep Learning Models for Real-Time Industrial Surface Defect Detection
        </div>
        <hr style="border:none;border-top:1px solid #1e2330;margin:16px 0">
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label">Model performance</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1: st.metric("Accuracy","89.26%"); st.metric("FPS","131")
        with c2: st.metric("F1 Score","0.890");  st.metric("Size","13.9 MB")

        st.markdown('<hr style="border:none;border-top:1px solid #1e2330;margin:16px 0">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Architecture</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="arch-step"><div class="arch-dot" style="background:#3d7aed"></div>
          <div class="arch-text"><b>CNN Branch</b> — Multi-Scale Pyramid Pooling</div></div>
        <div class="arch-step"><div class="arch-dot" style="background:#8b5cf6"></div>
          <div class="arch-text"><b>Transformer Branch</b> — CLS Token + Self-Attention</div></div>
        <div class="arch-step"><div class="arch-dot" style="background:#22c55e"></div>
          <div class="arch-text"><b>Adaptive Gate (AFG)</b> — Learnable α per sample</div></div>
        <div style="font-size:11px;color:#3d4658;margin-top:8px;line-height:1.7">
          Backbone: MobileNetV2 (pretrained)<br>Params: 3.58M · Layers: 2 Transformer
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<hr style="border:none;border-top:1px solid #1e2330;margin:16px 0">', unsafe_allow_html=True)
        st.markdown('<div class="section-label">Defect classes</div>', unsafe_allow_html=True)
        for cls, info in CLASS_INFO.items():
            st.markdown(f"""
            <div style="display:flex;gap:8px;align-items:flex-start;margin-bottom:9px;
                        padding-bottom:9px;border-bottom:1px solid #1a1e28">
              <div style="width:3px;min-height:34px;border-radius:2px;
                          background:{info['color']};flex-shrink:0;margin-top:2px"></div>
              <div>
                <div style="font-size:12px;font-weight:500;color:#c4cad6;
                            text-transform:capitalize;margin-bottom:1px">
                  {cls.replace('_',' ')}
                </div>
                <div style="font-size:11px;color:#3d4658;line-height:1.4">{info['desc']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <hr style="border:none;border-top:1px solid #1e2330;margin:16px 0">
        <div style="font-size:11px;color:#3d4658;line-height:1.7">
          Devyani Ghildiyal · Reg. 229301091<br>Manipal University Jaipur<br>Guide: Dr. Anil Kumar
        </div>
        """, unsafe_allow_html=True)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    sidebar()

    # Title
    st.markdown("""
    <div class="app-title">SteelScan</div>
    <div class="app-subtitle">
      Prototype for the research paper —
      <em>"Deep Learning Models for Real-Time Industrial Surface Defect Detection"</em>
    </div>
    """, unsafe_allow_html=True)

    # About section (at top)
    with st.expander("About DualPath-AFNet — how this model works", expanded=False):
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("""<div class="card">
              <span class="tag tag-blue">CNN Branch</span>
              <div style="margin-top:10px;font-size:13px;color:#8892a4;line-height:1.7">
                Uses MobileNetV2 features at 3 scales (56×56, 14×14, 7×7) and pools them
                into a 256-dim local feature vector. Captures textures, edges, and fine-grain patterns.
              </div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="card">
              <span class="tag tag-purple">Transformer Branch</span>
              <div style="margin-top:10px;font-size:13px;color:#8892a4;line-height:1.7">
                Treats the 7×7 feature map as 49 tokens, prepends a CLS token, and runs
                2 self-attention layers. Captures long-range spatial relationships across the image.
              </div></div>""", unsafe_allow_html=True)
        with c3:
            st.markdown("""<div class="card">
              <span class="tag tag-green">Adaptive Fusion Gate</span>
              <div style="margin-top:10px;font-size:13px;color:#8892a4;line-height:1.7">
                Computes a per-sample scalar α. Inclusion (α=0.343) is Transformer-dominant.
                Texture defects like patches (α=0.975) are CNN-dominant — learned automatically.
              </div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Model status
    with st.spinner("Loading DualPath-AFNet..."):
        model, device, loaded = load_model()

    dlabel = "GPU (CUDA)" if device.type=="cuda" else "CPU"
    dnote  = "" if device.type=="cuda" else \
             " — Expected for web deployment. GPU is used during training on Kaggle."

    if loaded:
        st.markdown(f"""<div class="info-box">
          ✅ <strong>DualPath-AFNet v2 loaded</strong> · 3.58M parameters ·
          Running on <strong>{dlabel}</strong>{dnote}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="warn-box">
          ⚠️ <strong>Demo mode</strong> — <code>DualPathAFNet_v2_best.pth</code> not found.
          Place it in the same folder as app.py. Running on {dlabel}.
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload + Results
    col_l, col_r = st.columns([1,1], gap="large")

    with col_l:
        st.markdown('<div class="section-label">Upload image</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload", type=["png","jpg","jpeg","bmp","tiff","tif"],
                                    label_visibility="collapsed")
        if uploaded:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded image", use_column_width=True)
            w, h = image.size
            st.markdown(f"""
            <div style="display:flex;gap:6px;margin-top:6px;flex-wrap:wrap">
              <span class="tag tag-gray">{w}×{h}px</span>
              <span class="tag tag-gray">{uploaded.type.split('/')[-1].upper()}</span>
              <span class="tag tag-gray">{uploaded.size/1024:.1f} KB</span>
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            run = st.button("🔬  Analyse Defect")
        else:
            st.markdown("""<div class="empty-state">
              <div style="font-size:28px;opacity:0.15;margin-bottom:10px">⬆</div>
              <div style="font-size:14px;color:#3d4658">Upload a steel surface image to begin</div>
              <div style="font-size:12px;color:#2a3040;margin-top:4px">PNG · JPG · BMP · TIFF supported</div>
            </div>""", unsafe_allow_html=True)
            run = False

    with col_r:
        st.markdown('<div class="section-label">Analysis results</div>', unsafe_allow_html=True)

        if uploaded and run:
            with st.spinner("Analysing..."):
                pred, conf, probs, av, lat = predict(model, device, image)

            color    = CLASS_INFO[pred]['color']
            desc     = CLASS_INFO[pred]['desc']
            cpct     = av * 100
            tpct     = (1-av) * 100
            branch   = "CNN branch" if av > 0.5 else "Transformer branch"
            btag     = "tag-blue"   if av > 0.5 else "tag-purple"

            st.markdown(f"""<div class="card">
              <div class="section-label">Predicted class</div>
              <div class="result-class" style="color:{color}">
                {pred.replace('_',' ')}
              </div>
              <div class="result-conf">Confidence {conf:.1%}</div>
              <div style="background:#0d1017;border-radius:5px;height:5px;overflow:hidden;margin-bottom:14px">
                <div style="height:100%;width:{conf*100:.1f}%;
                            background:linear-gradient(90deg,{color},#22c55e);border-radius:5px"></div>
              </div>
              <div style="font-size:13px;color:#5a6478;line-height:1.6">{desc}</div>
            </div>""", unsafe_allow_html=True)

            m1,m2,m3 = st.columns(3)
            with m1: st.metric("Confidence", f"{conf:.1%}")
            with m2: st.metric("Latency",    f"{lat:.0f} ms")
            with m3: st.metric("Gate α",     f"{av:.3f}")

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">All class scores</div>', unsafe_allow_html=True)
            st.image(conf_chart(probs), use_column_width=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<div class="section-label">Adaptive fusion gate</div>', unsafe_allow_html=True)

            insight = ('The model is using <strong style="color:#5b9cf6">local texture features</strong> '
                       'from the CNN branch — typical for texture-based defects.'
                       if av > 0.5 else
                       'The model is using <strong style="color:#a78bfa">global spatial context</strong> '
                       'from the Transformer branch — the defect requires whole-image reasoning.')

            st.markdown(f"""<div class="card">
              <div style="display:flex;gap:8px;align-items:center;margin-bottom:10px">
                <span class="tag {btag}">{branch} dominant</span>
                <span class="mono" style="color:#3d4658">α = {av:.3f}</span>
              </div>
              <div style="display:flex;height:8px;border-radius:4px;overflow:hidden;
                          margin-bottom:6px;background:#0d1017">
                <div style="width:{cpct:.1f}%;background:#3d7aed;border-radius:4px 0 0 4px"></div>
                <div style="width:{tpct:.1f}%;background:#8b5cf6;border-radius:0 4px 4px 0"></div>
              </div>
              <div style="display:flex;justify-content:space-between;margin-bottom:12px">
                <span class="mono" style="color:#5b9cf6">CNN {cpct:.0f}%</span>
                <span class="mono" style="color:#a78bfa">Transformer {tpct:.0f}%</span>
              </div>
              <div style="font-size:12px;color:#5a6478;line-height:1.7;
                          padding-top:10px;border-top:1px solid #1e2330">{insight}</div>
            </div>""", unsafe_allow_html=True)

            with st.expander("Per-class gate analysis chart"):
                st.image(alpha_chart(pred, av), use_column_width=True)
                st.markdown("""
                <div style="font-size:11px;color:#3d4658;margin-top:6px;line-height:1.6">
                  Yellow line = this image's gate value. Highlighted bar = predicted class.
                  Inclusion (α=0.343) is the only Transformer-dominant class —
                  the novel finding of DualPath-AFNet.
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="empty-state" style="min-height:400px">
              <div style="font-size:28px;opacity:0.15;margin-bottom:10px">📊</div>
              <div style="font-size:14px;color:#3d4658">Results will appear here after analysis</div>
              <div style="font-size:12px;color:#2a3040;margin-top:4px">Upload an image and click Analyse</div>
            </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
