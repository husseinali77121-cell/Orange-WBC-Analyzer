"""
🔬 Orange WBC Analyzer
======================
White Blood Cell detection & classification from blood film images
using Claude Vision AI — by Orange Lab, 6 October City, Giza, Egypt.

Requirements:
    pip install streamlit anthropic pillow plotly pandas

Run:
    streamlit run orange_wbc_analyzer.py
"""
import streamlit as st

# ── API Key: load from Streamlit Secrets, fallback to sidebar input ──
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    api_key = ""

import anthropic
import base64
import json
import re
from PIL import Image
import io
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Orange WBC Analyzer | Orange Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS  — Orange Lab brand palette
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Space+Grotesk:wght@500;700&display=swap');

:root {
    --orange:     #FF6B1A;
    --orange-lt:  #FF8C42;
    --orange-bg:  #FFF4EE;
    --dark:       #1A1A2E;
    --mid:        #2E3250;
    --accent:     #00C9A7;
    --warn:       #FFB347;
    --danger:     #FF4D6D;
    --neutral:    #E8EAF0;
    --text-body:  #3A3D52;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Header banner */
.lab-header {
    background: linear-gradient(135deg, var(--dark) 0%, var(--mid) 60%, #3a1a05 100%);
    border-bottom: 3px solid var(--orange);
    padding: 1.2rem 2rem;
    border-radius: 0 0 12px 12px;
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.lab-header .logo { font-size: 2rem; }
.lab-header h1 {
    font-family: 'Space Grotesk', sans-serif;
    color: white;
    font-size: 1.6rem;
    margin: 0;
    font-weight: 700;
    letter-spacing: -0.5px;
}
.lab-header h1 span { color: var(--orange); }
.lab-header p { color: #a0a8c0; font-size: 0.8rem; margin: 0; }

/* Cell type cards */
.cell-card {
    background: white;
    border: 1px solid var(--neutral);
    border-left: 4px solid var(--orange);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.7rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}
.cell-card.abnormal { border-left-color: var(--danger); }
.cell-card.warning  { border-left-color: var(--warn); }
.cell-card.normal   { border-left-color: var(--accent); }

.cell-name {
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    color: var(--dark);
}
.cell-count {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--orange);
    font-family: 'Space Grotesk', sans-serif;
}
.cell-pct {
    font-size: 0.85rem;
    color: #888;
}
.cell-note {
    font-size: 0.78rem;
    color: var(--text-body);
    margin-top: 0.3rem;
    line-height: 1.4;
}

/* Finding badge */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
.badge-normal   { background:#e6faf6; color:#00a085; }
.badge-abnormal { background:#ffe8ed; color:#c82345; }
.badge-warning  { background:#fff4dd; color:#b07a00; }

/* Summary box */
.summary-box {
    background: var(--orange-bg);
    border: 1px solid #ffd4b8;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1.2rem;
}
.summary-box h3 {
    color: var(--dark);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    margin-bottom: 0.5rem;
}

/* Interpretation box */
.interp-box {
    background: linear-gradient(135deg, #1A1A2E 0%, #2E3250 100%);
    border-left: 4px solid var(--orange);
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    color: #d0d4e8;
    font-size: 0.9rem;
    line-height: 1.7;
}
.interp-box h3 {
    color: var(--orange);
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    margin-bottom: 0.6rem;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: var(--dark) !important;
}
[data-testid="stSidebar"] * { color: #c0c8e0 !important; }
[data-testid="stSidebar"] h2 {
    color: var(--orange) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* Upload area */
[data-testid="stFileUploader"] {
    border: 2px dashed var(--orange) !important;
    border-radius: 12px !important;
    background: var(--orange-bg) !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, var(--orange), var(--orange-lt)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    font-size: 0.95rem !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 14px rgba(255,107,26,0.4) !important;
}

.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #aaa;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid var(--neutral);
}
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# WBC REFERENCE DATA
# ──────────────────────────────────────────────
WBC_REFERENCE = {
    # Normal cells
    "Neutrophil (Segmented)":     {"type": "normal",   "ref_low": 50, "ref_high": 70, "emoji": "🔵", "arabic": "نيوتروفيل مجزأ"},
    "Neutrophil (Band)":          {"type": "normal",   "ref_low": 0,  "ref_high": 5,  "emoji": "🔵", "arabic": "نيوتروفيل شريطي"},
    "Lymphocyte":                 {"type": "normal",   "ref_low": 20, "ref_high": 40, "emoji": "🟢", "arabic": "لمفاوية"},
    "Monocyte":                   {"type": "normal",   "ref_low": 2,  "ref_high": 8,  "emoji": "🟤", "arabic": "أحادية النواة"},
    "Eosinophil":                 {"type": "normal",   "ref_low": 1,  "ref_high": 4,  "emoji": "🟠", "arabic": "حمضية"},
    "Basophil":                   {"type": "normal",   "ref_low": 0,  "ref_high": 1,  "emoji": "🔴", "arabic": "قاعدية"},
    # Abnormal cells
    "Hypersegmented Neutrophil":  {"type": "abnormal", "ref_low": 0,  "ref_high": 0,  "emoji": "⚠️", "arabic": "نيوتروفيل مفرط التجزئة"},
    "Blast Cell":                 {"type": "abnormal", "ref_low": 0,  "ref_high": 0,  "emoji": "🚨", "arabic": "خلية أرومية (بلاست)"},
    "Atypical Lymphocyte":        {"type": "abnormal", "ref_low": 0,  "ref_high": 0,  "emoji": "⚠️", "arabic": "لمفاوية غير نمطية"},
    "Plasma Cell":                {"type": "abnormal", "ref_low": 0,  "ref_high": 0,  "emoji": "🚨", "arabic": "خلية بلازمية"},
    "Prolymphocyte":              {"type": "abnormal", "ref_low": 0,  "ref_high": 0,  "emoji": "🚨", "arabic": "برولمفاوية"},
    "Promyelocyte":               {"type": "abnormal", "ref_low": 0,  "ref_high": 0,  "emoji": "🚨", "arabic": "ميلوسيت مبكر"},
    "Myelocyte":                  {"type": "abnormal", "ref_low": 0,  "ref_high": 0,  "emoji": "⚠️", "arabic": "ميلوسيت"},
    "Metamyelocyte":              {"type": "abnormal", "ref_low": 0,  "ref_high": 0,  "emoji": "⚠️", "arabic": "ميتاميلوسيت"},
    "Smudge Cell":                {"type": "warning",  "ref_low": 0,  "ref_high": 0,  "emoji": "⚠️", "arabic": "خلية محطمة"},
    "Other / Unclassified":       {"type": "warning",  "ref_low": 0,  "ref_high": 0,  "emoji": "❓", "arabic": "غير مصنف"},
}

CLINICAL_HINTS = {
    "Hypersegmented Neutrophil":
        "Seen in B12/folate deficiency (megaloblastic anemia), uremia. ≥5 lobes in >5% of neutrophils is significant.",
    "Blast Cell":
        "URGENT — presence of blasts suggests acute leukemia (AML/ALL). Immediate hematology referral required.",
    "Atypical Lymphocyte":
        "Common in viral infections (EBV/CMV infectious mononucleosis, CMV, COVID-19). >10% is significant.",
    "Plasma Cell":
        "Suggests multiple myeloma or reactive plasmacytosis if >2%. Protein electrophoresis recommended.",
    "Prolymphocyte":
        "Elevated in Prolymphocytic Leukemia (PLL) or CLL transformation. Further flow cytometry needed.",
    "Promyelocyte":
        "Seen in AML-M3 (APL) — URGENT, DIC risk. Also CML in accelerated phase.",
    "Myelocyte":
        "Part of left shift — seen in severe infections, CML, myeloproliferative disorders.",
    "Metamyelocyte":
        "Left shift indicator — seen in infections, G-CSF therapy, CML.",
    "Smudge Cell":
        "Fragile lymphocytes — classic finding in CLL. >30% smudge cells is highly suggestive.",
}

# ──────────────────────────────────────────────
# CLAUDE API CALL
# ──────────────────────────────────────────────
def analyze_image_with_claude(image_bytes: bytes, api_key: str, stain_type: str, magnification: str) -> dict:
    """Send blood film image to Claude Vision and get structured WBC analysis."""

    client = anthropic.Anthropic(api_key=api_key)

    b64_image = base64.standard_b64encode(image_bytes).decode("utf-8")

    system_prompt = """You are an expert clinical hematology AI assistant trained in morphological analysis of peripheral blood films.
You assist clinical laboratory scientists (MLTs) in identifying and classifying white blood cells (WBCs/leukocytes).

Your task:
1. Carefully examine the blood film image
2. Identify and count ALL visible WBCs
3. Classify each cell type precisely
4. Flag any abnormal morphology

You must respond ONLY with a valid JSON object — no markdown, no code fences, no extra text.

JSON structure:
{
  "total_wbc_counted": <integer — total WBCs you identified>,
  "quality_note": "<comment on image quality, staining, and suitability for analysis>",
  "cells": [
    {
      "cell_type": "<exact cell type name>",
      "count": <integer>,
      "percentage": <float rounded to 1 decimal>,
      "morphology_notes": "<specific morphological observations>",
      "is_abnormal": <true/false>
    }
  ],
  "overall_impression": "<1-2 sentence clinical summary of the WBC picture>",
  "flagged_findings": ["<finding 1>", "<finding 2>"],
  "recommended_action": "<clinical recommendation based on findings>",
  "confidence": "<High / Moderate / Low — with brief reason>"
}

Cell type vocabulary (use these exact names):
Normal: Neutrophil (Segmented), Neutrophil (Band), Lymphocyte, Monocyte, Eosinophil, Basophil
Abnormal: Hypersegmented Neutrophil, Blast Cell, Atypical Lymphocyte, Plasma Cell, Prolymphocyte, Promyelocyte, Myelocyte, Metamyelocyte, Smudge Cell, Other / Unclassified

Important rules:
- Only count cells you can clearly identify; do not guess
- If image quality is poor, note it but still attempt analysis
- Percentages must sum to 100%
- Be strict about blast identification — when in doubt, classify conservatively and flag for review"""

    user_msg = f"""Please analyze this peripheral blood film image.

Stain: {stain_type}
Magnification: {magnification}

Perform a complete differential WBC count and identify any morphological abnormalities.
Return ONLY the JSON response with your findings."""

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=2000,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64_image,
                        },
                    },
                    {"type": "text", "text": user_msg},
                ],
            }
        ],
    )

    raw = response.content[0].text.strip()
    # Strip possible markdown fences
    raw = re.sub(r"^```(?:json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    return json.loads(raw)


# ──────────────────────────────────────────────
# CHART HELPERS
# ──────────────────────────────────────────────
def build_donut_chart(cells: list) -> go.Figure:
    labels, values, colors_list = [], [], []
    color_map = {
        "normal":   ["#00C9A7", "#26D9BC", "#4DE5CC", "#73EDD9", "#99F2E5", "#BFF8F0"],
        "abnormal": ["#FF4D6D", "#FF7090", "#FF96B3"],
        "warning":  ["#FFB347", "#FFC870"],
    }
    normal_idx = abnormal_idx = warning_idx = 0

    for cell in cells:
        if cell["count"] == 0:
            continue
        name = cell["cell_type"]
        ref  = WBC_REFERENCE.get(name, {"type": "warning"})
        ctype = ref["type"]
        labels.append(name)
        values.append(cell["count"])
        if ctype == "normal":
            c = color_map["normal"][normal_idx % len(color_map["normal"])]
            normal_idx += 1
        elif ctype == "abnormal":
            c = color_map["abnormal"][abnormal_idx % len(color_map["abnormal"])]
            abnormal_idx += 1
        else:
            c = color_map["warning"][warning_idx % len(color_map["warning"])]
            warning_idx += 1
        colors_list.append(c)

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors_list, line=dict(color="#fff", width=2)),
        textinfo="label+percent",
        textfont=dict(size=11),
        hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        showlegend=False,
        margin=dict(t=10, b=10, l=10, r=10),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        annotations=[dict(text="WBC<br>Diff", x=0.5, y=0.5,
                          font=dict(size=14, color="#1A1A2E", family="Space Grotesk"),
                          showarrow=False)],
    )
    return fig


def build_bar_chart(cells: list) -> go.Figure:
    df_rows = []
    for cell in cells:
        if cell["count"] == 0:
            continue
        name = cell["cell_type"]
        ref = WBC_REFERENCE.get(name, {"type": "warning", "ref_low": 0, "ref_high": 0})
        df_rows.append({
            "Cell": name,
            "Percentage": cell["percentage"],
            "Type": ref["type"],
            "Ref Low": ref["ref_low"],
            "Ref High": ref["ref_high"],
        })

    if not df_rows:
        return go.Figure()

    df = pd.DataFrame(df_rows)
    color_discrete = {"normal": "#00C9A7", "abnormal": "#FF4D6D", "warning": "#FFB347"}

    fig = px.bar(df, x="Percentage", y="Cell", orientation="h",
                 color="Type", color_discrete_map=color_discrete,
                 text="Percentage")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        xaxis_title="Percentage (%)", yaxis_title="",
        legend_title="Category",
        height=max(250, len(df_rows) * 42),
        margin=dict(l=0, r=40, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e8eaf0"),
    )
    return fig


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("""
<div class="lab-header">
  <div class="logo">🔬</div>
  <div>
    <h1><span>Orange</span> WBC Analyzer</h1>
    <p>AI-Powered Peripheral Blood Film · White Cell Differential & Morphology · Orange Lab, 6 October City</p>
  </div>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    # Show API key status or input field
    if api_key:
        st.success("✅ API Key loaded from secrets")
    else:
        api_key = st.text_input(
            "Anthropic API Key",
            type="password",
            help="Your Claude API key — never stored. Or add to Streamlit secrets.",
        )

    st.divider()
    st.markdown("### 🔬 Slide Parameters")
    stain_type = st.selectbox("Stain Type",
                              ["Leishman", "Giemsa", "Wright", "May-Grünwald-Giemsa", "H&E", "Other"])
    magnification = st.selectbox("Objective Magnification",
                                 ["40×", "100× (Oil)", "60×", "20×"])
    st.divider()
    st.markdown("### 📋 WBC Reference Ranges")
    ref_df = pd.DataFrame([
        {"Cell": k, "Low%": v["ref_low"], "High%": v["ref_high"]}
        for k, v in WBC_REFERENCE.items() if v["type"] == "normal"
    ])
    st.dataframe(ref_df, hide_index=True, use_container_width=True)
    st.divider()
    st.caption("🏥 Orange Lab · Clinical Microbiology & Diagnostics")
    st.caption("⚠️ For professional use only — not a substitute for expert morphological review")

# ──────────────────────────────────────────────
# MAIN LAYOUT
# ──────────────────────────────────────────────
col_upload, col_results = st.columns([1, 1.6], gap="large")

with col_upload:
    st.markdown("### 📤 Upload Blood Film Image")
    uploaded_file = st.file_uploader(
        "Drop blood film image here",
        type=["jpg", "jpeg", "png", "tif", "tiff", "bmp"],
        help="Best results with 100× oil-immersion images, good staining"
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

        # Image stats
        w, h = img.size
        st.markdown(f"""
        <div style="background:#f5f6fa;border-radius:8px;padding:0.6rem 1rem;font-size:0.8rem;color:#555;margin-top:0.5rem;">
        📐 {w}×{h}px &nbsp;|&nbsp; 🎨 {img.mode} &nbsp;|&nbsp; 📁 {uploaded_file.size/1024:.1f} KB
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        analyze_btn = st.button("🔍 Analyze WBCs", use_container_width=True)
    else:
        st.info("👆 Upload a peripheral blood film image to begin analysis")
        analyze_btn = False

    # Sample info
    st.markdown("#### 💡 Tip — Optimal Image Quality")
    st.markdown("""
- Use **100× oil-immersion** images for best results  
- Good **Leishman or Giemsa** staining — clear nuclear detail  
- Capture from the **monolayer zone** (not thick or thin edges)  
- Avoid **out-of-focus** or **overexposed** images  
""")

# ──────────────────────────────────────────────
# ANALYSIS & RESULTS
# ──────────────────────────────────────────────
with col_results:
    st.markdown("### 📊 Analysis Results")

    if analyze_btn:
        if not api_key:
            st.error("❌ Please enter your Anthropic API key in the sidebar.")
        elif not uploaded_file:
            st.error("❌ Please upload a blood film image first.")
        else:
            with st.spinner("🔬 Claude is analyzing the blood film…"):
                try:
                    img_bytes = uploaded_file.getvalue()
                    # Resize if very large (>4 MB) to stay within API limits
                    if len(img_bytes) > 4_000_000:
                        pil_img = Image.open(io.BytesIO(img_bytes))
                        pil_img.thumbnail((2048, 2048), Image.LANCZOS)
                        buf = io.BytesIO()
                        pil_img.save(buf, format="JPEG", quality=85)
                        img_bytes = buf.getvalue()

                    result = analyze_image_with_claude(img_bytes, api_key, stain_type, magnification)
                    st.session_state["last_result"] = result
                    st.session_state["last_filename"] = uploaded_file.name

                except json.JSONDecodeError as e:
                    st.error(f"❌ Failed to parse Claude response as JSON: {e}")
                    st.session_state.pop("last_result", None)
                except anthropic.AuthenticationError:
                    st.error("❌ Invalid API key. Please check your Anthropic API key.")
                    st.session_state.pop("last_result", None)
                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    st.session_state.pop("last_result", None)

    # Display cached result
    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        cells  = result.get("cells", [])
        total  = result.get("total_wbc_counted", 0)
        abnormal_cells = [c for c in cells if c.get("is_abnormal") and c["count"] > 0]
        normal_cells   = [c for c in cells if not c.get("is_abnormal") and c["count"] > 0]

        # ── Summary row ──
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total WBCs Counted", total)
        c2.metric("Cell Types Found", len([c for c in cells if c["count"] > 0]))
        c3.metric("Abnormal Types", len(abnormal_cells),
                  delta="⚠️ Flag" if abnormal_cells else None,
                  delta_color="inverse" if abnormal_cells else "off")
        c4.metric("Confidence", result.get("confidence", "—").split("—")[0].split("–")[0].strip())

        # ── Quality note ──
        if result.get("quality_note"):
            st.markdown(f"""
            <div style="background:#f0f4ff;border-left:3px solid #5c7cfa;border-radius:8px;
                        padding:0.6rem 1rem;font-size:0.82rem;color:#3a3d52;margin-bottom:0.8rem;">
            🖼️ <b>Image quality:</b> {result["quality_note"]}
            </div>""", unsafe_allow_html=True)

        # ── Flags ──
        flags = result.get("flagged_findings", [])
        if flags:
            st.markdown("#### 🚨 Flagged Findings")
            for flag in flags:
                st.markdown(f"""
                <div style="background:#fff0f3;border-left:4px solid #FF4D6D;border-radius:8px;
                            padding:0.5rem 1rem;margin-bottom:0.4rem;font-size:0.88rem;color:#c82345;">
                ⚠️ {flag}
                </div>""", unsafe_allow_html=True)

        # ── Charts ──
        tab_chart, tab_cells, tab_interp = st.tabs(["📈 Charts", "🔬 Cell Details", "📝 Interpretation"])

        with tab_chart:
            ch1, ch2 = st.columns([1, 1.4])
            with ch1:
                st.plotly_chart(build_donut_chart(cells), use_container_width=True, config={"displayModeBar": False})
            with ch2:
                st.plotly_chart(build_bar_chart(cells), use_container_width=True, config={"displayModeBar": False})

        with tab_cells:
            # Normal cells
            if normal_cells:
                st.markdown("**✅ Normal WBC Population**")
                for cell in normal_cells:
                    name = cell["cell_type"]
                    ref  = WBC_REFERENCE.get(name, {"type": "normal", "ref_low": 0, "ref_high": 100, "arabic": name})
                    pct  = cell["percentage"]
                    lo, hi = ref["ref_low"], ref["ref_high"]
                    status = "normal" if lo <= pct <= hi else "warning"
                    badge_class = f"badge-{status}"
                    badge_label = "Within Range" if status == "normal" else f"Out of Range ({lo}–{hi}%)"
                    emoji = WBC_REFERENCE.get(name, {}).get("emoji", "🔵")
                    arabic = ref.get("arabic", "")
                    st.markdown(f"""
                    <div class="cell-card {status}">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div>
                          <div class="cell-name">{emoji} {name} <span style="color:#aaa;font-size:0.8rem;font-weight:400;">({arabic})</span></div>
                          <div class="cell-note">{cell.get("morphology_notes","")}</div>
                        </div>
                        <div style="text-align:right;">
                          <div class="cell-count">{cell["count"]}</div>
                          <div class="cell-pct">{pct}%</div>
                          <span class="badge {badge_class}">{badge_label}</span>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            # Abnormal cells
            if abnormal_cells:
                st.markdown("**⚠️ Abnormal / Immature Cells**")
                for cell in abnormal_cells:
                    name  = cell["cell_type"]
                    ref   = WBC_REFERENCE.get(name, {"type": "abnormal", "arabic": name})
                    emoji = ref.get("emoji", "⚠️")
                    arabic = ref.get("arabic", "")
                    hint  = CLINICAL_HINTS.get(name, "")
                    st.markdown(f"""
                    <div class="cell-card abnormal">
                      <div style="display:flex;justify-content:space-between;align-items:flex-start;">
                        <div style="flex:1;">
                          <div class="cell-name">{emoji} {name} <span style="color:#aaa;font-size:0.8rem;font-weight:400;">({arabic})</span></div>
                          <div class="cell-note">{cell.get("morphology_notes","")}</div>
                          {f'<div class="cell-note" style="color:#c82345;margin-top:4px;">💡 {hint}</div>' if hint else ""}
                        </div>
                        <div style="text-align:right;margin-left:1rem;">
                          <div class="cell-count" style="color:#FF4D6D;">{cell["count"]}</div>
                          <div class="cell-pct">{cell["percentage"]}%</div>
                          <span class="badge badge-abnormal">ABNORMAL</span>
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

        with tab_interp:
            st.markdown(f"""
            <div class="interp-box">
              <h3>🔬 Overall Impression</h3>
              <p>{result.get("overall_impression","—")}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")
            if result.get("recommended_action"):
                st.markdown(f"""
                <div class="summary-box">
                  <h3>📋 Recommended Action</h3>
                  <p style="color:#3a3d52;font-size:0.9rem;">{result["recommended_action"]}</p>
                </div>
                """, unsafe_allow_html=True)

            # Differential table
            st.markdown("**📊 Complete Differential Count**")
            df_out = pd.DataFrame([
                {
                    "Cell Type": c["cell_type"],
                    "Count": c["count"],
                    "%": f"{c['percentage']:.1f}%",
                    "Morphology Notes": c.get("morphology_notes", "—"),
                    "Status": "⚠️ Abnormal" if c["is_abnormal"] else "✅ Normal"
                }
                for c in cells if c["count"] > 0
            ])
            st.dataframe(df_out, use_container_width=True, hide_index=True)

            # Export
            csv = df_out.to_csv(index=False)
            fname = st.session_state.get("last_filename", "blood_film")
            ts = datetime.now().strftime("%Y%m%d_%H%M")
            st.download_button(
                "⬇️ Download CSV Report",
                data=csv,
                file_name=f"WBC_diff_{fname}_{ts}.csv",
                mime="text/csv",
                use_container_width=True,
            )

    else:
        st.markdown("""
        <div style="background:#f8f9fb;border-radius:12px;padding:2.5rem;text-align:center;
                    border:2px dashed #dde1f0;margin-top:1rem;">
          <div style="font-size:3rem;margin-bottom:1rem;">🔬</div>
          <h3 style="color:#3a3d52;font-family:'Space Grotesk',sans-serif;">Ready to Analyze</h3>
          <p style="color:#888;font-size:0.9rem;">Upload a blood film image and click <strong>Analyze WBCs</strong><br>
          Results include full differential count + abnormal cell detection</p>
          <br>
          <div style="display:flex;justify-content:center;gap:1rem;flex-wrap:wrap;">
            <span style="background:#e6faf6;color:#00a085;padding:4px 12px;border-radius:20px;font-size:0.8rem;">✅ 5 Normal WBC Types</span>
            <span style="background:#ffe8ed;color:#c82345;padding:4px 12px;border-radius:20px;font-size:0.8rem;">🚨 Blasts / Leukemia</span>
            <span style="background:#fff4dd;color:#b07a00;padding:4px 12px;border-radius:20px;font-size:0.8rem;">⚠️ Hypersegmented</span>
            <span style="background:#ffe8ed;color:#c82345;padding:4px 12px;border-radius:20px;font-size:0.8rem;">⚠️ Atypical Lymphs</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────
st.markdown("""
<div class="footer">
  🏥 <strong>Orange Lab</strong> · Clinical Microbiology & Diagnostics · 6 October City, Giza, Egypt<br>
  ⚠️ This tool is designed to <strong>assist</strong> qualified laboratory professionals — not replace expert morphological review.<br>
  All results must be verified by a licensed MLT or hematologist before clinical use.
</div>
""", unsafe_allow_html=True)
