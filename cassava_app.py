"""
Cassava Leaf Disease Detection — Streamlit App
============================================================
Run with:
    streamlit run cassava_app.py

Requirements:
    pip install streamlit tensorflow pillow plotly numpy boto3

Models are downloaded automatically from AWS S3:
    s3://cassava-disease-models/
============================================================
Author : Ebaju Edward
Student: 2400723929
"""

import streamlit as st
import numpy as np
from PIL import Image
import plotly.graph_objects as go
import os
import io
import boto3
import tempfile

# ── AWS S3 config ──────────────────────────────────────────────────────────────
S3_BUCKET = "cassava-disease-models"
S3_REGION = "eu-north-1"

# ── Download models from S3 if not already cached ─────────────────────────────
@st.cache_resource(show_spinner="Downloading models from AWS S3...")
def download_models():
    s3 = boto3.client(
        "s3",
        region_name=S3_REGION,
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    )
    model_files = [
        "cassava_efficientnetb0_final.keras",
        "cassava_resnet50_final.keras",
        "cassava_mobilenetv2_final.h5",
    ]
    paths = {}
    tmp_dir = tempfile.mkdtemp()
    for filename in model_files:
        local_path = os.path.join(tmp_dir, filename)
        s3.download_file(S3_BUCKET, filename, local_path)
        paths[filename] = local_path
    return paths

MODEL_PATHS = download_models()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cassava Leaf Disease Detector",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0d1f0f;
        color: #e8f0e9;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0a1a0c;
        border-right: 1px solid #1e3a20;
    }

    /* Header */
    .main-header {
        font-family: 'Syne', sans-serif;
        font-size: 2.8rem;
        font-weight: 800;
        color: #7ddb8a;
        letter-spacing: -1px;
        line-height: 1.1;
        margin-bottom: 0.2rem;
    }
    .main-subtitle {
        font-family: 'DM Sans', sans-serif;
        font-size: 1rem;
        color: #5a8a60;
        font-weight: 300;
        margin-bottom: 2rem;
    }

    /* Cards */
    .result-card {
        background: #122415;
        border: 1px solid #1e3a20;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    .disease-name {
        font-family: 'Syne', sans-serif;
        font-size: 1.6rem;
        font-weight: 700;
        color: #7ddb8a;
        margin-bottom: 0.3rem;
    }
    .confidence-badge {
        display: inline-block;
        background: #1e3a20;
        color: #7ddb8a;
        font-family: 'Syne', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        padding: 0.3rem 1rem;
        border-radius: 999px;
        margin-bottom: 1rem;
    }
    .info-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #5a8a60;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .info-text {
        font-size: 0.95rem;
        color: #c8deca;
        line-height: 1.6;
        margin-bottom: 1rem;
    }
    .treatment-item {
        display: flex;
        align-items: flex-start;
        gap: 0.6rem;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        color: #c8deca;
    }
    .treatment-dot {
        color: #7ddb8a;
        margin-top: 2px;
        flex-shrink: 0;
    }

    /* Healthy card */
    .healthy-card {
        background: #0d2e14;
        border: 1px solid #2a5c30;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
    }

    /* Upload zone */
    [data-testid="stFileUploader"] {
        background: #122415;
        border: 2px dashed #2a5c30;
        border-radius: 16px;
        padding: 1rem;
    }

    /* Buttons */
    .stButton > button {
        background: #7ddb8a;
        color: #0d1f0f;
        font-family: 'Syne', sans-serif;
        font-weight: 700;
        border: none;
        border-radius: 999px;
        padding: 0.6rem 2rem;
        font-size: 0.95rem;
        letter-spacing: 0.02em;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background: #a0e8aa;
        transform: translateY(-1px);
    }

    /* Model selector */
    .stSelectbox > div > div {
        background: #122415;
        border: 1px solid #2a5c30;
        border-radius: 10px;
        color: #e8f0e9;
    }

    /* Divider */
    hr {
        border-color: #1e3a20;
    }

    /* Image caption */
    .img-caption {
        font-size: 0.78rem;
        color: #5a8a60;
        text-align: center;
        margin-top: 0.4rem;
    }

    /* Model badge */
    .model-badge {
        display: inline-block;
        background: #1e3a20;
        color: #5dbb6a;
        font-size: 0.75rem;
        font-weight: 500;
        padding: 0.2rem 0.7rem;
        border-radius: 999px;
        margin-bottom: 1.5rem;
    }

    /* Warning */
    .stWarning {
        background: #1e2e10;
        border-color: #4a7a20;
    }
</style>
""", unsafe_allow_html=True)

# ── Disease information ────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Cassava___healthy": {
        "display_name": "Healthy",
        "emoji": "✅",
        "description": "The cassava plant appears healthy with no visible signs of disease. Continue standard agronomic practices to maintain plant health.",
        "severity": "None",
        "treatments": [
            "Maintain regular watering and fertilisation schedule",
            "Continue monitoring for early signs of disease",
            "Practice crop rotation to prevent future infections",
            "Ensure proper spacing for adequate airflow"
        ],
        "is_healthy": True
    },
    "Cassava___mosaic_disease": {
        "display_name": "Cassava Mosaic Disease",
        "emoji": "⚠️",
        "description": "Caused by Cassava Mosaic Virus (CMV) transmitted by whiteflies. Characterised by mosaic-like yellow and green patterns on leaves, leaf distortion, and stunted growth. One of the most damaging cassava diseases in Africa.",
        "severity": "High",
        "treatments": [
            "Remove and destroy infected plants immediately to prevent spread",
            "Control whitefly populations using insecticides (e.g. imidacloprid)",
            "Plant virus-resistant cassava varieties (e.g. TME 419, NASE 14)",
            "Use clean, certified disease-free planting materials",
            "Introduce natural whitefly predators as biological control",
            "Avoid planting near other infected fields"
        ],
        "is_healthy": False
    },
    "Cassava___green_mottle": {
        "display_name": "Cassava Green Mottle",
        "emoji": "⚠️",
        "description": "Caused by Cassava Green Mottle Virus (CGMV). Presents as green mottling and mosaic patterns on leaves, mild leaf distortion, and can reduce photosynthetic efficiency leading to yield loss.",
        "severity": "Moderate",
        "treatments": [
            "Use certified virus-free planting materials for new crops",
            "Remove heavily infected plants from the field",
            "Control insect vectors through targeted pesticide application",
            "Practice good field hygiene — clean tools between plants",
            "Plant tolerant varieties where available",
            "Monitor fields regularly for early detection"
        ],
        "is_healthy": False
    },
    "Cassava___brown_streak_disease": {
        "display_name": "Cassava Brown Streak Disease",
        "emoji": "🔴",
        "description": "Caused by Cassava Brown Streak Virus (CBSV), transmitted by whiteflies. Causes yellow streaks on leaves and brown necrotic lesions on the roots (tubers), making them inedible. Considered the most destructive cassava disease in East Africa.",
        "severity": "Very High",
        "treatments": [
            "Immediately destroy all infected plants including roots",
            "Never use stems from infected plants for propagation",
            "Plant only certified CBSD-resistant varieties (e.g. Narocas 1)",
            "Aggressively control whitefly vectors with systemic insecticides",
            "Establish a buffer zone around infected fields",
            "Report outbreaks to local agricultural authorities",
            "Consider fallowing the field for one season"
        ],
        "is_healthy": False
    }
}

# ── Model paths (resolved from AWS S3 cache) ──────────────────────────────────
MODEL_OPTIONS = {
    "EfficientNetB0": MODEL_PATHS["cassava_efficientnetb0_final.keras"],
    "ResNet50":       MODEL_PATHS["cassava_resnet50_final.keras"],
    "MobileNetV2":    MODEL_PATHS["cassava_mobilenetv2_final.h5"],
}

CLASS_NAMES = [
    "Cassava___brown_streak_disease",
    "Cassava___green_mottle",
    "Cassava___healthy",
    "Cassava___mosaic_disease"
]

# ── Load model (cached) ────────────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name):
    import tensorflow as tf
    model_path = MODEL_OPTIONS[model_name]
    return tf.keras.models.load_model(model_path)

# ── Preprocess image ───────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image, model_name: str) -> np.ndarray:
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32)

    if model_name == "EfficientNetB0":
        from tensorflow.keras.applications.efficientnet import preprocess_input
    elif model_name == "ResNet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input
    else:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    arr = preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(model, img_array: np.ndarray):
    preds = model.predict(img_array, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    return preds, top_idx, CLASS_NAMES[top_idx]

# ── Confidence bar chart ───────────────────────────────────────────────────────
def make_chart(preds, predicted_class):
    labels = [DISEASE_INFO[c]["display_name"] for c in CLASS_NAMES]
    colors = ["#7ddb8a" if CLASS_NAMES[i] == predicted_class else "#2a5c30"
              for i in range(len(CLASS_NAMES))]

    fig = go.Figure(go.Bar(
        x=[p * 100 for p in preds],
        y=labels,
        orientation='h',
        marker_color=colors,
        text=[f"{p*100:.1f}%" for p in preds],
        textposition='outside',
        textfont=dict(color='#c8deca', size=12)
    ))

    fig.update_layout(
        plot_bgcolor='#122415',
        paper_bgcolor='#122415',
        font=dict(color='#c8deca', family='DM Sans'),
        xaxis=dict(
            showgrid=False, zeroline=False,
            showticklabels=False, range=[0, 115]
        ),
        yaxis=dict(showgrid=False, tickfont=dict(size=12)),
        margin=dict(l=10, r=60, t=10, b=10),
        height=180,
        bargap=0.35
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌿 Settings")
    st.markdown("---")

    selected_model = st.selectbox(
        "Select Model",
        options=list(MODEL_OPTIONS.keys()),
        help="Choose which trained model to use for prediction"
    )

    st.markdown("---")
    st.markdown("**About the Classes**")
    for key, info in DISEASE_INFO.items():
        st.markdown(f"{info['emoji']} **{info['display_name']}**")
        if not info['is_healthy']:
            st.caption(f"Severity: {info['severity']}")
    st.markdown("---")
    st.caption("Cassava Leaf Disease Detection System")
    st.caption("Powered by Deep Learning & AWS S3")
    st.markdown("---")
    st.markdown("**Author**")
    st.caption("Ebaju Edward")
    st.caption("Student No: 2400723929")

# ── Main area ──────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">🌿 Cassava Disease<br>Detector</div>', unsafe_allow_html=True)
st.markdown('<div class="main-subtitle">Upload cassava leaf images to detect diseases using deep learning</div>', unsafe_allow_html=True)
st.markdown(f'<div class="model-badge">Model: {selected_model}</div>', unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
model = load_model(selected_model)

# ── Camera image state ────────────────────────────────────────────────────────
if "camera_image" not in st.session_state:
    st.session_state.camera_image = None
if "show_camera" not in st.session_state:
    st.session_state.show_camera = False

# ── Always-visible input section ──────────────────────────────────────────────
col_btn1, col_btn2, col_spacer = st.columns([1, 1, 2])

with col_btn1:
    if st.button("📷  Take Photo", use_container_width=True):
        st.session_state.show_camera = not st.session_state.show_camera
        st.session_state.camera_image = None

with col_btn2:
    pass  # upload widget rendered below

# Camera widget — shown inline below buttons when toggled
if st.session_state.show_camera and st.session_state.camera_image is None:
    captured = st.camera_input("Point camera at a cassava leaf")
    if captured:
        st.session_state.camera_image = captured
        st.session_state.show_camera = False   # close camera after capture
        st.rerun()

# Show retake option if a camera image exists
if st.session_state.camera_image is not None:
    col_prev, col_retake = st.columns([3, 1])
    with col_retake:
        if st.button("🔄 Retake Photo"):
            st.session_state.camera_image = None
            st.session_state.show_camera = True
            st.rerun()

# Upload widget — always visible
with col_btn2:
    pass

uploaded_from_file = st.file_uploader(
    "📁  Upload image(s) from device",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed"
)

# ── Gather images for prediction ──────────────────────────────────────────────
uploaded_files = []

if st.session_state.camera_image:
    uploaded_files = [st.session_state.camera_image]
elif uploaded_from_file:
    uploaded_files = uploaded_from_file

if uploaded_files and model is not None:
    st.markdown("---")
    st.markdown(f"**{len(uploaded_files)} image(s) ready — running predictions...**")
    st.markdown("")

    for i, uploaded_file in enumerate(uploaded_files):
        img = Image.open(uploaded_file)

        col1, col2 = st.columns([1, 1.6], gap="large")

        with col1:
            st.image(img, use_container_width=True)
            st.markdown(f'<div class="img-caption">{uploaded_file.name}</div>', unsafe_allow_html=True)

        with col2:
            with st.spinner("Analysing..."):
                arr = preprocess_image(img, selected_model)
                preds, top_idx, predicted_class = predict(model, arr)
                info = DISEASE_INFO[predicted_class]
                confidence = preds[top_idx] * 100

            if info["is_healthy"]:
                st.markdown(f"""
                <div class="healthy-card">
                    <div style="font-size:2.5rem">✅</div>
                    <div class="disease-name" style="color:#7ddb8a">Healthy Plant</div>
                    <div class="confidence-badge">{confidence:.1f}% confidence</div>
                    <div class="info-text">{info['description']}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                severity_color = {"Moderate": "#f0c040", "High": "#e07030", "Very High": "#d03030"}.get(info["severity"], "#7ddb8a")
                treatments_html = "".join(
                    f'<div class="treatment-item"><span class="treatment-dot">▸</span><span>{t}</span></div>'
                    for t in info["treatments"]
                )
                card_html = f"""
                <div class="result-card">
                    <div style="font-size:1.5rem">{info['emoji']}</div>
                    <div class="disease-name">{info['display_name']}</div>
                    <div class="confidence-badge">{confidence:.1f}% confidence</div>
                    <div class="info-label">Severity</div>
                    <div style="color:{severity_color}; font-weight:600; margin-bottom:1rem">{info['severity']}</div>
                    <div class="info-label">Description</div>
                    <div class="info-text">{info['description']}</div>
                    <div class="info-label">Recommended Treatment</div>
                    """ + treatments_html + """
                </div>"""
                st.markdown(card_html, unsafe_allow_html=True)

            # Confidence chart
            st.markdown('<div class="info-label" style="margin-top:1rem">All Class Probabilities</div>', unsafe_allow_html=True)
            st.plotly_chart(make_chart(preds, predicted_class), use_container_width=True, config={"displayModeBar": False}, key=f"chart_{i}")

        if i < len(uploaded_files) - 1:
            st.markdown("---")

elif uploaded_files and model is None:
    st.error("Please ensure the model file is available before uploading images.")

elif not uploaded_files and not st.session_state.show_camera:
    st.markdown("""
    <div style="text-align:center; padding: 3rem 2rem; color: #3a6a40;">
        <div style="font-size: 3.5rem; margin-bottom: 1rem">🍃</div>
        <div style="font-family: 'Syne', sans-serif; font-size: 1.1rem; color: #5a8a60; margin-bottom: 0.4rem">
            Take a photo or upload an image to get started
        </div>
    </div>
    """, unsafe_allow_html=True)
