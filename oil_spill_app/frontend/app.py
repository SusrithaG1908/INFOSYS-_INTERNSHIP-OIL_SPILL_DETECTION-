# app.py (final merged version)
import streamlit as st
import requests
import io
import base64
from PIL import Image, ImageEnhance
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import json
import os
from dotenv import load_dotenv


load_dotenv()  # 👈 loads variables from .env file into environment
# ---------------- CONFIG ----------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
BACKEND_URL_PREDICT = BACKEND_URL + "/predict"  # change if needed
BACKEND_GET_ALL_PAST_PREDICTIONS = BACKEND_URL + "/get_all_detections"
DEFAULT_THRESHOLD = 0.5
DEFAULT_ALPHA = 0.4
MODEL_INPUT_SIZE = (256, 256)

st.set_page_config(
    page_title="Oil Spill Detection",
    layout="wide",
    page_icon="🛢️",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS (full original) ----------------
st.markdown(""" 
    <style>
    .main-title {
        text-align: center;
        font-size: 56px;
        font-weight: 900;
        background: linear-gradient(90deg, #0077b6, #00b4d8, #90e0ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: 2px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        margin-bottom: -10px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #444;
        margin-top: 5px;
        font-weight: 500;
    }
    .divider {
        height: 4px;
        width: 160px;
        background: linear-gradient(90deg, #0077b6, #00b4d8, #90e0ef);
        margin: 20px auto 40px auto;
        border-radius: 5px;
    }
    .feature-card {
        background: linear-gradient(135deg, #0077b6 0%, #00b4d8 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
    }
    .feature-card h3 {
        color: white;
        font-size: 20px;
        margin-bottom: 10px;
    }
    .feature-card p {
        color: #e6f7ff;
        font-size: 14px;
        margin: 0;
    }
    .info-card {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
    }
    .info-card h3 {
        color: white;
        font-size: 18px;
        margin-bottom: 10px;
    }
    .info-card p {
        color: #e8f6ef;
        font-size: 14px;
        margin: 5px 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        color: white;
        text-align: center;
        border: none;
    }
    .metric-card h3 {
        color: white;
        font-size: 16px;
        margin-bottom: 10px;
        font-weight: 600;
    }
    .metric-card h2 {
        color: white;
        font-size: 28px;
        margin: 10px 0;
        font-weight: 700;
    }
    .metric-card p {
        color: #fdedec;
        font-size: 12px;
        margin: 5px 0;
    }
    .upload-box {
        border: 2px dashed #0077b6;
        border-radius: 10px;
        padding: 30px;
        text-align: center;
        background-color: #f8f9fa;
        margin: 20px 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #0077b6, #00b4d8, #90e0ef);
    }
    .section-header {
        color: #00509e;
        font-size: 34px;
        font-weight: 800;
        margin: 30px 0 20px 0;
        padding-bottom: 10px;
        border-bottom: 4px solid #00b4d8;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
    }
    .subsection-header {
        color: #00509e;
        font-size: 24px;
        font-weight: 600;
        margin: 25px 0 15px 0;
    }
    .content-section {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        padding: 25px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #0077b6;
    }
    .content-section h3 {
        color: #2c3e50;
        font-size: 20px;
        margin-bottom: 15px;
    }
    .content-section ul {
        color: #2c3e50;
        font-size: 16px;
        line-height: 1.6;
    }
    .content-section li {
        margin: 8px 0;
        color: #2c3e50;
    }
    .welcome-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 40px;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 20px 0;
    }
    .welcome-section h2 {
        color: white;
        font-size: 32px;
        margin-bottom: 15px;
    }
    .welcome-section p {
        color: #e8eaf6;
        font-size: 18px;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    .sidebar-section h3 {
        font-size: 20px;
        font-weight: 800;
        margin-top: 20px;
        margin-bottom: 10px;
        background: linear-gradient(90deg, #0077b6, #00b4d8, #90e0ef);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.15);
    }
    .sidebar-section p, .sidebar-section li {
        color: #2c3e50 !important;
        font-size: 15px;
        line-height: 1.5;
    }
    .sidebar-section strong {
        color: #00509e;
    }
    .css-1d391kg { color: #2c3e50 !important; }
    .stSidebar .stMarkdown { color: #2c3e50 !important; }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: #00509e !important;
    }
    .stMarkdown { color: #2c3e50; }
    h1,h2,h3,h4,h5,h6 { color: #2c3e50 !important; }
    [data-testid="stExpander"] {
        background: linear-gradient(135deg, #ffffff 0%, #f0faff 100%);
        border: 2px solid #0077b6;
        border-radius: 12px !important;
        box-shadow: 0 4px 10px rgba(0, 119, 182, 0.15);
        margin-top: 20px;
        color: #2c3e50;
    }
    [data-testid="stExpander"] > div:first-child {
        background: linear-gradient(90deg, #0077b6, #00b4d8, #90e0ef);
        border-radius: 12px 12px 0 0;
        color: white !important;
        font-weight: 700;
        font-size: 18px;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    .expander-content { padding: 12px 16px 10px 16px; color: #1e2a36; font-size: 16px; line-height: 1.6; }
    .expander-content h3 { color: #005f8f; font-size: 20px; font-weight: 750; border-left: 4px solid #00b4d8; padding-left: 8px; margin-top: 12px; margin-bottom: 10px; }
    .expander-content strong { color: #004b76; }
    .expander-content li { margin-bottom: 5px; color: #1e2a36; }
    </style>
""", unsafe_allow_html=True)

# ---------------- Utility functions ----------------
def image_to_bytes(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def decode_base64_image(base64_str):
    """Safely decode base64 image string to PIL Image. Returns None on failure."""
    if not base64_str:
        return None
    try:
        # If the string contains a data URL prefix, strip it
        if isinstance(base64_str, str) and base64_str.startswith("data:"):
            base64_str = base64_str.split(",", 1)[1]
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data)).convert("RGBA")
    except Exception:
        return None

def mask_from_overlay(overlay_img, threshold_val=50):
    arr = np.array(overlay_img.convert("RGB"))
    red_chan = arr[..., 0].astype(np.int32)
    green_chan = arr[..., 1].astype(np.int32)
    blue_chan = arr[..., 2].astype(np.int32)
    mask = ((red_chan - ((green_chan + blue_chan) // 2)) > threshold_val).astype(np.uint8) * 255
    return Image.fromarray(mask)

def create_area_chart(area_data):
    labels = ['Oil Spill', 'Water']
    values = [area_data['oil_pixels'], max(area_data['total_pixels'] - area_data['oil_pixels'], 0)]
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title="Area Distribution", height=300)
    return fig

def create_confidence_heatmap_from_array(array2d):
    fig = px.imshow(array2d, color_continuous_scale='viridis', title="Confidence Heatmap", aspect='equal')
    fig.update_layout(coloraxis_colorbar=dict(title="Confidence"))
    return fig

# ---------------- MAIN HEADER ----------------
st.markdown('<h1 class="main-title">🌊 OIL SPILL DETECTION SYSTEM 🛢️</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered satellite image analysis for environmental protection</p>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------------- SIDEBAR (full options) ----------------
st.sidebar.markdown("<div class='sidebar-section'><h3>⚙️ Settings</h3></div>", unsafe_allow_html=True)

# Detection Settings
st.sidebar.markdown("<div class='sidebar-section'><h3>🔍 Detection Settings</h3></div>", unsafe_allow_html=True)
threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, DEFAULT_THRESHOLD, 0.05)
alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, DEFAULT_ALPHA, 0.05)

# Image Enhancement
st.sidebar.markdown("<div class='sidebar-section'><h3>🖼️ Image Enhancement</h3></div>", unsafe_allow_html=True)
brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
sharpness = st.sidebar.slider("Sharpness", 0.0, 2.0, 1.0, 0.1)

# Analysis Options
st.sidebar.markdown("<div class='sidebar-section'><h3>📊 Analysis Options</h3></div>", unsafe_allow_html=True)
pixel_size = st.sidebar.number_input("Pixel size (km²)", 0.001, 1.0, 0.01, 0.001, help="Estimated area each pixel represents in square kilometers")
show_confidence_map = st.sidebar.checkbox("Show Confidence Map", True)
show_metrics = st.sidebar.checkbox("Show Detailed Metrics", True)

# ----------------------------------------------
# Session state initialization
# ----------------------------------------------
if "detections" not in st.session_state:
    st.session_state.detections = []

if "mode" not in st.session_state:
    st.session_state.mode = "predict"

if "analyze" not in st.session_state:
    st.session_state.analyze = False

if "selected_detection" not in st.session_state:
    st.session_state.selected_detection = None

if "last_saved" not in st.session_state:
    st.session_state.last_saved = None

if 'view_mode' not in st.session_state:
    st.session_state['view_mode'] = 'default'

# ----------------------------------------------
# 🗄️ VIEW PAST DETECTIONS
# ----------------------------------------------
st.sidebar.markdown("<div class='sidebar-section'><h3>🛢️ View Past Detections</h3></div>", unsafe_allow_html=True)

# Custom CSS for green button (kept)
st.markdown(
    """
    <style>
    div[data-testid="stSidebar"] button[kind="secondary"] {
        background-color: #2ecc71 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        box-shadow: 0 3px 6px rgba(0,0,0,0.1) !important;
        transition: all 0.2s ease-in-out !important;
    }
    div[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background-color: #27ae60 !important;
        transform: scale(1.03);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- View Records button ---
if st.sidebar.button("View Records"):
    st.session_state.mode = "view"
    st.session_state.analyze = False

# --- Load records if view mode ---
if st.session_state.mode == "view":
    if st.sidebar.button("🔄 Refresh Records"):
        try:
            resp = requests.get(BACKEND_GET_ALL_PAST_PREDICTIONS, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.detections = data.get("detections", [])
                if not st.session_state.detections:
                    st.sidebar.info("No past detections found.")
            else:
                st.sidebar.error(f"Error fetching detections: {resp.status_code}")
        except Exception as e:
            st.sidebar.error(f"Backend connection error: {e}")

    if not st.session_state.detections:
        st.sidebar.warning("No previous detections found yet — upload an image to start analysis.")
    else:
        # list stored detections and make buttons to select them
        for idx, det in enumerate(st.session_state.detections):
            filename = det.get("filename", f"record_{idx}")
            percent = det.get("percentage", 0.0)
            btn_label = f"🖼️ {filename} ({percent:.1f}% coverage)"
            if st.sidebar.button(btn_label, key=f"rec_{idx}"):
                st.session_state.selected_detection = det

    if st.session_state.selected_detection:
        selected_detection = st.session_state.selected_detection
        st.markdown('<div class="section-header">📜 Past Detection Details</div>', unsafe_allow_html=True)
        st.success(f"Showing stored results for **{selected_detection.get('filename', 'N/A')}**")

        base64_overlay = selected_detection.get("overlay_image")
        base64_original = selected_detection.get("original_image")
        overlay_img = decode_base64_image(base64_overlay) if base64_overlay else None
        original_img = decode_base64_image(base64_original) if base64_original else None

        col_a, col_b = st.columns(2)
        with col_a:
            if original_img:
                st.image(original_img, caption="Original Image", use_container_width=True)
            else:
                st.info("No original image available in record.")

        with col_b:
            if overlay_img:
                st.image(overlay_img, caption="Detected Oil Spill Overlay", use_container_width=True)
            else:
                st.info("No overlay image available.")

        st.markdown('<div class="section-header">📈 Detection Summary</div>', unsafe_allow_html=True)
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            st.metric("🛢️ Oil Coverage", f"{selected_detection.get('percentage',0):.2f}%")
        with col_m2:
            st.metric("📏 IoU", f"{selected_detection.get('iou',0):.3f}")
        with col_m3:
            st.metric("🎲 Dice", f"{selected_detection.get('dice_coef',0):.3f}")

        st.markdown(
            f"<p style='color:#555;'><strong>Timestamp:</strong> {selected_detection.get('timestamp','--')}</p>",
            unsafe_allow_html=True
        )

        if "technical_report" in selected_detection and selected_detection["technical_report"]:
            with st.expander("📄 Technical Report"):
                st.text(selected_detection["technical_report"])
        st.markdown("<hr>", unsafe_allow_html=True)

    if st.sidebar.button("⬅️ Back to Detection Mode"):
        st.session_state.mode = "predict"
        st.session_state.analyze = False
        st.session_state.selected_detection = None
        st.session_state.last_saved = None
        st.session_state.view_mode = "default"
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div class='sidebar-section'>
    <h3>💡 How to Use</h3>
    <p><strong>1. Upload a satellite image</strong></p>
    <p><strong>2. Adjust settings if needed</strong></p>
    <p><strong>3. Click Detect Oil Spill</strong></p>
    <p><strong>4. Analyze results and download</strong></p>
    <h3>🎯 Tips for Best Results</h3>
    <p><strong>• Use high-resolution satellite images</strong></p>
    <p><strong>• Ensure good contrast between water and potential spills</strong></p>
    <p><strong>• Adjust threshold for sensitivity control</strong></p>
</div>
""", unsafe_allow_html=True)

# ---------------- STOP default UI when viewing a selected record ----------------
# Prevent the main upload/welcome UI from rendering when a record is selected in view mode
if st.session_state.mode == "view" and st.session_state.selected_detection:
    st.stop()

# ---------------- UPLOAD SECTION ----------------
st.markdown('<div class="section-header">📤 Upload Satellite Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drag and drop or click to upload JPG/PNG image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    original_img = Image.open(uploaded_file).convert("RGB")

    # Apply local enhancements for preview only
    enhancer_b = ImageEnhance.Brightness(original_img)
    enhanced_img = enhancer_b.enhance(brightness)
    enhancer_c = ImageEnhance.Contrast(enhanced_img)
    enhanced_img = enhancer_c.enhance(contrast)
    enhancer_s = ImageEnhance.Sharpness(enhanced_img)
    enhanced_img = enhancer_s.enhance(sharpness)

    # Layout: preview + info
    col_preview, col_info = st.columns([2, 1])
    with col_preview:
        st.image(enhanced_img, caption="Enhanced Input Image", use_container_width=True)
    with col_info:
        st.markdown('<div class="subsection-header">📋 Image Info</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-card'>
            <h3>📷 Image Details</h3>
            <p><strong>Filename:</strong> {uploaded_file.name}</p>
            <p><strong>Dimensions:</strong> {original_img.width} × {original_img.height}</p>
            <p><strong>Format:</strong> {uploaded_file.type}</p>
            <p><strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="subsection-header">⚡ Quick Actions</div>', unsafe_allow_html=True)
        # Use standard valid st.button signature and keys; visual style preserved by CSS
        #detect = st.button("🔍 Detect Oil Spill", key="detect_btn")
        #reset = st.button("🔄 Reset Analysis", key="reset_btn")
        detect = st.button("🔍 Detect Oil Spill", use_container_width=True, type="primary")
        reset = st.button("🔄 Reset Analysis", use_container_width=True)

        if reset:
            st.session_state.analyze = False
            st.rerun()

    # Begin analysis when requested
    if detect:
        st.session_state.analyze = True

    if st.session_state.get('analyze', False):
        # Show spinner & progress
        with st.spinner("🔄 Sending image to backend for analysis..."):
            prog = st.progress(0)
            for i in range(40):
                time.sleep(0.01)
                prog.progress(int((i/40)*100))

            try:
                # Prepare image bytes
                buf = io.BytesIO()
                original_img.save(buf, format="PNG")
                buf.seek(0)
                files = {"file": ("image.png", buf.getvalue(), "image/png")}
                params = {"threshold": threshold}

                # POST to backend
                resp = requests.post(BACKEND_URL_PREDICT, files=files, params=params, timeout=120)
                if resp.status_code != 200:
                    st.error(f"Backend error ({resp.status_code}): {resp.text}")
                    st.session_state.analyze = False
                else:
                    data = resp.json()
                    prog.progress(100)

                    # Extract overlay image
                    overlay_img = None
                    if "overlay_image" in data and data["overlay_image"]:
                        overlay_img = decode_base64_image(data["overlay_image"])
                    else:
                        overlay_img = None

                    # If backend sends pred_map as base64 image or flattened array, attempt to reconstruct
                    pred_map_2d = None
                    if "pred_map" in data and data["pred_map"]:
                        try:
                            possible = data["pred_map"]
                            if isinstance(possible, str) and (possible.strip().startswith("/9j") or possible.strip().startswith("iVB") or possible.strip().startswith("data:")):
                                pm_img = decode_base64_image(possible)
                                if pm_img:
                                    pred_map_2d = np.array(pm_img.convert("L"), dtype=float) / 255.0
                            else:
                                arr = np.array(possible)
                                if arr.ndim == 1:
                                    L = arr.size
                                    side = int(np.sqrt(L))
                                    if side * side == L:
                                        pred_map_2d = arr.reshape((side, side))
                                    else:
                                        pred_map_2d = None
                                elif arr.ndim == 2:
                                    pred_map_2d = arr
                        except Exception:
                            pred_map_2d = None

                    # If pred_map missing, attempt heuristic from overlay
                    if pred_map_2d is None and overlay_img is not None:
                        arr_overlay = np.array(overlay_img.convert("RGB")).astype(float)
                        red = arr_overlay[..., 0]
                        green = arr_overlay[..., 1]
                        blue = arr_overlay[..., 2]
                        pred_map_2d = np.clip((red - (green + blue) / 2) / 255.0, 0.0, 1.0)

                    # area metrics (fall back computed)
                    try:
                        oil_pixels = int(data.get("oil_pixels", 0))
                    except Exception:
                        oil_pixels = 0

                    if pred_map_2d is not None:
                        total_pixels = int(pred_map_2d.size)
                    else:
                        total_pixels = int(original_img.width * original_img.height)

                    percentage = float(data.get("percentage", (oil_pixels / total_pixels * 100) if total_pixels else 0.0))

                    dice_value = data.get("dice_coef", None)
                    iou_value = data.get("iou", None)
                    acc_value = data.get("accuracy", None)

                    st.success("✅ Analysis Complete!")

                    # Save confirmation feedback (newly stored in DB)
                    try:
                        if "detection_id" in data:
                            st.session_state.last_saved = data["detection_id"]
                            st.success(f"✅ Detection successfully saved to database (ID: {data['detection_id']})")
                    except Exception:
                        pass

                    # Auto-refresh detections list for sidebar later (best-effort)
                    try:
                        r = requests.get(BACKEND_GET_ALL_PAST_PREDICTIONS, timeout=5)
                        if r.status_code == 200:
                            st.session_state.detections = r.json().get("detections", [])
                    except Exception:
                        pass

                    # ---------------- RESULTS DISPLAY ----------------
                    st.markdown('<div class="section-header">📊 Detection Results</div>', unsafe_allow_html=True)
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(original_img, caption="Original Image", use_container_width=True)
                    with col2:
                        if show_confidence_map and pred_map_2d is not None:
                            fig_heat = create_confidence_heatmap_from_array(pred_map_2d)
                            st.plotly_chart(fig_heat, use_container_width=True)
                        else:
                            if overlay_img:
                                st.image(overlay_img.convert("L"), caption="Binary / Confidence Map", use_container_width=True)
                            else:
                                st.info("No confidence map available.")
                    with col3:
                        if overlay_img:
                            st.image(overlay_img, caption="Detected Oil Spill (overlay)", use_container_width=True)
                        else:
                            st.info("No overlay returned from backend.")

                    # ---------------- METRICS AND ANALYTICS ----------------
                    st.markdown('<div class="section-header">📈 Detailed Analysis</div>', unsafe_allow_html=True)

                    # Compute area_data dict
                    area_data = {
                        "oil_pixels": oil_pixels,
                        "total_pixels": total_pixels,
                        "percentage": percentage,
                        "area_km2": oil_pixels * pixel_size  # rough estimate
                    }

                    col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
                    with col_metrics1:
                        st.markdown(f"""
                        <div class='metric-card' style='background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);'>
                            <h3>🛢️ Oil Coverage</h3>
                            <h2>{area_data['percentage']:.2f}%</h2>
                            <p>{area_data['area_km2']:.2f} km² estimated</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_metrics2:
                        acc_display = f"{acc_value*100:.1f}%" if acc_value is not None else "N/A"
                        st.markdown(f"""
                        <div class='metric-card' style='background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);'>
                            <h3>🎯 Detection Accuracy</h3>
                            <h2>{acc_display}</h2>
                            <p>Pixel-level precision</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_metrics3:
                        iou_display = f"{iou_value:.3f}" if iou_value is not None else "N/A"
                        st.markdown(f"""
                        <div class='metric-card' style='background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);'>
                            <h3>📏 IoU Score</h3>
                            <h2>{iou_display}</h2>
                            <p>Overlap quality</p>
                        </div>
                        """, unsafe_allow_html=True)
                    with col_metrics4:
                        dice_display = f"{dice_value:.3f}" if dice_value is not None else "N/A"
                        st.markdown(f"""
                        <div class='metric-card' style='background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);'>
                            <h3>🎲 Dice Coefficient</h3>
                            <h2>{dice_display}</h2>
                            <p>Similarity measure</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Charts
                    if show_metrics:
                        col_chart1, col_chart2 = st.columns(2)
                        with col_chart1:
                            fig_pie = create_area_chart(area_data)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with col_chart2:
                            if pred_map_2d is not None:
                                hist_fig = px.histogram(x=pred_map_2d.flatten(), nbins=50, title="Confidence Score Distribution", labels={'x':'Confidence','y':'Count'})
                                hist_fig.add_vline(x=threshold, line_dash="dash", annotation_text=f"Threshold: {threshold}")
                                st.plotly_chart(hist_fig, use_container_width=True)
                            else:
                                st.info("No confidence distribution available.")

                    # ---------------- EXPORT SECTION ----------------
                    st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
                    col_d1, col_d2, col_d3 = st.columns(3)
                    # Download mask
                    if overlay_img is not None:
                        mask_img = mask_from_overlay(overlay_img, threshold_val=20)
                        buf_mask = io.BytesIO()
                        mask_img.save(buf_mask, format="PNG")
                        buf_mask.seek(0)
                        col_d1.download_button("⬇️ Download Detection Mask", buf_mask.getvalue(), file_name="oil_spill_mask.png", mime="image/png")
                    else:
                        col_d1.info("No mask available")

                    # Download overlay image
                    if overlay_img is not None:
                        buf_overlay = io.BytesIO()
                        overlay_img.save(buf_overlay, format="PNG")
                        buf_overlay.seek(0)
                        col_d2.download_button("⬇️ Download Overlay Image", buf_overlay.getvalue(), file_name="oil_spill_overlay.png", mime="image/png")
                    else:
                        col_d2.info("No overlay available")

                    # Download report
                    report_text = f"""OIL SPILL DETECTION REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

IMAGE ANALYSIS RESULTS:
- Oil Coverage: {area_data['percentage']:.2f}%
- Estimated Area: {area_data['area_km2']:.2f} km²
- Oil Pixels: {area_data['oil_pixels']}
- Total Pixels: {area_data['total_pixels']}
- Detection Accuracy: {acc_value if acc_value is not None else 'N/A'}
- IoU Score: {iou_value if iou_value is not None else 'N/A'}
- Dice Coefficient: {dice_value if dice_value is not None else 'N/A'}

DETECTION SETTINGS:
- Confidence Threshold: {threshold}
- Pixel Size: {pixel_size} km²
"""
                    col_d3.download_button("📄 Download Analysis Report", report_text, file_name="oil_spill_report.txt", mime="text/plain")

                    # ---------------- TECHNICAL DETAILS ----------------
                    with st.expander("🔍 Technical Details & Interpretation"):
                        st.markdown("""
                        ### 📊 Understanding the Metrics
                        **Dice Coefficient (F1 Score)**  
                        - Measures spatial overlap accuracy  
                        - Range: 0 (no overlap) to 1 (perfect overlap)  
                        - Good: >0.7, Excellent: >0.9

                        **IoU (Intersection over Union)**  
                        - Ratio of correctly predicted area to total area  
                        - More strict than Dice score  
                        - Good: >0.5, Excellent: >0.8

                        **Pixel Accuracy**  
                        - Fraction of correctly classified pixels  
                        - Can be misleading if class imbalance exists

                        ### 🛢️ Oil Spill Severity Assessment
                        - **<1% coverage**: Minor spill
                        - **1-5% coverage**: Moderate spill
                        - **5-10% coverage**: Significant spill
                        - **>10% coverage**: Major environmental incident
                        """)

            except requests.exceptions.RequestException as re:
                st.error(f"Request error when contacting backend: {re}")
                st.session_state.analyze = False
            except Exception as e:
                st.error(f"Processing error: {e}")
                st.session_state.analyze = False

else:
    # No file uploaded (welcome page)
    st.markdown("""
    <div class='welcome-section'>
        <h2>🛢️ Welcome to Oil Spill Detection System</h2>
        <p>Upload a satellite image to start environmental monitoring</p>
    </div>
    """, unsafe_allow_html=True)

    col_demo1, col_demo2, col_demo3 = st.columns(3)
    with col_demo1:
        st.markdown("""
        <div class='content-section'>
            <h3>🌊 Why Monitor Oil Spills?</h3>
            <ul>
            <li>Early detection prevents environmental damage</li>
            <li>Quick response reduces cleanup costs</li>
            <li>Protects marine ecosystems</li>
            <li>Supports regulatory compliance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_demo2:
        st.markdown("""
        <div class='content-section'>
            <h3>🛠️ How It Works</h3>
            <ul>
            <li>Upload satellite imagery</li>
            <li>AI Analysis detects oil patterns</li>
            <li>Visualize spill boundaries</li>
            <li>Quantify impact area</li>
            <li>Export for reporting</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col_demo3:
        st.markdown("""
        <div class='content-section'>
            <h3>📈 Benefits</h3>
            <ul>
            <li>Real-time monitoring</li>
            <li>High accuracy detection</li>
            <li>Automated reporting</li>
            <li>Cost-effective solution</li>
            <li>Scalable for large areas</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>Developed with ❤️ for environmental protection | Powered by Streamlit & TensorFlow</p>
        <p>🛡️ Helping protect our oceans one detection at a time</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Ensure analyze flag exists
if 'analyze' not in st.session_state:
    st.session_state.analyze = False
