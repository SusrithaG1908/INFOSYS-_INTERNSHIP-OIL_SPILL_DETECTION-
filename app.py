# app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import time

# ---------------- CONFIG ----------------
MODEL_PATH = "final_unet_oilspill.h5"
IMG_SIZE = (256, 256)
THRESHOLD = 0.5
# ----------------------------------------

st.set_page_config(
    page_title="Oil Spill Detection", 
    layout="wide", 
    page_icon="🛢️",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CSS ----------------
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
    
    /* Section headers with better visibility */
    .section-header {
        color: #00509e; /* brighter blue for visibility */
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
    
    /* Content sections with better visibility */
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
    
    /* Welcome section */
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
    
    /* Sidebar specific styles */
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

    /* Sidebar text and list styling */
    .sidebar-section p, .sidebar-section li {
        color: #2c3e50 !important;
        font-size: 15px;
        line-height: 1.5;
    }

    .sidebar-section strong {
        color: #00509e;
    }
    
    /* Improve overall sidebar text visibility */
    .css-1d391kg {
        color: #2c3e50 !important;
    }
    
    .stSidebar .stMarkdown {
        color: #2c3e50 !important;
    }
    
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
        color: #00509e !important;
    }
    
    /* Improve overall text visibility */
    .stMarkdown {
        color: #2c3e50;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #2c3e50 !important;
    }
            /* ========================= */
/* EXPANDER STYLING SECTION */
/* ========================= */
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

/* ========================= */
/* EXPANDER STYLING SECTION */
/* ========================= */
[data-testid="stExpander"] {
    background-color: #f8fbfd; /* very light blue background */
    border: 1.5px solid #cce5f6;
    border-radius: 14px !important;
    box-shadow: 0 3px 6px rgba(0, 119, 182, 0.08);
    margin-top: 20px;
    transition: all 0.3s ease;
}

[data-testid="stExpander"]:hover {
    box-shadow: 0 4px 10px rgba(0, 119, 182, 0.2);
    transform: scale(1.01);
}

/* Expander label bar */
[data-testid="stExpander"] > div:first-child {
    background: #0077b6;
    color: white !important;
    font-weight: 700;
    font-size: 18px;
    border-radius: 12px 12px 0 0;
    padding: 10px 12px;
    text-shadow: 0px 1px 2px rgba(0,0,0,0.25);
}

/* Inner content styling */
.expander-content {
    padding: 12px 16px 10px 16px;
    color: #1e2a36;
    font-size: 16px;
    line-height: 1.6;
}

/* Subheadings inside expander */
.expander-content h3 {
    color: #005f8f;
    font-size: 20px;
    font-weight: 750;
    border-left: 4px solid #00b4d8;
    padding-left: 8px;
    margin-top: 12px;
    margin-bottom: 10px;
}

/* Emphasis text */
.expander-content strong {
    color: #004b76;
}

/* Bullet/number list items */
.expander-content li {
    margin-bottom: 5px;
    color: #1e2a36;
}


    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🌊 OIL SPILL DETECTION SYSTEM 🛢️</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered satellite image analysis for environmental protection</p>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------------- CUSTOM METRICS ----------------
@tf.keras.utils.register_keras_serializable(package="Custom")
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

@tf.keras.utils.register_keras_serializable(package="Custom")
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

@tf.keras.utils.register_keras_serializable(package="Custom")
def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

@tf.keras.utils.register_keras_serializable(package="Custom")
def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    return bce + dice_loss(y_true, y_pred)

# ---------------- LOAD MODEL ----------------
@st.cache_resource(show_spinner=False)
def load_ai_model():
    with st.spinner("🚀 Loading AI model..."):
        try:
            model = load_model(
                MODEL_PATH,
                custom_objects={
                    "dice_coef": dice_coef,
                    "iou_metric": iou_metric,
                    "dice_loss": dice_loss,
                    "bce_dice_loss": bce_dice_loss,
                },
                compile=False,
            )
            st.success("✅ Model loaded successfully!")
            return model
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None

model = load_ai_model()

# ---------------- UTILITIES ----------------
def preprocess_image(img_pil, target_size):
    img = img_pil.convert("RGB")
    img_resized = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    arr = np.array(img_resized).astype("float32") / 255.0
    return arr

def get_mask(pred, threshold):
    if pred.ndim == 4:
        pred = pred[0]
    if pred.ndim == 3 and pred.shape[2] == 1:
        pred = pred[..., 0]
    mask = (pred >= threshold).astype(np.uint8)
    return mask, pred

def overlay_mask(img_pil, mask, color=(255, 0, 0), alpha=0.4):
    orig = img_pil.convert("RGBA")
    mask_img = Image.fromarray((mask * 255).astype("uint8")).resize(orig.size, Image.NEAREST)
    overlay = Image.new("RGBA", orig.size, color + (0,))
    alpha_mask = Image.fromarray((mask * int(255 * alpha)).astype("uint8")).resize(orig.size, Image.NEAREST)
    overlay.putalpha(alpha_mask)
    blended = Image.alpha_composite(orig, overlay)
    return blended

def image_to_bytes(pil_img):
    """Convert PIL image to downloadable bytes."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def mask_to_bytes(mask):
    pil = Image.fromarray((mask * 255).astype("uint8"))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

def calc_pixel_accuracy(y_true, y_pred):
    correct = np.sum(y_true == y_pred)
    total = y_true.size
    return correct / total

def calculate_area_metrics(mask, pixel_size_km=0.01):
    """Calculate estimated area of oil spill"""
    oil_pixels = np.sum(mask)
    total_pixels = mask.size
    percentage = (oil_pixels / total_pixels) * 100
    
    # Estimate area (assuming each pixel represents pixel_size_km²)
    area_km2 = oil_pixels * pixel_size_km
    
    return {
        'oil_pixels': oil_pixels,
        'total_pixels': total_pixels,
        'percentage': percentage,
        'area_km2': area_km2
    }

def create_area_chart(area_data):
    """Create a pie chart showing oil vs water distribution"""
    labels = ['Oil Spill', 'Water']
    values = [area_data['oil_pixels'], area_data['total_pixels'] - area_data['oil_pixels']]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values,
        hole=.3,
        marker_colors=['#FF4B4B', '#0077b6']
    )])
    fig.update_layout(
        title="Area Distribution",
        showlegend=True,
        height=300
    )
    return fig

def create_confidence_heatmap(pred_map):
    """Create a proper heatmap visualization for confidence scores"""
    fig = px.imshow(pred_map, 
                   color_continuous_scale='viridis',
                   title="Confidence Heatmap",
                   aspect='equal')
    fig.update_layout(coloraxis_colorbar=dict(title="Confidence"))
    return fig

# ---------------- SIDEBAR ----------------
st.sidebar.markdown("""
<div class='sidebar-section'>
    <h3>⚙️ Settings</h3>
</div>
""", unsafe_allow_html=True)

# Detection Settings
st.sidebar.markdown("""
<div class='sidebar-section'>
    <h3>🔍 Detection Settings</h3>
</div>
""", unsafe_allow_html=True)
threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, THRESHOLD, 0.05)
alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.4, 0.05)

# Image Enhancement
st.sidebar.markdown("""
<div class='sidebar-section'>
    <h3>🖼️ Image Enhancement</h3>
</div>
""", unsafe_allow_html=True)
brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1)
sharpness = st.sidebar.slider("Sharpness", 0.0, 2.0, 1.0, 0.1)

# Analysis Options
st.sidebar.markdown("""
<div class='sidebar-section'>
    <h3>📊 Analysis Options</h3>
</div>
""", unsafe_allow_html=True)
pixel_size = st.sidebar.number_input("Pixel size (km²)", 0.001, 1.0, 0.01, 0.001, 
                                   help="Estimated area each pixel represents in square kilometers")
show_confidence_map = st.sidebar.checkbox("Show Confidence Map", True)
show_metrics = st.sidebar.checkbox("Show Detailed Metrics", True)

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

# ---------------- MAIN UI ----------------
# Feature Introduction
col_intro1, col_intro2, col_intro3 = st.columns(3)
with col_intro1:
    st.markdown("""
    <div class='feature-card'>
        <h3>🚀 Fast Detection</h3>
        <p>Real-time AI analysis with high accuracy</p>
    </div>
    """, unsafe_allow_html=True)
with col_intro2:
    st.markdown("""
    <div class='feature-card'>
        <h3>📊 Detailed Analytics</h3>
        <p>Comprehensive metrics and area calculations</p>
    </div>
    """, unsafe_allow_html=True)
with col_intro3:
    st.markdown("""
    <div class='feature-card'>
        <h3>💾 Export Results</h3>
        <p>Download masks and analysis reports</p>
    </div>
    """, unsafe_allow_html=True)

# File Upload with Enhanced UI
st.markdown('<div class="section-header">📤 Upload Satellite Image</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Drag and drop or click to upload JPG/PNG image", 
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file:
    # Image Preview and Enhancement
    col_preview, col_info = st.columns([2, 1])
    
    with col_preview:
        img = Image.open(uploaded_file)
        
        # Apply enhancements
        enhancer_b = ImageEnhance.Brightness(img)
        enhancer_c = ImageEnhance.Contrast(enhancer_b.enhance(brightness))
        enhancer_s = ImageEnhance.Sharpness(enhancer_c.enhance(contrast))
        enhanced_img = enhancer_s.enhance(sharpness)
        
        st.image(enhanced_img, caption="Enhanced Input Image", use_container_width=True)
    
    with col_info:
        st.markdown('<div class="subsection-header">📋 Image Info</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-card'>
            <h3>📷 Image Details</h3>
            <p><strong>Dimensions:</strong> {img.size[0]} × {img.size[1]}</p>
            <p><strong>Format:</strong> {img.format}</p>
            <p><strong>Mode:</strong> {img.mode}</p>
            <p><strong>Size:</strong> {uploaded_file.size / 1024:.1f} KB</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        st.markdown('<div class="subsection-header">⚡ Quick Actions</div>', unsafe_allow_html=True)
        if st.button("🔍 Detect Oil Spill", use_container_width=True, type="primary"):
            st.session_state.analyze = True
        if st.button("🔄 Reset Analysis", use_container_width=True):
            st.session_state.analyze = False
            st.rerun()

    # Analysis Section
    if st.session_state.get('analyze', False) and model is not None:
        with st.spinner("🔄 Analyzing image for oil spills..."):
            # Progress bar
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Processing
            x = preprocess_image(enhanced_img, IMG_SIZE)
            x_batch = np.expand_dims(x, 0)
            pred = model.predict(x_batch, verbose=0)
            mask, pred_map = get_mask(pred, threshold)
            overlayed = overlay_mask(img, mask, alpha=alpha)
            
            # Calculate metrics
            area_data = calculate_area_metrics(mask, pixel_size)
            dice_value = float(dice_coef(tf.constant(mask, dtype=tf.float32), 
                                       tf.constant(pred_map > threshold, dtype=tf.float32)))
            iou_value = float(iou_metric(tf.constant(mask, dtype=tf.float32), 
                                       tf.constant(pred_map > threshold, dtype=tf.float32)))
            acc_value = calc_pixel_accuracy(mask, (pred_map > threshold).astype(np.uint8))

        st.success("✅ Analysis Complete!")
        
        # Results Display
        st.markdown('<div class="section-header">📊 Detection Results</div>', unsafe_allow_html=True)
        
        # Image Comparison
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        with col2:
            if show_confidence_map:
                # Use Plotly for confidence heatmap instead of direct image display
                fig = create_confidence_heatmap(pred_map)
                st.plotly_chart(fig, use_container_width=True)
            else:
                binary_mask = Image.fromarray((mask * 255).astype('uint8'))
                st.image(binary_mask, caption="Binary Mask", use_container_width=True)
        with col3:
            st.image(overlayed, caption="Detected Oil Spill", use_container_width=True)

        # Metrics and Analytics
        st.markdown('<div class="section-header">📈 Detailed Analysis</div>', unsafe_allow_html=True)
        
        # Key Metrics - Different colors for each metric card
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
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);'>
                <h3>🎯 Detection Accuracy</h3>
                <h2>{acc_value*100:.1f}%</h2>
                <p>Pixel-level precision</p>
            </div>
            """, unsafe_allow_html=True)
        with col_metrics3:
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #9b59b6 0%, #8e44ad 100%);'>
                <h3>📏 IoU Score</h3>
                <h2>{iou_value:.3f}</h2>
                <p>Overlap quality</p>
            </div>
            """, unsafe_allow_html=True)
        with col_metrics4:
            st.markdown(f"""
            <div class='metric-card' style='background: linear-gradient(135deg, #f39c12 0%, #e67e22 100%);'>
                <h3>🎲 Dice Coefficient</h3>
                <h2>{dice_value:.3f}</h2>
                <p>Similarity measure</p>
            </div>
            """, unsafe_allow_html=True)

        # Charts and Visualizations
        if show_metrics:
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                # Area distribution pie chart
                fig_pie = create_area_chart(area_data)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col_chart2:
                # Confidence distribution
                fig_hist = px.histogram(x=pred_map.flatten(), 
                                      nbins=50,
                                      title="Confidence Score Distribution",
                                      labels={'x': 'Confidence', 'y': 'Count'})
                fig_hist.add_vline(x=threshold, line_dash="dash", line_color="red", 
                                 annotation_text=f"Threshold: {threshold}")
                st.plotly_chart(fig_hist, use_container_width=True)

        # Download Section
        st.markdown('<div class="section-header">💾 Export Results</div>', unsafe_allow_html=True)
        
        col_d1, col_d2, col_d3 = st.columns(3)
        with col_d1:
            st.download_button(
                "⬇️ Download Detection Mask",
                mask_to_bytes(mask),
                file_name="oil_spill_mask.png",
                mime="image/png",
                use_container_width=True
            )
        with col_d2:
            st.download_button(
                "⬇️ Download Overlay Image",
                image_to_bytes(overlayed),
                file_name="oil_spill_overlay.png",
                mime="image/png",
                use_container_width=True
            )
        with col_d3:
            # Generate report
            report = f"""
            OIL SPILL DETECTION REPORT
            Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            
            IMAGE ANALYSIS RESULTS:
            - Oil Coverage: {area_data['percentage']:.2f}%
            - Estimated Area: {area_data['area_km2']:.2f} km²
            - Detection Confidence: {acc_value*100:.1f}%
            - IoU Score: {iou_value:.3f}
            - Dice Coefficient: {dice_value:.3f}
            
            DETECTION SETTINGS:
            - Confidence Threshold: {threshold}
            - Pixel Size: {pixel_size} km²
            """
            st.download_button(
                "📄 Download Analysis Report",
                report,
                file_name="oil_spill_report.txt",
                mime="text/plain",
                use_container_width=True
            )

        # Expert Information
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

elif model is None:
    st.error("❌ Model could not be loaded. Please check if the model file exists and try again.")

else:
    # Welcome and instructions when no file uploaded
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

# Initialize session state
if 'analyze' not in st.session_state:
    st.session_state.analyze = False