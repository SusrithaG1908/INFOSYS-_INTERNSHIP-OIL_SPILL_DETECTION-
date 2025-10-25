# app.py
import streamlit as st
import numpy as np
from PIL import Image, ImageEnhance
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
MODEL_PATH = "final_unet_oilspill.h5"
IMG_SIZE = (256, 256)
THRESHOLD = 0.5
# ----------------------------------------

st.set_page_config(page_title="Oil Spill Detection", layout="wide", page_icon="🛢️")
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
    </style>
    """, unsafe_allow_html=True)

st.markdown('<h1 class="main-title">🌊 OIL SPILL DETECTION SYSTEM 🛢️</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-powered satellite image analysis for environmental protection</p>', unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown(
    """
    <style>
    .title {text-align:center; font-size:40px; font-weight:800; color:#003366;}
    .subtitle {text-align:center; color:#555; font-size:18px;}
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown('<p class="title">🌊 Oil Spill Detection using Deep Learning 🛢️</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Upload a satellite image to automatically detect oil spill regions.</p>', unsafe_allow_html=True)

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
with st.spinner("🚀 Loading model..."):
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

# ---------------- UTILITIES ----------------
def preprocess_image(img_pil, target_size):
    img = img_pil.convert("RGB")
    img_resized = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    arr = np.array(img_resized).astype("float32") / 255.0
    return arr

def get_mask(pred):
    if pred.ndim == 4:
        pred = pred[0]
    if pred.ndim == 3 and pred.shape[2] == 1:
        pred = pred[..., 0]
    mask = (pred >= THRESHOLD).astype(np.uint8)
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

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Settings")
alpha = st.sidebar.slider("Overlay Transparency", 0.0, 1.0, 0.4, 0.05)
brightness = st.sidebar.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
contrast = st.sidebar.slider("Contrast", 0.5, 2.0, 1.0, 0.1)

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Tip:** Adjust brightness/contrast for better visibility of oil regions.\n\n"
    "🧠 Model: UNet (trained on oil spill satellite imagery)"
)

# ---------------- MAIN UI ----------------
uploaded_file = st.file_uploader("📤 Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    enhancer_b = ImageEnhance.Brightness(img)
    enhancer_c = ImageEnhance.Contrast(enhancer_b.enhance(brightness))
    enhanced_img = enhancer_c.enhance(contrast)

    st.image(enhanced_img, caption="Enhanced Input Image", use_container_width=True)

    if st.button("🔍 Detect Oil Spill", use_container_width=True):
        with st.spinner("Running segmentation... ⏳"):
            x = preprocess_image(enhanced_img, IMG_SIZE)
            x_batch = np.expand_dims(x, 0)
            pred = model.predict(x_batch)
            mask, pred_map = get_mask(pred)
            overlayed = overlay_mask(img, mask, alpha=alpha)

        # ---- Columns ----
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(img, caption="Original Image", use_container_width=True)
        with col2:
            st.image(pred_map, caption="Prediction Probability Map", use_container_width=True)
        with col3:
            st.image(overlayed, caption="Detected Oil Spill", use_container_width=True)

        # ---- Metrics ----
        dice_value = float(dice_coef(tf.constant(mask, dtype=tf.float32), tf.constant(pred_map > THRESHOLD, dtype=tf.float32)))
        iou_value = float(iou_metric(tf.constant(mask, dtype=tf.float32), tf.constant(pred_map > THRESHOLD, dtype=tf.float32)))
        acc_value = calc_pixel_accuracy(mask, (pred_map > THRESHOLD).astype(np.uint8))

        st.markdown("### 📊 Detection Metrics")
        colm1, colm2, colm3 = st.columns(3)
        colm1.metric("Dice Coefficient", f"{dice_value:.3f}")
        colm2.metric("IoU Score", f"{iou_value:.3f}")
        colm3.metric("Pixel Accuracy", f"{acc_value*100:.2f}%")

        st.markdown("### 💾 Download Results")
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "⬇️ Download Mask (PNG)",
                mask_to_bytes(mask),
                file_name="oil_spill_mask.png",
                mime="image/png",
                use_container_width=True
            )
        with col_d2:
            st.download_button(
                "⬇️ Download Predicted Image (PNG)",
                image_to_bytes(overlayed),
                file_name="oil_spill_overlay.png",
                mime="image/png",
                use_container_width=True
            )

        with st.expander("📘 What These Metrics Mean"):
            st.markdown(
                """
                - **Dice Coefficient**: Measures overlap between predicted and actual oil regions (1.0 = perfect).  
                - **IoU (Intersection over Union)**: Ratio of correctly predicted area to the total combined area.  
                - **Pixel Accuracy**: Fraction of correctly classified pixels.  
                """
            )

else:
    st.info("👆 Upload a satellite image to detect oil spills.")

st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#777;'>Developed by Durga Gudimetla | Powered by Streamlit & TensorFlow</p>",
    unsafe_allow_html=True,
)
