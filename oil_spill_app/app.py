# app.py
import streamlit as st
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

# ---------------- CONFIG ----------------
MODEL_PATH = "final_unet_oilspill.h5"  # downloaded model path
IMG_SIZE = (256, 256)                  # set this to match your model input size
THRESHOLD = 0.5                        # probability threshold for binary mask
# ----------------------------------------

st.set_page_config(page_title="Oil Spill Detection", layout="wide")
st.title("Oil Spill Detection App 🌊🛢️")

# ---------------- CUSTOM METRICS ----------------
# Same as your training
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
# ------------------------------------------------

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_trained_model():
    model = load_model(
        MODEL_PATH,
        custom_objects={
            "dice_coef": dice_coef,
            "iou_metric": iou_metric,
            "dice_loss": dice_loss,
            "bce_dice_loss": bce_dice_loss
        },
        compile=False
    )
    return model

model = load_trained_model()
st.success("✅ Model loaded successfully.")

# ---------------- PREPROCESS IMAGE ----------------
def preprocess_image(img_pil, target_size):
    img = img_pil.convert("RGB")
    img_resized = img.resize((target_size[1], target_size[0]), Image.BILINEAR)
    arr = np.array(img_resized).astype("float32") / 255.0
    return arr

# ---------------- POSTPROCESS PREDICTION ----------------
def get_mask(pred):
    # Handle single output channel
    if pred.ndim == 4:
        pred = pred[0]  # remove batch
    if pred.ndim == 3 and pred.shape[2] == 1:
        pred = pred[..., 0]
    mask = (pred >= THRESHOLD).astype(np.uint8)
    return mask

# ---------------- OVERLAY ----------------
def overlay_mask(img_pil, mask, color=(255,0,0), alpha=0.5):
    orig = img_pil.convert("RGBA")
    mask_img = Image.fromarray((mask*255).astype("uint8")).resize(orig.size, Image.NEAREST)
    overlay = Image.new("RGBA", orig.size, color + (0,))
    alpha_mask = Image.fromarray((mask*int(255*alpha)).astype("uint8")).resize(orig.size, Image.NEAREST)
    overlay.putalpha(alpha_mask)
    blended = Image.alpha_composite(orig, overlay)
    return blended

def mask_to_bytes(mask):
    pil = Image.fromarray((mask*255).astype("uint8"))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf.getvalue()

# ---------------- STREAMLIT UI ----------------
uploaded_file = st.file_uploader("Upload an image (jpg/png)", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Original Image", use_column_width=True)

    if st.button("Run Prediction"):
        with st.spinner("Detecting oil spill..."):
            x = preprocess_image(img, IMG_SIZE)
            x_batch = np.expand_dims(x, 0)
            pred = model.predict(x_batch)
            mask = get_mask(pred)
            overlayed = overlay_mask(img, mask, alpha=0.5)
            
            st.image(overlayed, caption="Oil Spill Prediction", use_column_width=True)
            st.download_button("Download Mask (PNG)", mask_to_bytes(mask), file_name="oil_spill_mask.png", mime="image/png")
