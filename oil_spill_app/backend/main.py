from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model
import base64
import os
from dotenv import load_dotenv


load_dotenv()  # 👈 loads variables from .env file into environment

# ---------------- CONFIG ----------------
MODEL_PATH = os.getenv("MODEL_PATH", "final_unet_oilspil.h5")
IMG_SIZE = (256, 256)
THRESHOLD = 0.5
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "oil_spill_db"

# ---------------- FASTAPI SETUP ----------------
app = FastAPI(title="Oil Spill Detection Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow Streamlit or other frontends
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- DATABASE SETUP ----------------
try:
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    db = client[DB_NAME]
    detections_collection = db["detections"]
    client.server_info()  # force connection test
    print("✅ Connected to MongoDB successfully!")
except Exception as e:
    print(f"⚠️ MongoDB connection failed: {e}")
    detections_collection = None

# ---------------- CUSTOM METRICS ----------------
@tf.keras.utils.register_keras_serializable(package="Custom")
def dice_coef(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (
        tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth
    )

@tf.keras.utils.register_keras_serializable(package="Custom")
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
    return intersection / (union + 1e-7)

# ---------------- LOAD MODEL ----------------
print("🔄 Loading AI model...")
model = load_model(
    MODEL_PATH,
    custom_objects={"dice_coef": dice_coef, "iou_metric": iou_metric},
    compile=False,
)
print("✅ Model loaded successfully!")

# ---------------- HELPERS ----------------
def preprocess_image(img_pil):
    img = img_pil.convert("RGB")
    img_resized = img.resize(IMG_SIZE, Image.BILINEAR)
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
    overlay = Image.new("RGBA", orig.size, color + (0,))
    alpha_mask = Image.fromarray((mask * int(255 * alpha)).astype("uint8")).resize(
        orig.size, Image.NEAREST
    )
    overlay.putalpha(alpha_mask)
    return Image.alpha_composite(orig, overlay)

def encode_image(img_pil):
    buf = io.BytesIO()
    img_pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ---------------- API ENDPOINT ----------------
@app.post("/predict")
async def predict_image(file: UploadFile = File(...), threshold: float = THRESHOLD):
    try:
        # 1️⃣ Read and preprocess
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        x = preprocess_image(img)
        x_batch = np.expand_dims(x, 0)

        # 2️⃣ Model inference
        pred = model.predict(x_batch, verbose=0)
        mask, _ = get_mask(pred, threshold)

        # 3️⃣ Generate overlay and encode
        overlayed = overlay_mask(img, mask)
        encoded_original = encode_image(img)
        encoded_overlay = encode_image(overlayed)
        encoded_mask = encode_image(Image.fromarray((mask * 255).astype("uint8")))

        # 4️⃣ Compute metrics
        oil_pixels = int(np.sum(mask))
        total_pixels = int(mask.size)
        percentage = (oil_pixels / total_pixels) * 100
        dice_value = float(dice_coef(tf.convert_to_tensor(mask, dtype=tf.float32),
                                     tf.convert_to_tensor(mask, dtype=tf.float32)).numpy())
        iou_value = float(iou_metric(tf.convert_to_tensor(mask, dtype=tf.float32),
                                     tf.convert_to_tensor(mask, dtype=tf.float32)).numpy())

        # 5️⃣ Prepare record for DB
        record = {
            "timestamp": datetime.utcnow(),
            "oil_pixels": oil_pixels,
            "total_pixels": total_pixels,
            "percentage": percentage,
            "iou": iou_value,
            "dice_coef": dice_value,
            "threshold": threshold,
            "original_image": encoded_original,
            "overlay_image": encoded_overlay,
            "mask_image": encoded_mask,
            "report_text": f"Detected oil spill coverage of {percentage:.2f}% with IOU={iou_value:.3f} and Dice={dice_value:.3f}.",
        }

        # 6️⃣ Save to MongoDB (if connected)
        if detections_collection is not None:
            detections_collection.insert_one(record)
            print("🗄️ Record saved to MongoDB.")
        else:
            print("⚠️ MongoDB not available; skipping record save.")

        # 7️⃣ Send response to frontend
        return JSONResponse({
            "oil_pixels": oil_pixels,
            "total_pixels": total_pixels,
            "percentage": percentage,
            "iou": iou_value,
            "dice_coef": dice_value,
            "overlay_image": encoded_overlay,
            "mask_image": encoded_mask,
        })

    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


# 🧾 GET: All detection summaries (lightweight, no large images)
@app.get("/get_all_detections")
def get_all_detections():
    """Retrieve all stored detection summaries (without large images)."""
    if detections_collection is None:
        return JSONResponse({"error": "MongoDB not connected"}, status_code=500)

    docs = list(
        detections_collection.find({}, {"mask_image": 1, "percentage": 1, "iou": 1,
                                "dice_coef": 1, "timestamp": 1, "original_image": 1}).sort("timestamp", -1)
    )

    for d in docs:
        d["_id"] = str(d["_id"])
        if "timestamp" in d and hasattr(d["timestamp"], "isoformat"):
            d["timestamp"] = d["timestamp"].isoformat()

    return {"detections": docs}   # ✅ Correct key matches frontend


# 🧩 GET: Specific detection with images & details
@app.get("/get_detection/{record_id}")
def get_detection(record_id: str):
    """Retrieve full record with images and analysis details."""
    if detections_collection is None:
        return JSONResponse({"error": "MongoDB not connected"}, status_code=500)

    try:
        doc = detections_collection.find_one({"_id": ObjectId(record_id)})
    except Exception as e:
        return JSONResponse({"error": f"Invalid record ID: {e}"}, status_code=400)

    if not doc:
        return JSONResponse({"error": "Record not found"}, status_code=404)

    # Convert ObjectId and datetime
    doc["_id"] = str(doc["_id"])
    if "timestamp" in doc and hasattr(doc["timestamp"], "isoformat"):
        doc["timestamp"] = doc["timestamp"].isoformat()

    return {"detection": doc}  # ✅ Match key expected by frontend


@app.get("/")
def root():
    return {"message": "✅ Oil Spill Detection Backend is running!"}
