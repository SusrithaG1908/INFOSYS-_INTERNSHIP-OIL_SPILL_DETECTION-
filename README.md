
# 🛢️ Oil Spill Detection and Monitoring using CNN (U-Net)
## 📌 Project Overview

Oil spills pose a serious threat to marine ecosystems, coastal regions, and local economies. Traditional detection methods such as manual inspection of satellite images or physical patrolling are time-consuming, labor-intensive, and often delayed.

This project develops an AI-powered oil spill detection system using Convolutional Neural Networks (CNN) and U-Net for image segmentation. The model analyzes satellite imagery to detect and localize oil spills efficiently, supporting real-time monitoring and rapid intervention.

---
## 🎯 Objectives

- Automatically detect and segment oil spills in satellite images.

- Learn visual patterns specific to oil-contaminated regions.

- Generate segmentation masks highlighting affected areas.

- Assist environmental monitoring agencies with early detection.

- Provide a deployable model for real-time applications.

---

## ⚙️ Tech Stack

- Language: Python

- Libraries: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Seaborn

- Platform: Google Colab

---

## 🔄 Project Workflow: Oil Spill Detection using CNN (U-Net)

```text
 Data Collection → Data Preprocessing → U-Net Model Development → Model Training → Model Evaluation → Visualization → Deployment
```


### 1️⃣ Data Collection

- Acquire satellite images of ocean regions.

- Use Kaggle Oil Spill Detection Dataset (with masks).

- Organize into train / validation / test sets.
  
📂 Dataset

```text
dataset/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
├── test/
    ├── images/
    └── masks/

```

### 2️⃣ Data Preprocessing

- Resize images and masks to a fixed size (e.g., 128×128 or 256×256).

- Normalize pixel values (scale between 0 and 1).

- Augmentation: rotation, flip, zoom, brightness adjustment → increases dataset variety.

### 3️⃣ Model Development (U-Net CNN)

- Encoder (Downsampling): Extracts features from the image (Conv → ReLU → MaxPooling).

- Bottleneck: Deep representation of features.

- Decoder (Upsampling): Reconstructs segmentation mask using transposed convolutions.

- Skip Connections: Combine encoder features with decoder features for better localization.

- Output Layer: Sigmoid activation → binary mask (oil spill vs. non-oil spill).

### 4️⃣ Model Training

Loss Functions:

- Dice Loss

- Binary Cross-Entropy (BCE) (sometimes BCE + Dice combined)

- Optimizer: Adam

Batch Size & Epochs: Tuned for performance.

### 5️⃣ Model Evaluation

Metrics used:

- Accuracy – overall performance

- IoU (Intersection over Union) – overlap between prediction & ground truth

- Dice Coefficient – similarity measure

- Precision / Recall / F1-score – quality of detection

### 6️⃣ Visualization

- Training & validation loss/accuracy curves.

- Predicted segmentation masks vs. ground truth masks.

- Heatmaps to highlight detected oil spill regions.

---


## 🚀 How to Run

Clone this repository:

git clone https://github.com/your-username/oil-spill-detection.git
cd oil-spill-detection


Install dependencies:

pip install -r requirements.txt


Open Jupyter/Colab and run:

OilSpillDetectionMonitoring.ipynb

