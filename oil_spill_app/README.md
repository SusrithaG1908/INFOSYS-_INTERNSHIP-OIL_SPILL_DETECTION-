# 🌊 AI-Driven Oil Spill Detection and Monitoring System

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red?logo=streamlit)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green?logo=fastapi)
![MongoDB](https://img.shields.io/badge/Database-MongoDB-brightgreen?logo=mongodb)
![Render](https://img.shields.io/badge/Deployment-Render-purple?logo=render)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

---

## 🛰️ Overview

An end-to-end intelligent system that automatically identifies and segments **oil spills from satellite imagery** using deep learning — enabling faster and more accurate environmental monitoring.

---

## 🌍 Problem Statement

Oil spills cause severe environmental damage, affecting marine ecosystems, fisheries, and coastal communities.  
Traditional detection methods rely on **manual satellite image inspection** or **field surveys**, which are:

- Time-consuming
- Costly and inefficient
- Prone to human errors

---

## 🎯 Objective

To build an **AI-powered Oil Spill Detection & Monitoring System** that can:

- Automatically detect oil spills from satellite imagery
- Highlight affected regions using semantic segmentation
- Provide real-time visualization via a user-friendly web interface
- Log detection results and analytics in a database for future monitoring

---

## 🧩 System Architecture

**Pipeline Overview:**

Satellite Image → Preprocessing → U-Net Model → FastAPI Backend → MongoDB → Streamlit Frontend → User

### 🔹 Components

- **Data Source:** Sentinel-1 SAR / MODIS / Kaggle datasets
- **Preprocessing:** Normalization, resizing, speckle noise reduction, and augmentation
- **Model:** U-Net segmentation CNN for oil spill detection
- **Backend:** FastAPI for inference and data management
- **Frontend:** Streamlit for visualization and user interaction
- **Database:** MongoDB for storing input images, results, and reports

---

## ⚙️ Technology Stack

| Layer                  | Tools / Frameworks Used       |
| :--------------------- | :---------------------------- |
| **Language**           | Python                        |
| **ML / DL Frameworks** | TensorFlow, Keras             |
| **Model Architecture** | U-Net (Semantic Segmentation) |
| **Data Handling**      | NumPy, Pandas, OpenCV, PIL    |
| **Visualization**      | Matplotlib, Plotly            |
| **Backend**            | FastAPI                       |
| **Frontend**           | Streamlit                     |
| **Database**           | MongoDB                       |
| **Deployment**         | Render Cloud                  |
| **Environment Config** | `.env`, Virtual Environment   |

---

## 🧠 Machine Learning Pipeline

1. **Data Collection** – Oil Spill dataset (Kaggle / Sentinel-1)
2. **Preprocessing & Augmentation** – Resize to 256×256, normalize, denoise, and apply rotations/flips
3. **Model Design** – U-Net encoder–decoder segmentation architecture
4. **Training** – Dice + Binary Cross-Entropy loss, Adam optimizer
5. **Evaluation Metrics** – IoU, Dice Coefficient, Precision, Recall
6. **Visualization** – Overlay masks on satellite images for clear detection
7. **Deployment** – Integrated with FastAPI backend and Streamlit frontend

---

## 🚀 Application Workflow

1️⃣ User uploads a satellite image via **Streamlit UI**  
2️⃣ Image is sent to the **FastAPI backend**  
3️⃣ The trained **U-Net model** performs segmentation  
4️⃣ The system returns:

- Original image
- Segmentation mask
- Overlayed detection image
- Technical summary (IoU, Dice, timestamp)  
  5️⃣ **MongoDB** stores results for historical reference

---

## 💻 How to Run Locally

### 1️⃣ Clone the Repository

```bash
git lfs install
git clone https://github.com/<your-org>/oil-spill-detection.git
cd oil-spill-detection
```

### 2️⃣ Create a Virtual Environment

### Windows (CMD / PowerShell):

```
cd backend

# create venv
python -m venv venv

# activate (PowerShell)
venv\Scripts\activate
```

### 3️⃣ Upgrade pip and Install Dependencies

```
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4️⃣ Create .env in backend/

### Create a file backend/.env with:

```
MODEL_PATH=final_unet_oilspill.h5
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=oilspill_db
```

## Frontend (Streamlit)

```
cd frontend

# create venv
python -m venv venv

# activate (PowerShell)
venv\Scripts\activate
```

### 3️⃣ Upgrade pip and Install Dependencies

```
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 4️⃣ Create .env in frontend/

### Create a file frontend/.env with:

```
BACKEND_URL=http://127.0.0.1:8000
```

### 5️⃣ Run Backend

```
cd backend
python -m uvicorn main:app --reload --port 8000
```

---

## Frontend (Streamlit)

### 1. Create another Web Service for frontend.

### 2. Set build command:

```
pip install -r requirements.txt
```

### 3. Set start command:

```
streamlit run app.py --server.port 10000 --server.address 0.0.0.0
```

### 4. Set BACKEND_URL in frontend .env to your backend's Render URL.

### 5.Deploy.

---

## 📊 Results & Visualizations

### Input Image Predicted Output

### Model Metrics

|      Metric      | Value |
| :--------------: | :---: |
|       IoU        | ~0.89 |
| Dice Coefficient | ~0.91 |
|    Precision     | ~0.93 |
|      Recall      | ~0.88 |

---

## 🌟 Highlights

### > End-to-end pipeline: data → model → API → UI

### > Real-time oil spill detection with U-Net segmentation

### > Streamlit frontend for easy interaction and reporting

### > MongoDB storage for audit and historical analysis

### > Render deployment for accessibility

---

## 🧩 Key Learnings

### > Implemented full ML lifecycle: preprocessing, training, evaluation, deployment

### > Gained expertise in semantic segmentation for satellite imagery

### > Integrated FastAPI, Streamlit, and MongoDB in a production-like setup

### > Learned cloud deployment and secure environment configuration

---

## 👩‍💻 Contributors

### Developed by: [Susritha Gudimetla]

### Guided by: [Namala Eekshita / Infosys]

### Duration: 8 Weeks Internship

### 📧 Contact: gudimetlasusritha@gmail.com

### GitHub: https://github.com/SusrithaG1908

---

## 📜 License

### This project is for educational and research purposes.

### © 2025 Susritha Gudimetla. All rights reserved.

---
