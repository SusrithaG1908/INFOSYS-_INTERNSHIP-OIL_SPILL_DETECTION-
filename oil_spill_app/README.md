🌊 AI-Driven Oil Spill Detection and Monitoring System

An AI-powered environmental monitoring system that automatically detects and segments oil spills from satellite imagery using deep learning.
This project combines machine learning (U-Net segmentation), FastAPI backend, Streamlit frontend, and MongoDB to provide a real-time oil spill detection and visualization platform.

🧭 Table of Contents

Problem Statement

Objectives

System Architecture

Technology Stack

Machine Learning Pipeline

Application Workflow

How to Run Locally

Deployment (Render)

Results & Visualizations

Key Learnings

Contributors

🧩 Problem Statement

Oil spills pose a serious threat to marine ecosystems, coastal regions, and local economies. Traditional detection methods such as manual inspection of satellite images or physical patrolling are time-consuming, labor-intensive, and often delayed.

This project aims to develop an AI-driven solution to detect, localize, and visualize oil spills in near real-time using satellite imagery and deep learning segmentation models using Convolutional Neural Networks (CNN) and U-Net.

🎯 Objectives

Automatically detect and segment oil spills from satellite images.

Learn oil spill visual patterns using deep convolutional neural networks (CNN/U-Net).

Provide clear segmentation masks highlighting affected regions.

Enable rapid detection and response for environmental protection agencies.

Deliver results via a user-friendly web application (Streamlit + FastAPI).

🧱 System Architecture
      ┌────────────────────┐
      │  Satellite Images  │
      └────────┬───────────┘
               │
        Data Preprocessing
               │
         Deep Learning Model
        (U-Net Segmentation)
               │
         FastAPI Backend
               │
      Streamlit Frontend UI
               │
         MongoDB (Storage)


⚙️ Technology Stack
Layer	Tools / Frameworks Used
Language	Python
ML / DL Frameworks	TensorFlow, Keras
Model Architecture	U-Net (Semantic Segmentation)
Data Handling & Visualization	NumPy, Pandas, Matplotlib, OpenCV, PIL
Web Frameworks	FastAPI (Backend), Streamlit (Frontend)
Database	MongoDB
Deployment	Render Cloud
Environment Management	.env, Virtual Environment

🧠 Machine Learning Pipeline

1. Data Collection
2. Oil Spill Detection dataset from Kaggle / Sentinel-1 SAR / NOAA sources.
3. Organized into training, validation, and test directories.
4. Data Preprocessing & Augmentation
5. Resize images to 256×256.
6. Normalize pixel values, remove speckle noise.
7. Augment data using rotation, flipping, scaling, contrast, and brightness adjustments.
8. Model Development (U-Net)
9. Encoder–decoder architecture for image segmentation.
10. Dice + Binary Cross-Entropy loss functions.
11. Optimized using Adam optimizer.
12. Training & Evaluation
13. Metrics: Accuracy, IoU, Dice Coefficient, Precision, Recall.
14. Visualized results using Matplotlib and overlay maps.
15. Deployment
16. Saved trained model as .h5 and .keras.
17. Integrated with FastAPI backend for inference.
18. Streamlit frontend for visualization.


🚀 Application Workflow

1. User uploads a satellite image via Streamlit UI.
2. Image is sent to the FastAPI backend.
3. The trained U-Net model performs segmentation to identify oil spills.
4. The system returns:
    Original image
    Segmentation mask
    Overlayed detection image
    Technical summary (IoU, Dice score, timestamp)
5. Results are stored in MongoDB for record-keeping and reporting.


💻 How to Run Locally
1️⃣ Clone Repository
git clone https://github.com/<org>>/oil-spill-detection.git
cd oil-spill-detection

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate (#Windows)

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Setup Environment Variables
Create a .env file in your backend folder:

MODEL_PATH=final_unet_oilspill.h5
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=oilspill_detection

5️⃣ Run Backend
cd backend
uvicorn main:app --reload

6️⃣ Run Frontend
cd frontend
streamlit run app.py

☁️ Deployment (Render)

1. Both frontend and backend can be deployed on Render Cloud:
2. Deploy backend (FastAPI) as Web Service → https://oilspill-backend.onrender.com
3. Deploy frontend (Streamlit) as Web App → https://oilspill-frontend.onrender.com
4. Update API URLs in app.py to match deployed backend endpoint.

📊 Results & Visualizations

Model Performance:

IoU: ~0.89
Dice Coefficient: ~0.91
Precision: ~0.93
Recall: ~0.88

🌟 Key Highlights

1. End-to-end AI system integrating ML, API, and UI.
2. Real-time oil spill detection from satellite images.
3. MongoDB integration for report logging.
4. Clean, modular, and deployable architecture.
5. Cloud-hosted solution (Render) for accessibility.

🧩 Key Learnings

1. End-to-end understanding of ML pipelines (data → model → deployment).
2. Experience with semantic segmentation and satellite data.
3. Integration of FastAPI, Streamlit, and MongoDB.
4. Deployment on cloud platforms using .env configuration.
5. Enhanced skills in debugging, model optimization, and API design.

👩‍💻 Contributors

[Susritha Gudimetla] – Machine Learning & Full Stack Developer

Mentor / Guide: [, Infosys]