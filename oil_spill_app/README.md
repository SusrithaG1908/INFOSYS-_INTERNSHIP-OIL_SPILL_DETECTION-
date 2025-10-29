#backend
.env
----
MONGO_URI=mongodb://localhost:27017
MODEL_PATH=final_unet_oilspill.h5

commands
--------
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8000

#frontend
.env
----
BACKEND_URL=http://127.0.0.1:8000

commands
--------
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m streamlit run app.py


