# Churn Prediction API

This is a Flask API that serves a machine learning model for predicting customer churn.

## 🚀 Deployment (Render)

1. Upload this folder to a GitHub repo.
2. Go to [https://render.com](https://render.com)
3. Click "New Web Service", connect your GitHub, and choose the repo.
4. Make sure the following files are in the `backend/` folder:
   - `app.py`
   - `customer_churn_model.pkl`
   - `encoder.pkl`
   - `requirements.txt`
   - `render.yaml`

5. Set the **start command**: `python app.py`
6. Set the **port**: `5000` (Render auto-detects this)

## 📦 API Endpoint

**POST** `/predict`

**Request JSON:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "tenure": 10,
  ...
}
