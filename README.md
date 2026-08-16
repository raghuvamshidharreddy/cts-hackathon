# CarPrice AI — Used Car Price Prediction Flask Application

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-3.0%2B-green)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

A production-grade Flask web application designed for predicting used-car resale market prices. The application features a modern dark glassmorphic user interface and utilizes a pre-trained **RandomForestRegressor** model (`car_price_model.pkl`) with a feature engineering pipeline reproducing the transformations from `carpriceprediction.ipynb`.

---

## 🌟 Key Features

- **ML Price Prediction**: Predicts estimated resale value using a 100-tree Random Forest model (R² ≈ 0.91 on test split).
- **Automated Feature Engineering**: Transforms raw car specs into the 19 exact features required by the model.
- **Dual Currency Output**: Displays prediction in Indian Rupees (`₹8,75,000`) and Lakhs (`₹8.75 Lakhs`).
- **Model Explainability**: Highlights key driving factors and global feature importances (`model.feature_importances_`).
- **Modern UI/UX**: Dark glassmorphic theme with responsive layout, inline validation, and loading indicators.
- **Robust Backend Validation**: Server-side checks for numerical ranges, zero/negative bounds, and category validation.
- **JSON API Endpoints**: Exposes `/api/predict` and `/api/model-info` for integration with external services.
- **Automated Test Suite**: Comprehensive unit and integration test suite covering preprocessing math, model contract, and Flask routes.

---

## ⚙️ Machine Learning Pipeline

```text
Raw User Specifications
(Brand, Year, KM, Engine, Power, Mileage, Fuel, Transmission, Seller, Owner)
                          │
                          ▼
            Feature Engineering Pipeline
  ┌──────────────────────────────────────────────┐
  │ 1. Engine Log (np.log)                       │
  │ 2. Max Power Log (np.log1p)                  │
  │ 3. One-Hot Categoricals (Fuel, Seller, Owner)│
  │ 4. Luxury Brand Detection & Interaction      │
  │ 5. Performance Index (power_log * engine_log)│
  │ 6. Petrol Wear (km * fuel_Petrol)            │
  │ 7. Age Wear ((2025 - year) * km)             │
  │ 8. Tech Score (power_log - engine_log)       │
  └──────────────────────────────────────────────┘
                          │
                          ▼
             Exact 19-Feature Input Vector
                          │
                          ▼
              RandomForestRegressor Model
                  (car_price_model.pkl)
                          │
                          ▼
             Estimated Price (INR & Lakhs)
```

---

## 📁 Project Architecture

```text
cts hackathon/
│
├── app.py                      # Flask main entrypoint, routes & error handlers
├── config.py                   # Configuration, feature lists, reference year
├── requirements.txt            # Minimal Python dependencies
├── README.md                   # Complete documentation
├── .gitignore                  # Git ignore file
│
├── model/
│   └── car_price_model.pkl     # Pre-trained 100-estimator Random Forest model
│
├── data/
│   └── Car_Price_Prediction_Dataset.csv # Raw dataset reference
│
├── services/
│   ├── __init__.py
│   ├── preprocessing.py        # Feature transformation logic (19 features)
│   ├── validation.py           # Server input validation rules
│   └── prediction.py           # Model loading, prediction & explainability
│
├── templates/
│   ├── base.html               # Shared HTML layout shell
│   ├── index.html              # 3-Section Prediction Form UI
│   └── result.html             # Price Result Card & Feature Breakdown
│
├── static/
│   ├── css/
│   │   └── style.css           # Vanilla CSS Dark Glassmorphic Design System
│   └── js/
│       └── app.js              # Client-side dynamic interaction & spinner
│
└── tests/
    ├── test_preprocessing.py   # Feature engineering unit tests
    ├── test_prediction.py      # Model contract & prediction tests
    └── test_routes.py          # Flask HTTP routes & API integration tests
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.10 or higher installed.

### 2. Setup Environment
Clone or navigate to the project directory and create a virtual environment:

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Application
```bash
python app.py
```

Open your browser and navigate to:
```text
http://127.0.0.1:5000
```

---

## 🧪 Running Automated Tests

Run the full test suite using Python's `unittest` framework:

```bash
python -m unittest discover -s tests
```

---

## 🌐 API Reference

### Health Check Endpoint
`GET /health`
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Metadata Endpoint
`GET /api/model-info`
```json
{
  "model_type": "RandomForestRegressor",
  "n_estimators": 100,
  "n_features": 19,
  "is_loaded": true
}
```

### Price Prediction Endpoint
`POST /api/predict` (Headers: `Content-Type: application.json`)

**Request Payload:**
```json
{
  "year": 2019,
  "km_driven": 45000,
  "transmission": "Automatic",
  "mileage": 18.5,
  "seats": 5,
  "engine": 1998,
  "max_power": 150,
  "fuel": "Petrol",
  "seller_type": "Individual",
  "owner": "First Owner",
  "brand": "BMW"
}
```

**Response Payload:**
```json
{
  "success": true,
  "predicted_price": 875000.0,
  "predicted_price_lakhs": 8.75,
  "formatted_inr": "₹8,75,000",
  "formatted_lakhs": "₹8.75 Lakhs",
  "top_features": [
    {
      "feature": "max_power_log",
      "label": "Max Power (Log)",
      "importance": 0.4215,
      "percentage": 42.2
    }
  ]
}
```

---

## 🔒 ML & Security Integrity Rules
1. **Preserved Reference Year**: Features such as `age_wear = (2025 - year) * km_driven` strictly use `2025` to maintain feature distribution alignment with the trained model.
2. **Feature Order Contract**: Startup asserts that the model input DataFrame matches the exact order of `19` features expected by `car_price_model.pkl`.
3. **No Tracebacks Exposed**: Custom Flask error handlers log internal exceptions on the server while returning clean, user-friendly responses.
