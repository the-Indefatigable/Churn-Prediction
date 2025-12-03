# Churn Prediction API

Predicts customer churn risk using machine learning (XGBoost, 74.2% accuracy).

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train model (first time only)
python train_model.py

# Run API
python app.py
```

API runs on `http://localhost:8080`

## API Endpoints

### Health Check
```bash
curl http://localhost:8080/health
```

### Predict Churn
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0
  }'
```

**Response:**
```json
{
  "churn_prediction": 1,
  "churn_probability": 0.752,
  "risk_level": "high",
  "recommendation": "Contact customer immediately"
}
```

## Model Performance
- **Accuracy:** 74.2%
- **Precision:** 51.0%
- **Recall:** 75.4%
- **F1-Score:** 60.8%

## Business Use Case
SaaS companies can integrate this API to:
- Identify at-risk customers before they churn
- Prioritize retention efforts
- Reduce customer acquisition costs by improving retention

## Tech Stack
- Python 3.x
- scikit-learn
- XGBoost
- Flask
- pandas

🌐 **Live Demo:** https://churn-prediction-production-8afe.up.railway.app

Try it:
```bash
curl https://churn-prediction-production-8afe.up.railway.app/health
