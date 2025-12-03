from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model when app starts
print("Loading model...")
model = joblib.load('churn_model.pkl')
feature_columns = joblib.load('feature_columns.pkl')
print("Model loaded successfully!")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok",
        "model": "churn_prediction_v1",
        "accuracy": "74.2%"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Convert to DataFrame with correct columns
        input_df = pd.DataFrame([data], columns=feature_columns)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        # Return result
        return jsonify({
            "churn_prediction": int(prediction),
            "churn_probability": round(float(probability), 3),
            "risk_level": "high" if probability > 0.7 else "medium" if probability > 0.4 else "low",
            "recommendation": "Contact customer immediately" if probability > 0.7 else "Monitor closely" if probability > 0.4 else "Low risk"
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=8080)
    
