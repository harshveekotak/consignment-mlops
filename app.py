from flask import Flask, request, jsonify
import pandas as pd
import joblib
import os

app = Flask(__name__)

# Load model and preprocessor paths
MODEL_PATH = "models/model.pkl"
PREPROCESSOR_PATH = "src/preprocessor.pkl"

# Load model and preprocessing pipeline (if available)
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH) if os.path.exists(PREPROCESSOR_PATH) else None

@app.route('/')
def home():
    return jsonify({"message": "✅ Flask API for Consignment Price Prediction is running!"})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Convert JSON to DataFrame
        input_df = pd.DataFrame([data])

        # Apply preprocessing (if any)
        if preprocessor:
            input_processed = preprocessor.transform(input_df)
        else:
            input_processed = input_df

        # Make prediction
        prediction = model.predict(input_processed)[0]
        return jsonify({
            "prediction": float(prediction),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/retrain', methods=['POST'])
def retrain():
    """
    Optional route to trigger retraining from the Flask API.
    This could internally call your retraining script.
    """
    try:
        os.system("dvc repro retrain_model")
        return jsonify({"message": "Retraining pipeline executed successfully!"})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
