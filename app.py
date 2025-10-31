from flask import Flask, jsonify, request
import pandas as pd
import os

app = Flask(__name__)

@app.route('/')
def home():
    return jsonify({"message": "Flask API running successfully!"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    value = data.get("input", 0)
    result = value * 2  # dummy logic, replace with your model inference
    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
