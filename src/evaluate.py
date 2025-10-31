import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import mlflow
import numpy as np
import json  # add this import

# Load data and model
df = pd.read_csv('data/processed/processed_data.csv')
model = joblib.load('models/model.pkl')

X = df.drop('line_item_value', axis=1)  # your target column
y = df['line_item_value']
y_pred = model.predict(X)

# Calculate metrics
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

# Log metrics to MLflow
mlflow.log_metric("r2_score", r2)
mlflow.log_metric("rmse", rmse)

# Save metrics to metrics.json for DVC
metrics = {"r2_score": r2, "rmse": rmse}
with open("metrics.json", "w") as f:
    json.dump(metrics, f)

print(f"R²: {r2:.3f}, RMSE: {rmse:.3f}")
