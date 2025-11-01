import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import mlflow
import mlflow.sklearn
import yaml
import numpy as np
import os
import shutil

# Load training parameters
params = yaml.safe_load(open("params.yaml"))['train']

# Load processed data
df = pd.read_csv('data/processed/processed_data.csv')

# Define features and target
X = df.drop('line_item_value', axis=1)
y = df['line_item_value']

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params['test_size'], random_state=params['random_state']
)

# Load old model if exists
old_model_path = 'models/model.pkl'
old_r2, old_rmse = None, None
old_model = None

if os.path.exists(old_model_path):
    old_model = joblib.load(old_model_path)
    y_pred_old = old_model.predict(X_test)
    old_r2 = r2_score(y_test, y_pred_old)
    old_rmse = np.sqrt(mean_squared_error(y_test, y_pred_old))
    print(f"📊 Old model - R²: {old_r2:.4f}, RMSE: {old_rmse:.4f}")
else:
    print("⚠️ No previous model found. Proceeding with new model training.")

# Start MLflow run
mlflow.set_experiment("Consignment Model Retraining")

with mlflow.start_run():
    new_model = RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        random_state=42
    )
    new_model.fit(X_train, y_train)

    # Evaluate new model
    y_pred_new = new_model.predict(X_test)
    new_r2 = r2_score(y_test, y_pred_new)
    new_rmse = np.sqrt(mean_squared_error(y_test, y_pred_new))

    print(f"🆕 New model - R²: {new_r2:.4f}, RMSE: {new_rmse:.4f}")

    mlflow.log_metric("r2_score_new", new_r2)
    mlflow.log_metric("rmse_new", new_rmse)

    # Save retrained model separately first
    retrained_path = 'models/model_retrained.pkl'
    joblib.dump(new_model, retrained_path)
    print(f"💾 Retrained model saved as {retrained_path}")

    # Compare performance
    if (old_model is None) or (new_r2 > old_r2) or (new_rmse < old_rmse):
        shutil.copy(retrained_path, old_model_path)
        print("✅ New retrained model performs better. Replaced old model.")
        mlflow.sklearn.log_model(new_model, "model_retrained")
    else:
        print("❌ Retrained model did not improve. Keeping old model.")

print("✅ Retraining pipeline completed.")
