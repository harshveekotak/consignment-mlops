# Consignment Pricing Prediction Project

## Project Goal
The goal of this project is to predict the **line item value** for consignments based on historical consignment data. The project implements a full reproducible machine learning pipeline using **DVC** for data versioning and **MLflow** for experiment tracking.

---

## Dataset Used
- **Dataset:** `Consignment_pricing_raw.csv`  
- **Location:** `data/raw/`  
- The dataset contains shipment and consignment details such as shipment mode, vendor, product details, prices, weights, and processing times.  
- Preprocessing includes missing value handling, normalization, feature engineering, outlier treatment, and encoding categorical variables.

---

## Pipeline Structure
The pipeline consists of the following stages:

1. **Preprocessing (`preprocess`)**  
   - Script: `src/data_preprocessing.py`  
   - Input: `data/raw/Consignment_pricing_raw.csv`  
   - Output: `data/processed/processed_data.csv`  
   - Tasks: Cleaning, normalization, feature engineering, and encoding.

2. **Model Training (`train_model`)**  
   - Script: `src/model_training.py`  
   - Input: `data/processed/processed_data.csv`  
   - Output: `models/model.pkl`  
   - Tasks: Train a regression model (Random Forest / Decision Tree / Lasso) and log it using MLflow.

3. **Evaluation (`evaluate`)**  
   - Script: `src/evaluate.py`  
   - Input: `models/model.pkl`, `data/processed/processed_data.csv`  
   - Output: `metrics.json`  
   - Tasks: Evaluate the trained model and store metrics (R² score, RMSE).

> **Pipeline DAG:**  
> You can visualize the pipeline DAG by running:  
> ```bash
> dvc dag
> ```
> This shows the dependencies between stages:  
> `preprocess → train_model → evaluate`

---

## How to Run the Project
1. **Reproduce the pipeline:**  
```bash
dvc repro
