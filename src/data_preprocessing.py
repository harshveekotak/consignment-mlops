import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def preprocess():
    # Load raw data
    raw_file = r'D:\Jupyter Notebook\Sem 3\MLOps\consignment\data\raw\Consignment_pricing_raw.csv'
    data = pd.read_csv(raw_file)
    print("Raw data loaded. Shape:", data.shape)

    # ---------------------------
    # Standardize column names
    # ---------------------------
    data.columns = (data.columns
                    .str.strip().str.lower()
                    .str.replace(" ", "_")
                    .str.replace("/", "_")
                    .str.replace("(", "").str.replace(")", ""))

    # ---------------------------
    # Drop irrelevant identifiers & text-heavy columns
    # ---------------------------
    drop_cols = ["id","project_code","pq_#","item_description","molecule_test_type","brand","vendor"]
    data = data.drop(columns=[c for c in drop_cols if c in data.columns], errors="ignore")
    print("Remaining columns after cleanup:", data.shape[1])

    # ---------------------------
    # Shipment Mode
    # ---------------------------
    if "shipment_mode" in data.columns:
        print("Before fill:\n", data["shipment_mode"].value_counts(dropna=False))
        data["shipment_mode"] = data["shipment_mode"].fillna("Air")
        print("\nAfter fill:\n", data["shipment_mode"].value_counts(dropna=False))

    # ---------------------------
    # Vendor Inco Term
    # ---------------------------
    if "vendor_inco_term" in data.columns:
        print("\nBefore normalization:\n", data["vendor_inco_term"].value_counts())
        data["vendor_inco_term"] = data["vendor_inco_term"].replace(["DDU","DAP","CIF"], "others")
        print("\nAfter normalization:\n", data["vendor_inco_term"].value_counts())

    # ---------------------------
    # First Line Designation
    # ---------------------------
    if "first_line_designation" in data.columns:
        print("\nUnique values before:", data["first_line_designation"].unique()[:10])
        data["first_line_designation"] = data["first_line_designation"].astype(str)
        print("Unique values after:", data["first_line_designation"].unique()[:10])

    # ---------------------------
    # Freight Cost
    # ---------------------------
    def transform_freight(x):
        if isinstance(x,str):
            if "Freight Included" in x or "Invoiced" in x:
                return 0
            if "See" in x:
                return np.nan
        return x

    if "freight_cost_usd" in data.columns:
        data["freight_cost_usd"] = data["freight_cost_usd"].apply(transform_freight)
        data["freight_cost_usd"] = pd.to_numeric(data["freight_cost_usd"], errors="coerce")
        data["freight_cost_usd"] = data["freight_cost_usd"].fillna(data["freight_cost_usd"].median())

    # ---------------------------
    # Line Item Insurance
    # ---------------------------
    if "line_item_insurance_usd" in data.columns:
        data["line_item_insurance_usd"] = pd.to_numeric(data["line_item_insurance_usd"], errors="coerce")
        data["line_item_insurance_usd"] = data["line_item_insurance_usd"].fillna(data["line_item_insurance_usd"].median())

    print("Numeric missing values are updated")

    # ---------------------------
    # Convert dates
    # ---------------------------
    date_cols = ["pq_first_sent_to_client_date","scheduled_delivery_date",
                 "delivered_to_client_date","delivery_recorded_date"]
    
    # Adjust this format to match your data, e.g., "%Y-%m-%d" or "%d/%m/%Y"
    date_format = "%Y-%m-%d"

    for col in date_cols:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], format=date_format, errors="coerce")

    # ---------------------------
    # Compute lead time
    # ---------------------------
    if {"delivery_recorded_date","pq_first_sent_to_client_date"}.issubset(data.columns):
        data["days_to_process"] = (data["delivery_recorded_date"] - data["pq_first_sent_to_client_date"]).dt.days
        data["days_to_process"] = data["days_to_process"].fillna(data["days_to_process"].median())

    # Drop raw dates
    data = data.drop(columns=[c for c in date_cols if c in data.columns], errors="ignore")
    print("Datetime converted, days_to_process added")

    # ---------------------------
    # Feature Engineering
    # ---------------------------
    if {"line_item_value","line_item_quantity"}.issubset(data.columns):
        data["cost_per_pack"] = data["line_item_value"] / (data["line_item_quantity"]+1)

    if {"freight_cost_usd","line_item_value"}.issubset(data.columns):
        data["freight_ratio"] = data["freight_cost_usd"] / (data["line_item_value"]+1)

    if {"line_item_value","weight_kilograms"}.issubset(data.columns):
        data['weight_kilograms'] = pd.to_numeric(data['weight_kilograms'], errors="coerce")
        data['weight_kilograms'] = data['weight_kilograms'].fillna(data['weight_kilograms'].median())
        data["cost_per_kg"] = data["line_item_value"] / (data['weight_kilograms'] + 1)

    print("Feature engineering done")
    print(data[["cost_per_pack","freight_ratio","cost_per_kg"]].head())

    # ---------------------------
    # Clamp numeric features between 1st and 99th percentile
    # ---------------------------
    num_cols = data.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        low, high = data[col].quantile([0.01, 0.99])
        data[col] = data[col].clip(lower=low, upper=high)
    print("Outlier treatment done")

    # ---------------------------
    # Label Encoding for categorical features
    # ---------------------------
    cat_cols = data.select_dtypes(include="object").columns
    le = LabelEncoder()
    for col in cat_cols:
        data[col] = le.fit_transform(data[col].astype(str))
    print("Encoding done")
    print("Encoded categorical columns:", list(cat_cols))
    print("Final dataset shape:", data.shape)

    # ---------------------------
    # Save processed data
    # ---------------------------
    data.to_csv("data/processed/processed_data.csv", index=False)
print("Preprocessing completed. Saved to data/processed/processed_data.csv")

if __name__ == "__main__":
    preprocess()
