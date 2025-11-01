import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import os

# ==============================
# 🎨 APP CONFIGURATION
# ==============================
st.set_page_config(page_title="📦 Consignment Price Predictor", page_icon="🚚", layout="wide")

# ==============================
# 🖼️ HEADER SECTION
# ==============================
col1, col2 = st.columns([1, 6])
with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/679/679922.png", width=90)
with col2:
    st.markdown("""
        # 🚛 Consignment Pricing Prediction System
        This app predicts **line item value (consignment cost)** based on shipment and vendor details.  
        It is part of your **MLOps pipeline with DVC, MLflow, and Docker integration**.
    """)

st.markdown("---")

# ==============================
# 📦 LOAD MODEL & DATA
# ==============================
model_path = "models/model.pkl"
data_path = "data/processed/processed_data.csv"

if not os.path.exists(model_path):
    st.error("❌ Model file not found! Please ensure `models/model.pkl` exists.")
    st.stop()

model = joblib.load(model_path)
sample_df = pd.read_csv(data_path)
feature_columns = [col for col in sample_df.columns if col != "line_item_value"]

# ==============================
# 🧭 FEATURE NAMES AND DESCRIPTIONS
# ==============================
feature_info = {
    "po___so_#": "Purchase or Sales Order Number",
    "asn_dn_#": "Advanced Shipping Notice / Delivery Note Number",
    "country": "Destination or source country for consignment",
    "managed_by": "Region or department managing the consignment",
    "fulfill_via": "Order fulfillment method (e.g., Warehouse, Direct Drop)",
    "vendor_inco_term": "Vendor International Commercial Terms (Incoterms)",
    "shipment_mode": "Mode of shipment (Air, Ocean, Truck, etc.)",
    "po_sent_to_vendor_date": "Date when the purchase order was sent to the vendor",
    "product_group": "Main category of the product group",
    "sub_classification": "Sub-category of the product",
    "dosage": "Dosage amount or specification",
    "dosage_form": "Form of dosage (Tablet, Liquid, etc.)",
    "unit_of_measure_per_pack": "Number of units per pack",
    "line_item_quantity": "Quantity of items in the purchase line",
    "pack_price": "Price per pack",
    "unit_price": "Price per individual unit",
    "manufacturing_site": "Factory or site where product is manufactured",
    "first_line_designation": "Indicates if the product is first-line treatment",
    "weight_kilograms": "Total shipment weight in kilograms",
    "freight_cost_usd": "Freight cost in USD",
    "line_item_insurance_usd": "Insurance cost in USD",
    "days_to_process": "Number of days required to process order",
    "cost_per_pack": "Calculated cost per pack",
    "freight_ratio": "Ratio of freight cost to total value",
    "cost_per_kg": "Cost per kilogram of shipment"
}

# Cleaner display names (UI → backend mapping)
display_to_feature = {v: k for k, v in feature_info.items()}

# ==============================
# ℹ️ FEATURE INFO SECTION
# ==============================
with st.expander("ℹ️ Feature Information", expanded=False):
    st.write("These are the input features used for prediction:")
    st.dataframe(pd.DataFrame.from_dict(feature_info, orient='index', columns=["Description"]))

st.markdown("---")

# ==============================
# 🧠 INPUT SECTION
# ==============================
st.subheader("🧾 Enter Consignment Details")

cols = st.columns(3)
input_data = {}

for i, (feature, desc) in enumerate(feature_info.items()):
    col = cols[i % 3]
    dtype = sample_df[feature].dtype

    if dtype in ["float64", "int64"]:
        input_data[feature] = col.number_input(
            label=f"📊 {desc}",
            value=float(sample_df[feature].mean()) if not sample_df[feature].isna().all() else 0.0,
            format="%.3f",
            help=desc
        )
    else:
        unique_vals = sample_df[feature].dropna().unique().tolist()
        if len(unique_vals) <= 20:
            input_data[feature] = col.selectbox(f"🧩 {desc}", options=unique_vals, help=desc)
        else:
            input_data[feature] = col.text_input(f"🧩 {desc}", value=str(sample_df[feature].iloc[0]), help=desc)

# ==============================
# 🚀 PREDICTION SECTION
# ==============================
st.markdown("---")
if st.button("🔮 Predict Line Item Value"):
    try:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        st.success(f"💰 Predicted Consignment Value: **${prediction:,.2f} USD**")
        st.balloons()
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")

# ==============================
# 📊 FOOTER SECTION
# ==============================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:gray;'>
    🚀 Built with ❤️ using Streamlit | MLflow | DVC | Docker  
    <br>Project: <b>Consignment-MLops</b>
</div>
""", unsafe_allow_html=True)
