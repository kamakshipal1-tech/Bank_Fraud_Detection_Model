import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Bank Fraud Detection",
    page_icon="🏦",
    layout="centered"
)

# -------------------------------
# Load Model
# -------------------------------
artifact = joblib.load("best_fraud_model_tuned.pkl")

model = artifact["model"]
threshold = artifact["threshold"]
p99 = artifact["amount_threshold"]
label_encoder = artifact["label_encoder"]

# -------------------------------
# Title
# -------------------------------
st.title("🏦 Bank Fraud Detection System")
st.markdown(
    "Predict whether a banking transaction is **Fraudulent** or **Legitimate** using a trained Machine Learning model."
)

st.divider()

# -------------------------------
# User Inputs
# -------------------------------

transaction_type = st.selectbox(
    "Transaction Type",
    list(label_encoder.classes_)
)

step = st.number_input(
    "Transaction Step (Hour in simulation)",
    min_value=1,
    value=100
)

amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

oldbalanceOrg = st.number_input(
    "Sender Old Balance",
    min_value=0.0,
    value=5000.0
)

newbalanceOrig = st.number_input(
    "Sender New Balance",
    min_value=0.0,
    value=4000.0
)

oldbalanceDest = st.number_input(
    "Receiver Old Balance",
    min_value=0.0,
    value=1000.0
)

newbalanceDest = st.number_input(
    "Receiver New Balance",
    min_value=0.0,
    value=2000.0
)

# -------------------------------
# Prediction
# -------------------------------

if st.button("Predict Fraud", use_container_width=True):

    log_amount = np.log1p(amount)

    is_high_amount = int(amount > p99)

    hour = step % 24

    is_night = int(hour in [0,1,2,3,4,5,22,23])

    balance_diff_orig = oldbalanceOrg - newbalanceOrig

    balance_diff_dest = newbalanceDest - oldbalanceDest

    type_enc = label_encoder.transform([transaction_type])[0]

    input_df = pd.DataFrame([{
        "step": step,
        "amount": amount,
        "log_amount": log_amount,
        "is_high_amount": is_high_amount,
        "hour": hour,
        "is_night": is_night,
        "balance_diff_orig": balance_diff_orig,
        "balance_diff_dest": balance_diff_dest,
        "type_enc": type_enc
    }])

    probability = model.predict_proba(input_df)[0][1]

    prediction = int(probability >= threshold)

    confidence = probability if prediction else (1-probability)

    st.divider()

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected")
    else:
        st.success("✅ Legitimate Transaction")

    st.metric(
        "Fraud Probability",
        f"{probability*100:.2f}%"
    )

    st.metric(
        "Model Confidence",
        f"{confidence*100:.2f}%"
    )

    if probability >= 0.80:
        risk = "🔴 High Risk"

    elif probability >= 0.50:
        risk = "🟠 Medium Risk"

    else:
        risk = "🟢 Low Risk"

    st.info(f"Risk Level: **{risk}**")