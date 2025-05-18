import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import xgboost as xgb

st.set_page_config(page_title="Churn Prediction App", layout="wide")

# Load or train model
@st.cache_resource
def load_model():
    try:
        return joblib.load("model/churn_model.pkl")
    except:
        return train_and_save_model()

def train_and_save_model():
    df = pd.read_csv("data/customer_churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(subset=['TotalCharges'], inplace=True)
    df.drop("customerID", axis=1, inplace=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

    cat_cols = X.select_dtypes(include='object').columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
    ])

    pipeline.fit(X, y)
    os.makedirs("model", exist_ok=True)
    joblib.dump(pipeline, "model/churn_model.pkl")
    return pipeline

model = load_model()

# Sample input builder
def build_input():
    st.sidebar.header("🧾 Customer Details")
    input_dict = {
        "gender": st.sidebar.selectbox("Gender", ["Male", "Female"]),
        "SeniorCitizen": st.sidebar.selectbox("Senior Citizen", [0, 1]),
        "Partner": st.sidebar.selectbox("Partner", ["Yes", "No"]),
        "Dependents": st.sidebar.selectbox("Dependents", ["Yes", "No"]),
        "tenure": st.sidebar.slider("Tenure (months)", 0, 72, 12),
        "PhoneService": st.sidebar.selectbox("Phone Service", ["Yes", "No"]),
        "MultipleLines": st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"]),
        "InternetService": st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"]),
        "OnlineSecurity": st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"]),
        "OnlineBackup": st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"]),
        "DeviceProtection": st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"]),
        "TechSupport": st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"]),
        "StreamingTV": st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"]),
        "StreamingMovies": st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"]),
        "Contract": st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": st.sidebar.selectbox("Paperless Billing", ["Yes", "No"]),
        "PaymentMethod": st.sidebar.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
        ]),
        "MonthlyCharges": st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0, step=0.1),
        "TotalCharges": st.sidebar.number_input("Total Charges", min_value=0.0, value=3000.0, step=1.0)
    }
    return pd.DataFrame([input_dict])

st.title("💼 Customer Churn Prediction with Explainability")

input_df = build_input()

# Predict and display
if st.button("🔍 Predict Churn"):
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0][1]

    st.subheader("🔎 Prediction Result:")
    st.markdown(f"**Churn Probability:** `{proba:.2%}`")

    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is not likely to churn.")

    # SHAP Explanation
    st.subheader("📊 Feature Importance (SHAP)")

    # Extract XGBoost model from pipeline
    encoder = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]
    
    # Create background and transformed input
    df_raw = pd.read_csv("data/customer_churn.csv")
    df_raw['TotalCharges'] = pd.to_numeric(df_raw['TotalCharges'], errors='coerce')
    df_raw.dropna(subset=['TotalCharges'], inplace=True)
    df_raw.drop("customerID", axis=1, inplace=True)
    X_raw = df_raw.drop("Churn", axis=1)
    X_transformed = encoder.fit_transform(X_raw)
    
    explainer = shap.Explainer(classifier)
    input_encoded = encoder.transform(input_df)
    shap_values = explainer(input_encoded)

    #st.set_option('deprecation.showPyplotGlobalUse', False)
    fig, ax = plt.subplots()
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(fig)

# Retrain option
with st.expander("🛠️ Retrain Model"):
    if st.button("🔁 Retrain on Full Dataset"):
        train_and_save_model()
        st.success("✅ Model retrained and saved!")
