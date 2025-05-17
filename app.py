import streamlit as st
import pickle
import numpy as np

# Load your trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction")
st.write("Enter customer details to predict churn")

# Example input features
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.slider("Tenure (months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Encoding categorical features
gender_encoded = 1 if gender == "Male" else 0
contract_encoded = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]

# Prediction
if st.button("Predict Churn"):
    features = np.array([[gender_encoded, tenure, monthly_charges, contract_encoded]])
    prediction = model.predict(features)
    result = "Churn" if prediction[0] == 1 else "No Churn"
    st.success(f"Prediction: {result}")
