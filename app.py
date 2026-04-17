import streamlit as st
import numpy as np
import joblib
import streamlit as st

st.title("🏦 Bank Churn Prediction")
st.write("App successfully deployed 🚀")
st.title("🏦 Bank Churn Prediction")

# Inputs
credit_score = st.number_input("Credit Score", 300, 900, 600)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.number_input("Products", 1, 4, 1)
has_card = st.selectbox("Has Card", [0, 1])
is_active = st.selectbox("Active Member", [0, 1])
salary = st.number_input("Salary", 0.0, 200000.0, 50000.0)

if st.button("Predict"):

    # Proper encoding
    gender_val = le.transform([gender])[0]

    # Input array
    input_data = np.array([[credit_score, geography, gender_val, age, tenure,
                            balance, num_products, has_card, is_active, salary]])

    # Apply transformations
    input_data = encoder.transform(input_data)
    input_data = scaler.transform(input_data)

    # Prediction
    pred = model.predict(input_data)
    result = (pred > 0.5).astype(int)

    if result[0][0] == 1:
        st.error("❌ Customer will churn")
    else:
        st.success("✅ Customer will stay")
