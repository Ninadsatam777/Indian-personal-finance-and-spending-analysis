# ================================
# STREAMLIT APP FOR INCOME PREDICTION
# ================================

import streamlit as st
import pandas as pd
import pickle
import os

# -------------------------------
# Load Model (SAFE VERSION)
# -------------------------------

model_path = "rf_model.pkl"

if not os.path.exists(model_path):
    st.error("❌ Model file not found. Please place rf_model.pkl in this folder.")
    st.stop()

model = pickle.load(open(model_path, "rb"))

# -------------------------------
# PAGE SETTINGS
# -------------------------------
st.set_page_config(page_title="Income Prediction App", layout="centered")

st.title("💰 Disposable Income Prediction App")
st.write("Enter details below to predict Income")

# -------------------------------
# USER INPUT SECTION
# -------------------------------

Age = st.number_input("Age", min_value=18, max_value=70, value=30)

Gender = st.selectbox("Gender", ["Male", "Female"])

City_Tier = st.selectbox("City Tier", ["Tier 1", "Tier 2", "Tier 3"])

Occupation = st.selectbox(
    "Occupation",
    ["Student", "Salaried", "Self_Employed", "Unemployed"]
)

Marital_Status = st.selectbox(
    "Marital Status",
    ["Single", "Married"]
)

Dependents = st.number_input("Dependents", min_value=0, max_value=6, value=0)

Education = st.selectbox(
    "Education",
    ["High School", "Graduate", "Post Graduate"]
)

Rent = st.number_input("Rent", min_value=0, value=5000)
Groceries = st.number_input("Groceries", min_value=0, value=3000)
Transport = st.number_input("Transport", min_value=0, value=1000)
Eating_Out = st.number_input("Eating Out", min_value=0, value=1000)
Entertainment = st.number_input("Entertainment", min_value=0, value=1000)

# -------------------------------
# CREATE INPUT DATAFRAME
# -------------------------------

input_dict = {
    "Age": Age,
    "Dependents": Dependents,
    "Rent": Rent,
    "Groceries": Groceries,
    "Transport": Transport,
    "Eating_Out": Eating_Out,
    "Entertainment": Entertainment,
    "Gender": Gender,
    "City_Tier": City_Tier,
    "Occupation": Occupation,
    "Marital_Status": Marital_Status,
    "Education": Education
}

input_df = pd.DataFrame([input_dict])

# Apply same encoding as training
input_df = pd.get_dummies(input_df, drop_first=True)

# -------------------------------
# ALIGN INPUT WITH TRAINING COLUMNS
# -------------------------------

model_columns = model.feature_names_in_

for col in model_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_columns]

# -------------------------------
# PREDICTION BUTTON
# -------------------------------

if st.button("Predict Income"):
    prediction = model.predict(input_df)[0]

    st.success(f"💵 Predicted Income: ₹ {round(prediction,2)}")

    st.info("📊 This prediction is based on Random Forest model trained in your notebook.")