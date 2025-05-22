# app.py

import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model
with open('logistic_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title("Loan Default Prediction App")

# User input form
Gender = st.selectbox("Gender", ['Male', 'Female'])
Married = st.selectbox("Married", ['Yes', 'No'])
Dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
Self_Employed = st.selectbox("Self Employed", ['Yes', 'No'])
ApplicantIncome = st.number_input("Applicant Income", min_value=0)
CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0)
LoanAmount = st.number_input("Loan Amount", min_value=0)
Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0)
Credit_History = st.selectbox("Credit History", [0.0, 1.0])
Property_Area = st.selectbox("Property Area", ['Urban', 'Semiurban', 'Rural'])

# Preprocess inputs
if st.button("Predict"):
    # Convert inputs to DataFrame
    input_data = pd.DataFrame({
        'Gender': [Gender],
        'Married': [Married],
        'Dependents': [Dependents],
        'Education': [Education],
        'Self_Employed': [Self_Employed],
        'ApplicantIncome': [ApplicantIncome],
        'CoapplicantIncome': [CoapplicantIncome],
        'LoanAmount': [LoanAmount],
        'Loan_Amount_Term': [Loan_Amount_Term],
        'Credit_History': [Credit_History],
        'Property_Area': [Property_Area]
    })

    # Process Dependents
    input_data['Dependents'] = input_data['Dependents'].replace('3+', 3).astype(int)

    # One-hot encoding
    categorical_cols = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    input_data = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)

    # Match training features
    expected_cols = model.feature_names_in_  # automatically comes from sklearn>=1.0
    for col in expected_cols:
        if col not in input_data:
            input_data[col] = 0
    input_data = input_data[expected_cols]

    # Predict
    prediction = model.predict(input_data)[0]
    result = "Approved ✅" if prediction == 1 else "Rejected ❌"
    st.subheader(f"Loan Status: {result}")
