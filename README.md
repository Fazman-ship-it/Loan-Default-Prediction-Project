# Loan Default Prediction 🏦📊

This project focuses on building a machine learning model to predict loan default risks based on historical loan applicant data. It includes a full pipeline from data preprocessing and exploratory data analysis (EDA) to model training, evaluation, and deployment via a Streamlit web application.

---

## 📁 Table of Contents

- [📌 Project Description](#project-description)
- [🧾 Dataset](#dataset)
- [🔍 Exploratory Data Analysis](#exploratory-data-analysis)
- [🧠 Modeling](#modeling)
- [🚀 Deployment](#deployment)
- [🗂️ Project Structure](#project-structure)
- [⚙️ How to Run the Project](#how-to-run-the-project)
- [🧪 Requirements](#requirements)

---

## 📌 Project Description

The goal of this project is to classify whether a loan applicant is likely to repay a loan (Loan_Status = Y) or default (Loan_Status = N). The model uses various demographic and financial features such as income, education level, loan amount, credit history, and more.

---

## 🧾 Dataset

The dataset includes:
- *Categorical features*: Gender, Marital Status, Dependents, Education, Self-Employed, Property Area
- *Numerical features*: Applicant Income, Coapplicant Income, Loan Amount, Loan Term, Credit History
- *Target variable*: Loan_Status (Y/N)

File: data/train.csv

---

## 🔍 Exploratory Data Analysis

EDA helped us understand:
- Most applicants with a credit history of 1 were approved.
- Graduates had a slightly higher approval rate.
- Distribution of loan amounts is right-skewed.

![EDA Visuals](visuals/loan_visuals.png)

---

## 🧠 Modeling

### Models Trained:
- *Logistic Regression* ✅ (Deployed model)
- *Decision Tree*
- *Random Forest*

### Best Model: *Logistic Regression*
- *Accuracy*: ~81%
- *Key Features*: Credit_History, ApplicantIncome, LoanAmount, Education

Notebook file: notebooks/Loan Default Prediction.ipynb  
Model file : models/logistic_regression_model.pkl

---

## 🚀 Deployment

A *Streamlit* web application allows users to input loan applicant details and get a prediction instantly.

App script: app/app.py

To run the app:
```bash
streamlit run app/app.py
