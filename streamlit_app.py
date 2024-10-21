


import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
import joblib  # Assuming you saved your model

st.title("Heart Failure Prediction")

# Load your trained model and scaler
rfc_mas_final = joblib.load('model.pkl')  # Replace with your model's filename
mas = joblib.load('scaler.pkl')  # Replace with your scaler's filename

# Function to make predictions
def predict_heart_failure(features):
    features_scaled = mas.transform(np.array(features).reshape(1, -1))
    prediction = rfc_mas_final.predict(features_scaled)
    return prediction[0]

# Streamlit app layout
st.title("Heart Failure Prediction")
st.write("Enter the details below to predict if the patient is alive (0) or dead (1):")

# Input fields
age = st.number_input("Age", min_value=0)
anaemia = st.selectbox("Anaemia (1: Yes, 0: No)", [0, 1])
creatinine_phosphokinase = st.number_input("Creatinine Phosphokinase")
diabetes = st.selectbox("Diabetes (1: Yes, 0: No)", [0, 1])
ejection_fraction = st.number_input("Ejection Fraction", min_value=0, max_value=100)
high_blood_pressure = st.selectbox("High Blood Pressure (1: Yes, 0: No)", [0, 1])
platelets = st.number_input("Platelets")
serum_creatinine = st.number_input("Serum Creatinine")
serum_sodium = st.number_input("Serum Sodium")
sex = st.selectbox("Sex (1: Male, 0: Female)", [0, 1])
smoking = st.selectbox("Smoking (1: Yes, 0: No)", [0, 1])
time = st.number_input("Time")

# Button for prediction
if st.button("Predict"):
    features = [age, anaemia, creatinine_phosphokinase, diabetes,
                ejection_fraction, high_blood_pressure, platelets,
                serum_creatinine, serum_sodium, sex, smoking, time]
    
    prediction = predict_heart_failure(features)
    result = "Dead" if prediction == 1 else "Alive"
    st.write(f"The prediction is: **{result}**")

