


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


st.title("Heart Failure Prediction")


## Loading model and scaler

df=pd.read_csv("heart_failure_clinical_records.csv")

# Correlation matrix
correlation_matrix = df.corr()

# Plotting the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')

# Display the heatmap in Streamlit
st.pyplot(plt)

## Loading model and scaler

## Splitting data into x and y
x=df.drop(columns=['DEATH_EVENT'])
y=df['DEATH_EVENT']

## splitting data into train test

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=2)

## applying maxabs scaler to the X_train, X_test

mas=MaxAbsScaler()
X_train_mas=mas.fit_transform(X_train)
X_test_mas=mas.transform(X_test)

## selecting the features of the dataset
features_final=df[['ejection_fraction', 'serum_creatinine']]
target=df['DEATH_EVENT']

## train test split
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(features_final, target, test_size=0.15,random_state=2)


## applying maxabs scaler to the X_train, X_test

#mas=MaxAbsScaler()
X_train_mas_final=mas.fit_transform(X_train_final)
X_test_mas_final=mas.transform(X_test_final)


rfc_mas_final=RandomForestClassifier(n_estimators=100)
rfc_mas_final.fit(X_train_mas_final, y_train_final)





# Function to make predictions
def predict_heart_failure(features):
    features_scaled = mas.transform(np.array(features).reshape(1, -1))
    prediction = rfc_mas_final.predict(features_scaled)
    return prediction[0]

# Streamlit app layout

st.write("Enter the details below to predict if the patient is alive (0) or dead (1):")

# Input fields


ejection_fraction = st.number_input("Ejection Fraction", min_value=0, max_value=100)

serum_creatinine = st.number_input("Serum Creatinine")


# Button for prediction
if st.button("Predict"):
    features = [ejection_fraction,serum_creatinine]
    
    prediction = predict_heart_failure(features)
    result = "Dead" if prediction == 1 else "Alive"
    st.write(f"The prediction is: **{result}**")




