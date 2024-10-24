


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.preprocessing import MaxAbsScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Custom CSS for styling the title

col1, col2 = st.columns([3,1], vertical_alignment = "center")
st.markdown(
    """
    <style>
    .title {
        color: CornflowerBlue;
        font-size: 26px;
        text-align: left;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        text-shadow: 1px 1px24px #000000;
        border-bottom: 3px solid #add8e6;
        padding-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add your title using the custom class
with col1:
    st.markdown('<h1 class="title">Heart Failure Prediction Model</h1>', unsafe_allow_html=True)



# Display local image
with col2:
    col2 = st.image('rb_1516.png', use_column_width=True)


# Model metrics

col3, col4, col5 = st.columns(3)

col3.metric("Model Accuracy", "90%")
col4.metric("Model Precision", "90%")
col5.metric("Model Recall", "87%")


# Custom CSS for styling
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;  /* Light blue background */
    }
    h1 {
        color: darkblue;
        text-align: center;
    }
    .stMetric {
        font-size: 24px;  /* Custom font size for metrics */
    }
    .stImage {
        border-radius: 10px;  /* Rounded corners for the image */
    }
    </style>
    """,
    unsafe_allow_html=True
)




## Loading model and scaler

df=pd.read_csv("heart_failure_clinical_records.csv")

# Correlation matrix


# Assuming df is already loaded with your dataset
# Exclude the 'time' column from the correlation matrix
df_filtered = df.drop(columns=['time'])

# Create the correlation matrix without the 'time' column
correlation_matrix = df_filtered.corr()

# Plotting the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm', vmin=-1,vmax=1,center=0, annot=True,fmt=".2f")
plt.title('Correlation Matrix')
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

st.write("Enter the details below to predict if the patient is alive  or dead :")

# Input fields


ejection_fraction = st.number_input("Ejection Fraction", min_value=0, max_value=100)

serum_creatinine = st.number_input("Serum Creatinine")


# Button for prediction
st.markdown('<div class="center-button">', unsafe_allow_html=True)
if st.button("Predict"):
    features = [ejection_fraction,serum_creatinine]
    
    prediction = predict_heart_failure(features)
    result = "Dead" if prediction == 1 else "Alive"
    st.write(f"The prediction is: **{result}**")
    
    if result=="Alive":
        st.balloons()
st.markdown('</div>', unsafe_allow_html=True)

st.image('conclusion page.png', use_column_width=True)
    





