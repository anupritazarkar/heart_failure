


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
st.markdown(
    """
    <style>
    .title {
        color: blue;
        font-size: 48px;
        text-align: center;
        font-family: 'Courier New', Courier, monospace;
        font-weight: bold;
        text-shadow: 2px 2px 4px #000000;
        border-bottom: 3px solid #add8e6;
        padding-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Add your title using the custom class
st.markdown('<h1 class="title">Heart Failure Prediction Model</h1>', unsafe_allow_html=True)





# Load and display the heart image

# Load and display the heart image
# If the image is local, use the path to the image file, for example:
image = Image.open('C:\Users\Anuprita\Downloads\rb_1516.png')  # Replace with the actual path to your image file

# For an image hosted online, use:
# st.image("https://example.com/your_heart_image.jpg", caption="Heart Health", use_column_width=True)

# Display the image in Streamlit
st.image(image, caption="Heart Health", use_column_width=True)




# Model metrics
model_accuracy = 0.90  # Example values, update with your actual model metrics
model_precision = 0.90
model_recall = 0.78

st.metric(label="Model Accuracy", value=f"{model_accuracy*100:.2f}%")
st.write(f"Precision: {model_precision:.2f}")
st.write(f"Recall: {model_recall:.2f}")

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
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, linewidths=0.5)
plt.title('Correlation Matrix (without time)')
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




