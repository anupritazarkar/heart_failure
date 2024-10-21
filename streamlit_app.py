import streamlit as st
import pandas as pd
import numpy as np


st.title("Heart Failure Prediction")
df=pd.read_csv("heart_failure_clinical_records.csv")
