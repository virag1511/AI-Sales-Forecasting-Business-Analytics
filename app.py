import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.title("AI-Powered Sales Forecasting Dashboard")

model = joblib.load("sales_forecast_model.pkl")

st.subheader("Enter Feature Values")

year = st.number_input("Year", min_value=2015, max_value=2030, value=2024)
month = st.slider("Month", 1, 12, 1)
lag1 = st.number_input("Previous Month Sales (Lag_1)")
lag2 = st.number_input("2 Months Ago Sales (Lag_2)")
rolling3 = st.number_input("3-Month Rolling Average")
rolling6 = st.number_input("6-Month Rolling Average")

if st.button("Predict Sales"):
    input_data = pd.DataFrame([{
        "Year": year,
        "Month": month,
        "Lag_1": lag1,
        "Lag_2": lag2,
        "Rolling_Mean_3": rolling3,
        "Rolling_Mean_6": rolling6
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"Predicted Sales: ₹ {round(prediction,2)}")
