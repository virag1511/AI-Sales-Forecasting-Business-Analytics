import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load Model
# ---------------------------

model = joblib.load("sales_forecast_model.pkl")

st.title("📊 AI-Powered Sales Forecasting")

st.write("Enter the required values to predict next month's sales.")

# ---------------------------
# User Inputs
# ---------------------------

year = st.number_input("Year", min_value=2015, max_value=2035, value=2020)
month = st.slider("Month", 1, 12, 1)

lag1 = st.number_input("Previous Month Sales (Lag_1)", min_value=0.0)
lag2 = st.number_input("2 Months Ago Sales (Lag_2)", min_value=0.0)
lag3 = st.number_input("3 Months Ago Sales (Lag_3)", min_value=0.0)

rolling3 = st.number_input("3-Month Rolling Average", min_value=0.0)
rolling6 = st.number_input("6-Month Rolling Average", min_value=0.0)

# Year index calculation (important)
base_year = 2015  # Change this if your dataset starts from a different year
year_index = year - base_year

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Sales"):

    input_data = pd.DataFrame([{
        "Year_Index": year_index,
        "Month": month,
        "Lag_1": lag1,
        "Lag_2": lag2,
        "Lag_3": lag3,
        "Rolling_Mean_3": rolling3,
        "Rolling_Mean_6": rolling6
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Sales: ₹ {round(prediction, 2)}")

    st.write("This prediction is based on historical trends and lag features.")
