import streamlit as st
import pandas as pd
import joblib

# ---------------------------
# Load Model
# ---------------------------

model = joblib.load("sales_forecast_model.pkl")

st.title("📊 AI-Powered Sales Forecasting")

st.write("Enter required values to predict next month's sales.")

# ---------------------------
# User Inputs
# ---------------------------

year = st.number_input("Year", min_value=2015, max_value=2035, value=2019)
month = st.slider("Month", 1, 12, 1)

lag1 = st.number_input("Previous Month Sales (Lag_1)", min_value=0.0, value=50000.0)
lag2 = st.number_input("2 Months Ago Sales (Lag_2)", min_value=0.0, value=48000.0)
lag3 = st.number_input("3 Months Ago Sales (Lag_3)", min_value=0.0, value=45000.0)

rolling3 = st.number_input("3-Month Rolling Average", min_value=0.0, value=49000.0)
rolling6 = st.number_input("6-Month Rolling Average", min_value=0.0, value=47000.0)

# ⚠️ IMPORTANT — Use SAME base year as training
BASE_YEAR = 2015   # Change if your dataset starts from different year
year_index = year - BASE_YEAR

# ---------------------------
# Prediction
# ---------------------------

if st.button("Predict Sales"):

    # EXACT same column names + same order
    input_data = pd.DataFrame(
        [[
            year_index,
            month,
            lag1,
            lag2,
            lag3,
            rolling3,
            rolling6
        ]],
        columns=[
            "Year_Index",
            "Month",
            "Lag_1",
            "Lag_2",
            "Lag_3",
            "Rolling_Mean_3",
            "Rolling_Mean_6"
        ]
    )

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Sales: ₹ {round(prediction, 2)}")
