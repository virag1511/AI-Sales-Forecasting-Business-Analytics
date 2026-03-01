import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ---------------------------
# 1. Load Dataset
# ---------------------------
df = pd.read_csv("train.csv")
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

# ---------------------------
# 2. Monthly Aggregation
# ---------------------------
df["YearMonth"] = df["Order Date"].dt.to_period("M")
monthly = df.groupby("YearMonth")["Sales"].sum().reset_index()
monthly["YearMonth"] = monthly["YearMonth"].dt.to_timestamp()

# ---------------------------
# 3. Feature Engineering
# ---------------------------
monthly["Year"] = monthly["YearMonth"].dt.year
monthly["Month"] = monthly["YearMonth"].dt.month

# Cyclical Encoding
monthly["Month_sin"] = np.sin(2 * np.pi * monthly["Month"] / 12)
monthly["Month_cos"] = np.cos(2 * np.pi * monthly["Month"] / 12)
monthly["Quarter"] = monthly["YearMonth"].dt.quarter

# Lag Feature (Previous Month Sales)
monthly["Lag_1"] = monthly["Sales"].shift(1)
monthly["Lag_2"] = monthly["Sales"].shift(2)

# Rolling Mean (3 month average)
monthly["Rolling_Mean_3"] = monthly["Sales"].rolling(window=3).mean()
monthly["Rolling_Mean_6"] = monthly["Sales"].rolling(6).mean()

# Drop NaN values created by lag/rolling
monthly = monthly.dropna()
monthly = monthly.dropna()

# ---------------------------
# 4. Model Preparation
# ---------------------------
X = monthly[[
    "Year",
    "Month",
    "Lag_1",
    "Lag_2",
    "Rolling_Mean_3",
    "Rolling_Mean_6"
]]
y = monthly["Sales"]

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ---------------------------
# 5. Train Model
# ---------------------------
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train)
predictions = model.predict(X_test)

# ---------------------------
# 6. Evaluation
# ---------------------------
print("\nModel Performance:")
print("MAE:", round(mean_absolute_error(y_test, predictions), 2))
print("R2 Score:", round(r2_score(y_test, predictions), 3))

# ---------------------------
# 7. Save Model
# ---------------------------
joblib.dump(model, "sales_forecast_model.pkl")
print("\nModel saved as sales_forecast_model.pkl")

# ---------------------------
# 8. Plot Results
# ---------------------------
plt.figure(figsize=(10,5))
plt.plot(monthly["YearMonth"], monthly["Sales"], label="Actual Sales")
plt.plot(monthly["YearMonth"][split:], predictions, label="Predicted Sales")
plt.legend()
plt.title("Advanced AI Sales Forecasting Model")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.show()
# ---------------------------
# 9. Feature Importance
# ---------------------------
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(6,4))
plt.bar(features, importances)
plt.title("Feature Importance")
plt.xticks(rotation=45)
plt.show()
# ---------------------------
# 10. Future Forecast (Walk Forward)
# ---------------------------
future_predictions = []

last_known = monthly.iloc[-1].copy()

for i in range(6):
    new_row = {}

    new_row["Year"] = last_known["YearMonth"].year
    new_row["Month"] = last_known["YearMonth"].month + 1
    if new_row["Month"] > 12:
        new_row["Month"] = 1
        new_row["Year"] += 1

    # Required features
    new_row["Lag_1"] = last_known["Sales"]
    new_row["Lag_2"] = monthly["Sales"].iloc[-2]
    new_row["Rolling_Mean_3"] = monthly["Sales"].tail(3).mean()
    new_row["Rolling_Mean_6"] = monthly["Sales"].tail(6).mean()

    new_X = pd.DataFrame([new_row])

    # IMPORTANT: same column order as training
    new_X = new_X[[
        "Year",
        "Month",
        "Lag_1",
        "Lag_2",
        "Rolling_Mean_3",
        "Rolling_Mean_6"
    ]]

    pred = model.predict(new_X)[0]
    future_predictions.append(pred)

    last_known["Sales"] = pred
    last_known["YearMonth"] = pd.Timestamp(
        year=new_row["Year"], month=new_row["Month"], day=1
    )

print("\nFuture 6-Month Forecast:")
for i, val in enumerate(future_predictions, 1):
    print(f"Month {i}: {round(val,2)}")


    #.....
    from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)

print("\nLinear Regression R2:", round(r2_score(y_test, lr_preds), 3))
#........
from sklearn.model_selection import GridSearchCV

param_grid = {
    "n_estimators": [200, 500],
    "max_depth": [5, 10, 15]
}

grid = GridSearchCV(RandomForestRegressor(random_state=42),
                    param_grid,
                    cv=3)

grid.fit(X_train, y_train)
model = grid.best_estimator_

print("Best Parameters:", grid.best_params_)
monthly.to_csv("D:/Projects/Sales_Forecasting_Model/monthly_sales_output.csv", index=False)
