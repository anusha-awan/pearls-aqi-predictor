import pandas as pd
import joblib

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("AQI MODEL EVALUATION")
print("=" * 60)

print("\nLoading training data...")

df = pd.read_csv("training_data.csv")

df["datetime"] = pd.to_datetime(
    df["datetime"]
)

df = (
    df
    .sort_values("datetime")
    .reset_index(drop=True)
)


# ==========================================================
# FEATURES
# ==========================================================

features = [
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "pm2_5",
    "pm10",
    "nh3",
    "hour",
    "day",
    "month",
    "day_of_week",
    "aqi",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_lag_48",
    "aqi_lag_72",
    "pm2_5_lag_1",
    "pm10_lag_1",
    "aqi_rolling_6",
    "aqi_rolling_24",
    "aqi_rolling_72",
    "aqi_change"
]


# ==========================================================
# CLEAN
# ==========================================================

df = df.dropna(
    subset=features + ["target_aqi"]
).reset_index(drop=True)


print(
    "Valid rows:",
    len(df)
)


# ==========================================================
# CHRONOLOGICAL 80/20 SPLIT
# ==========================================================

split_index = int(
    len(df) * 0.8
)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]


X_test = test_df[features]
y_test = test_df["target_aqi"]


print(
    "Training rows:",
    len(train_df)
)

print(
    "Testing rows:",
    len(test_df)
)


# ==========================================================
# LOAD LOCAL BEST MODEL
# ==========================================================

print(
    "\nLoading local Random Forest model..."
)

model = joblib.load(
    "aqi_model.pkl"
)

print(
    "Model loaded successfully!"
)


# ==========================================================
# PREDICTIONS
# ==========================================================

print(
    "\nGenerating test predictions..."
)

predictions = model.predict(
    X_test
)


# ==========================================================
# METRICS
# ==========================================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


# ==========================================================
# RESULTS
# ==========================================================

print("\n" + "=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

print(
    f"\nMAE  : {mae:.2f} AQI points"
)

print(
    f"RMSE : {rmse:.2f} AQI points"
)

print(
    f"R²   : {r2:.4f}"
)

print(
    "\nActual AQI range in test set:",
    f"{y_test.min():.0f} - {y_test.max():.0f}"
)

print(
    "Predicted AQI range:",
    f"{predictions.min():.0f} - {predictions.max():.0f}"
)

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)