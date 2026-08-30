import pandas as pd
import joblib
from datetime import timedelta

# Load trained model
model = joblib.load("aqi_model.pkl")

# Load historical data
df = pd.read_csv("historical_aqi.csv")

# Convert datetime
df["datetime"] = pd.to_datetime(df["datetime"])

# Sort data
df = df.sort_values("datetime").reset_index(drop=True)

# Create the same features used during training
df["hour"] = df["datetime"].dt.hour
df["day"] = df["datetime"].dt.day
df["month"] = df["datetime"].dt.month
df["day_of_week"] = df["datetime"].dt.dayofweek

# AQI lag features
df["aqi_lag_1"] = df["aqi"].shift(1)
df["aqi_lag_3"] = df["aqi"].shift(3)
df["aqi_lag_6"] = df["aqi"].shift(6)
df["aqi_lag_12"] = df["aqi"].shift(12)
df["aqi_lag_24"] = df["aqi"].shift(24)
df["aqi_lag_48"] = df["aqi"].shift(48)
df["aqi_lag_72"] = df["aqi"].shift(72)

# Pollution lag features
df["pm2_5_lag_1"] = df["pm2_5"].shift(1)
df["pm10_lag_1"] = df["pm10"].shift(1)

# Rolling AQI features
df["aqi_rolling_6"] = df["aqi"].rolling(6).mean()
df["aqi_rolling_24"] = df["aqi"].rolling(24).mean()
df["aqi_rolling_72"] = df["aqi"].rolling(72).mean()

# AQI change
df["aqi_change"] = df["aqi"].diff()

# Remove rows with missing feature values
df = df.dropna().reset_index(drop=True)

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

# Use latest available row as the starting point
latest_row = df.iloc[-1].copy()

print("Latest available data:")
print(latest_row["datetime"])
print("Current AQI:", latest_row["aqi"])

# Store predictions
predictions = []

# Make 72 hourly predictions
for i in range(1, 73):

    future_time = latest_row["datetime"] + timedelta(hours=i)

    # Start with latest known values
    future_row = latest_row.copy()

    # Update time-based features
    future_row["datetime"] = future_time
    future_row["hour"] = future_time.hour
    future_row["day"] = future_time.day
    future_row["month"] = future_time.month
    future_row["day_of_week"] = future_time.dayofweek

    # Predict AQI
    X_future = pd.DataFrame([future_row])[features]

    predicted_aqi = model.predict(X_future)[0]

    # Keep AQI within valid range
    predicted_aqi = max(1, min(5, predicted_aqi))

    predictions.append({
        "datetime": future_time,
        "predicted_aqi": round(predicted_aqi, 2)
    })

# Convert predictions to dataframe
prediction_df = pd.DataFrame(predictions)

# Add day number
prediction_df["day"] = (
    prediction_df["datetime"].dt.date
)

# Save predictions
prediction_df.to_csv("predictions_3_days.csv", index=False)

print("\n3-Day AQI Prediction:")
print(prediction_df)

print("\nPredictions saved as: predictions_3_days.csv")