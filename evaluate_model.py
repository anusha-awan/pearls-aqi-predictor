import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# LOAD TRAINING DATA
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

MODEL_FEATURES = [
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
# CLEAN DATA
# ==========================================================

df = df.dropna(
    subset=MODEL_FEATURES + ["target_aqi"]
).reset_index(drop=True)


print("Valid rows:", len(df))


# ==========================================================
# CHRONOLOGICAL 80/20 SPLIT
# ==========================================================

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]


print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))


# ==========================================================
# LOAD LOCAL RANDOM FOREST MODEL
# ==========================================================

print("\nLoading local Random Forest model...")

model = joblib.load(
    "aqi_model.pkl"
)

print("Model loaded successfully!")


# ==========================================================
# RANDOM FOREST PREDICTIONS
# ==========================================================

print("\nGenerating Random Forest test predictions...")

X_test = test_df[MODEL_FEATURES]

y_test = test_df["target_aqi"]

rf_predictions = model.predict(
    X_test
)


# ==========================================================
# PERSISTENCE BASELINE
# ==========================================================

print("Generating persistence baseline predictions...")

# Persistence assumes the next-hour AQI
# will be equal to the current AQI.

persistence_predictions = (
    test_df["aqi"]
    .values
)


# ==========================================================
# RANDOM FOREST METRICS
# ==========================================================

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

rf_rmse = (
    mean_squared_error(
        y_test,
        rf_predictions
    )
    ** 0.5
)

rf_r2 = r2_score(
    y_test,
    rf_predictions
)


# ==========================================================
# PERSISTENCE METRICS
# ==========================================================

p_mae = mean_absolute_error(
    y_test,
    persistence_predictions
)

p_rmse = (
    mean_squared_error(
        y_test,
        persistence_predictions
    )
    ** 0.5
)

p_r2 = r2_score(
    y_test,
    persistence_predictions
)


# ==========================================================
# RESULTS
# ==========================================================

print("\n" + "=" * 60)
print("FINAL MODEL COMPARISON")
print("=" * 60)

print("\nRandom Forest:")
print(
    f"MAE  : {rf_mae:.2f} AQI points"
)
print(
    f"RMSE : {rf_rmse:.2f} AQI points"
)
print(
    f"R²   : {rf_r2:.4f}"
)


print("\nPersistence Baseline:")
print(
    f"MAE  : {p_mae:.2f} AQI points"
)
print(
    f"RMSE : {p_rmse:.2f} AQI points"
)
print(
    f"R²   : {p_r2:.4f}"
)


# ==========================================================
# COMPARISON
# ==========================================================

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)


if rf_mae < p_mae:
    print("✅ Random Forest beats persistence on MAE.")
else:
    print("⚠️ Random Forest does not beat persistence on MAE.")


if rf_rmse < p_rmse:
    print("✅ Random Forest beats persistence on RMSE.")
else:
    print("⚠️ Random Forest does not beat persistence on RMSE.")


if rf_r2 > p_r2:
    print("✅ Random Forest beats persistence on R².")
else:
    print("⚠️ Random Forest does not beat persistence on R².")


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)