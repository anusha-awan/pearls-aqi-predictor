import pandas as pd
import joblib
from datetime import timedelta


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_FILE = "aqi_model.pkl"
DATA_FILE = "features.csv"
OUTPUT_FILE = "predictions_3_days.csv"

FORECAST_HOURS = 72


# =========================================================
# MODEL FEATURES
# =========================================================

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


# =========================================================
# LOAD MODEL
# =========================================================

print("=" * 60)
print("PEARLS AQI PREDICTION")
print("=" * 60)

print("\nLoading trained model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")
print("Model type:", type(model).__name__)


# =========================================================
# LOAD FEATURE DATA
# =========================================================

print("\nLoading feature data...")

df = pd.read_csv(DATA_FILE)

df["datetime"] = pd.to_datetime(
    df["datetime"],
    utc=True
)

df = (
    df
    .sort_values("datetime")
    .reset_index(drop=True)
)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# =========================================================
# VALIDATE MODEL FEATURES
# =========================================================

missing_features = [
    feature
    for feature in MODEL_FEATURES
    if feature not in df.columns
]

if missing_features:

    raise ValueError(
        "Missing model features:\n"
        + "\n".join(missing_features)
    )

print("All model features are available.")


# =========================================================
# REMOVE ROWS WITH MISSING MODEL FEATURES
# =========================================================

df_valid = (
    df[
        MODEL_FEATURES + ["datetime"]
    ]
    .dropna()
    .reset_index(drop=True)
)

if len(df_valid) == 0:

    raise ValueError(
        "No valid rows available for prediction."
    )


# =========================================================
# LATEST AVAILABLE ROW
# =========================================================

latest_row = df_valid.iloc[-1].copy()

latest_time = latest_row["datetime"]

print("\nLatest available data:")
print("Datetime:", latest_time)
print("Current AQI:", latest_row["aqi"])
print("Current PM2.5:", latest_row["pm2_5"])


# =========================================================
# CREATE HISTORY FOR RECURSIVE PREDICTION
# =========================================================
#
# We maintain the recent AQI history so that lag and
# rolling features can be updated using previous
# predictions.
#

aqi_history = (
    df["aqi"]
    .dropna()
    .astype(float)
    .tolist()
)

if len(aqi_history) < 72:

    raise ValueError(
        "At least 72 historical AQI observations "
        "are required for recursive forecasting."
    )


# Keep only the recent history needed for lag features.

aqi_history = aqi_history[-72:]


# PM2.5 and PM10 history

pm25_history = (
    df["pm2_5"]
    .dropna()
    .astype(float)
    .tolist()
)

pm10_history = (
    df["pm10"]
    .dropna()
    .astype(float)
    .tolist()
)

if len(pm25_history) == 0 or len(pm10_history) == 0:

    raise ValueError(
        "Pollution history is unavailable."
    )


latest_pm25 = pm25_history[-1]
latest_pm10 = pm10_history[-1]


# =========================================================
# GENERATE 72-HOUR FORECAST
# =========================================================

predictions = []

print("\nGenerating 72-hour EPA AQI forecast...")


for hour_ahead in range(
    1,
    FORECAST_HOURS + 1
):

    future_time = (
        latest_time
        + timedelta(hours=hour_ahead)
    )


    # -----------------------------------------------------
    # Start from latest known row
    # -----------------------------------------------------

    future_row = latest_row.copy()


    # -----------------------------------------------------
    # Update time features
    # -----------------------------------------------------

    future_row["datetime"] = future_time

    future_row["hour"] = future_time.hour

    future_row["day"] = future_time.day

    future_row["month"] = future_time.month

    future_row["day_of_week"] = (
        future_time.dayofweek
    )


    # -----------------------------------------------------
    # AQI lag features
    # -----------------------------------------------------

    future_row["aqi_lag_1"] = (
        aqi_history[-1]
    )

    future_row["aqi_lag_3"] = (
        aqi_history[-3]
    )

    future_row["aqi_lag_6"] = (
        aqi_history[-6]
    )

    future_row["aqi_lag_12"] = (
        aqi_history[-12]
    )

    future_row["aqi_lag_24"] = (
        aqi_history[-24]
    )

    future_row["aqi_lag_48"] = (
        aqi_history[-48]
    )

    future_row["aqi_lag_72"] = (
        aqi_history[-72]
    )


    # -----------------------------------------------------
    # Pollution lag features
    # -----------------------------------------------------

    future_row["pm2_5_lag_1"] = (
        latest_pm25
    )

    future_row["pm10_lag_1"] = (
        latest_pm10
    )


    # -----------------------------------------------------
    # Rolling AQI features
    # -----------------------------------------------------

    future_row["aqi_rolling_6"] = (
        sum(aqi_history[-6:])
        / 6
    )

    future_row["aqi_rolling_24"] = (
        sum(aqi_history[-24:])
        / 24
    )

    future_row["aqi_rolling_72"] = (
        sum(aqi_history[-72:])
        / 72
    )


    # -----------------------------------------------------
    # AQI change
    # -----------------------------------------------------

    future_row["aqi_change"] = (
        aqi_history[-1]
        - aqi_history[-2]
    )


    # -----------------------------------------------------
    # Current AQI input
    # -----------------------------------------------------

    future_row["aqi"] = (
        aqi_history[-1]
    )


    # -----------------------------------------------------
    # Prepare model input
    # -----------------------------------------------------

    X_future = pd.DataFrame(
        [future_row]
    )[MODEL_FEATURES]


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    predicted_aqi = model.predict(
        X_future
    )[0]


    # -----------------------------------------------------
    # EPA AQI range
    # -----------------------------------------------------
    #
    # IMPORTANT:
    # This is EPA-style AQI: 0–500.
    # DO NOT use the old 1–5 OpenWeather scale.
    #

    predicted_aqi = max(
        0,
        min(
            500,
            float(predicted_aqi)
        )
    )


    predicted_aqi = round(
        predicted_aqi,
        1
    )


    # -----------------------------------------------------
    # Save prediction
    # -----------------------------------------------------

    predictions.append({

        "datetime": future_time,

        "predicted_aqi": predicted_aqi,

        "day": future_time.date()

    })


    # -----------------------------------------------------
    # Add prediction to history
    #
    # This makes the next prediction recursive.
    # -----------------------------------------------------

    aqi_history.append(
        predicted_aqi
    )

    aqi_history = aqi_history[-72:]


# =========================================================
# CREATE OUTPUT DATAFRAME
# =========================================================

prediction_df = pd.DataFrame(
    predictions
)


# =========================================================
# SAVE
# =========================================================

prediction_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# SUMMARY
# =========================================================

print("\n" + "=" * 60)
print("72-HOUR FORECAST GENERATED SUCCESSFULLY")
print("=" * 60)

print(
    "\nForecast start:",
    prediction_df["datetime"].iloc[0]
)

print(
    "Forecast end:",
    prediction_df["datetime"].iloc[-1]
)

print(
    "Minimum predicted AQI:",
    prediction_df["predicted_aqi"].min()
)

print(
    "Maximum predicted AQI:",
    prediction_df["predicted_aqi"].max()
)

print(
    "Average predicted AQI:",
    round(
        prediction_df["predicted_aqi"].mean(),
        1
    )
)

print(
    "\nFirst 10 predictions:"
)

print(
    prediction_df.head(10).to_string(
        index=False
    )
)

print(
    "\nLast 10 predictions:"
)

print(
    prediction_df.tail(10).to_string(
        index=False
    )
)

print(
    "\nPredictions saved as:",
    OUTPUT_FILE
)

print("=" * 60)