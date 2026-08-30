import os
import joblib
import hopsworks
import pandas as pd

from dotenv import load_dotenv
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()


# =========================================================
# CONSTANTS
# =========================================================

FEATURE_GROUP_NAME = "aqi_features"
FEATURE_GROUP_VERSION = 1

MODEL_NAME = "aqi_random_forest"
MODEL_VERSION = 2


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
# CONNECT TO HOPSWORKS
# =========================================================

print("=" * 60)
print("AQI MODEL EVALUATION")
print("=" * 60)

api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    raise ValueError("HOPSWORKS_API_KEY not found.")

print("\nConnecting to Hopsworks...")

project = hopsworks.login(
    api_key_value=api_key
)

fs = project.get_feature_store()

print("Connected successfully!")


# =========================================================
# LOAD FEATURE GROUP
# =========================================================

print("\nLoading feature data...")

fg = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION
)

df = fg.select_all().read()

df["datetime"] = pd.to_datetime(
    df["datetime"]
)

df = (
    df
    .sort_values("datetime")
    .reset_index(drop=True)
)

print(f"Rows loaded: {len(df)}")


# =========================================================
# CLEAN DATA
# =========================================================

df = df.dropna(
    subset=MODEL_FEATURES + ["datetime"]
).reset_index(drop=True)

print(f"Valid rows: {len(df)}")


# =========================================================
# LOAD MODEL
# =========================================================

print("\nLoading Random Forest model...")

model_registry = project.get_model_registry()

model = model_registry.get_model(
    name=MODEL_NAME,
    version=MODEL_VERSION
)

model_dir = model.download()

model_file = None

for root, dirs, files in os.walk(model_dir):

    for file in files:

        if file.endswith(".pkl"):

            model_file = os.path.join(
                root,
                file
            )

            break

    if model_file:
        break


if not model_file:

    raise FileNotFoundError(
        "No .pkl model file found."
    )


trained_model = joblib.load(
    model_file
)

print("Model loaded successfully!")


# =========================================================
# PREPARE DATA
# =========================================================

X = df[MODEL_FEATURES]

y = df["aqi"]


# =========================================================
# MODEL PREDICTIONS
# =========================================================

print("\nGenerating predictions...")

predictions = trained_model.predict(X)


# =========================================================
# EVALUATION METRICS
# =========================================================

rmse = mean_squared_error(
    y,
    predictions
) ** 0.5

mae = mean_absolute_error(
    y,
    predictions
)

r2 = r2_score(
    y,
    predictions
)


# =========================================================
# RESULTS
# =========================================================

print("\n" + "=" * 60)
print("MODEL EVALUATION RESULTS")
print("=" * 60)

print(f"\nRMSE : {rmse:.4f}")
print(f"MAE  : {mae:.4f}")
print(f"R²   : {r2:.4f}")

print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)