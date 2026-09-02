import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================================
# CONFIGURATION
# ==========================================================

# ==========================================================
# CONFIGURATION
# ==========================================================

DATA_FILE = "features.csv"
MODEL_FILE = "aqi_model.pkl"

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
# START
# ==========================================================

print("=" * 60)
print("PEARLS AQI MODEL EVALUATION")
print("=" * 60)


# ==========================================================
# LOAD DATA
# ==========================================================

print("\nLoading training data...")

df = pd.read_csv(DATA_FILE)

df["datetime"] = pd.to_datetime(
    df["datetime"],
    utc=True,
    errors="coerce"
)

df = (
    df
    .sort_values("datetime")
    .reset_index(drop=True)
)


print("Rows loaded:", len(df))
print("Columns:", len(df.columns))


# ==========================================================
# VALIDATE FEATURES
# ==========================================================

print("\nValidating model features...")

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


if "target_aqi" not in df.columns:

    raise ValueError(
        "target_aqi column not found."
    )


print("All 26 model features are available.")


# ==========================================================
# CLEAN DATA
# ==========================================================

print("\nCleaning evaluation dataset...")

before = len(df)

df = df.dropna(
    subset=MODEL_FEATURES + ["target_aqi"]
).reset_index(drop=True)

after = len(df)

print(
    "Rows removed because of missing values:",
    before - after
)

print(
    "Valid evaluation rows:",
    after
)


# ==========================================================
# CHRONOLOGICAL 80/20 SPLIT
# ==========================================================

print("\nCreating chronological 80/20 split...")

split_index = int(len(df) * 0.8)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()


print(
    "Training rows:",
    len(train_df)
)

print(
    "Testing rows:",
    len(test_df)
)


print(
    "Training period:",
    train_df["datetime"].min(),
    "->",
    train_df["datetime"].max()
)

print(
    "Testing period:",
    test_df["datetime"].min(),
    "->",
    test_df["datetime"].max()
)


# ==========================================================
# LOAD MODEL
# ==========================================================

print("\nLoading trained AQI model...")

model = joblib.load(
    MODEL_FILE
)

print(
    "Model loaded successfully!"
)

print(
    "Model type:",
    type(model).__name__
)


# ==========================================================
# GENERATE MODEL PREDICTIONS
# ==========================================================

print("\nGenerating model predictions...")

X_test = test_df[
    MODEL_FEATURES
]

y_test = test_df[
    "target_aqi"
]

predictions = model.predict(
    X_test
)


# Keep predictions within valid EPA AQI range
predictions = predictions.clip(
    0,
    500
)


# ==========================================================
# PERSISTENCE BASELINE
# ==========================================================

print(
    "Generating persistence baseline..."
)

baseline_predictions = (
    test_df["aqi"]
    .values
)

baseline_predictions = baseline_predictions.clip(
    0,
    500
)


# ==========================================================
# MODEL METRICS
# ==========================================================

model_mae = mean_absolute_error(
    y_test,
    predictions
)

model_rmse = (
    mean_squared_error(
        y_test,
        predictions
    )
    ** 0.5
)

model_r2 = r2_score(
    y_test,
    predictions
)


# ==========================================================
# BASELINE METRICS
# ==========================================================

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

baseline_rmse = (
    mean_squared_error(
        y_test,
        baseline_predictions
    )
    ** 0.5
)

baseline_r2 = r2_score(
    y_test,
    baseline_predictions
)


# ==========================================================
# RESULTS
# ==========================================================

print("\n" + "=" * 60)
print("FINAL MODEL EVALUATION")
print("=" * 60)


print("\nGradient Boosting Model:")

print(
    f"MAE  : {model_mae:.2f} AQI points"
)

print(
    f"RMSE : {model_rmse:.2f} AQI points"
)

print(
    f"R²   : {model_r2:.4f}"
)


print("\nPersistence Baseline:")

print(
    f"MAE  : {baseline_mae:.2f} AQI points"
)

print(
    f"RMSE : {baseline_rmse:.2f} AQI points"
)

print(
    f"R²   : {baseline_r2:.4f}"
)


# ==========================================================
# IMPROVEMENT
# ==========================================================

mae_improvement = (
    (baseline_mae - model_mae)
    / baseline_mae
) * 100

rmse_improvement = (
    (baseline_rmse - model_rmse)
    / baseline_rmse
) * 100


print("\n" + "=" * 60)
print("MODEL IMPROVEMENT")
print("=" * 60)

print(
    f"MAE improvement : {mae_improvement:.2f}%"
)

print(
    f"RMSE improvement: {rmse_improvement:.2f}%"
)


# ==========================================================
# COMPARISON
# ==========================================================

print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)


if model_mae < baseline_mae:

    print(
        "PASS: Model beats persistence on MAE."
    )

else:

    print(
        "WARNING: Model does not beat persistence on MAE."
    )


if model_rmse < baseline_rmse:

    print(
        "PASS: Model beats persistence on RMSE."
    )

else:

    print(
        "WARNING: Model does not beat persistence on RMSE."
    )


if model_r2 > baseline_r2:

    print(
        "PASS: Model beats persistence on R²."
    )

else:

    print(
        "WARNING: Model does not beat persistence on R²."
    )


# ==========================================================
# SAVE EVALUATION RESULTS
# ==========================================================

results = pd.DataFrame({

    "metric": [
        "MAE",
        "RMSE",
        "R2"
    ],

    "gradient_boosting": [
        model_mae,
        model_rmse,
        model_r2
    ],

    "persistence_baseline": [
        baseline_mae,
        baseline_rmse,
        baseline_r2
    ]

})


results.to_csv(
    "model_evaluation_results.csv",
    index=False
)


print(
    "\nEvaluation results saved as:"
)

print(
    "model_evaluation_results.csv"
)


print("\n" + "=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)