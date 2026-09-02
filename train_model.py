
import os
import pandas as pd
import joblib

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

from sklearn.linear_model import Ridge

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ==========================================================
# CONFIGURATION
# ==========================================================

LOCAL_DATA_FILE = "features.csv"

MODEL_FILE = "aqi_model.pkl"
METADATA_FILE = "model_metadata.pkl"
COMPARISON_FILE = "model_comparison.csv"


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("AQI TRAINING PIPELINE")
print("=" * 60)

print("\nLoading local engineered dataset...")

if not os.path.exists(LOCAL_DATA_FILE):

    raise FileNotFoundError(
        "features.csv not found."
    )

df = pd.read_csv(
    LOCAL_DATA_FILE
)

print(
    "Loaded local engineered dataset:",
    len(df),
    "rows."
)


# ==========================================================
# DATETIME
# ==========================================================

df["datetime"] = pd.to_datetime(
    df["datetime"],
    utc=True
)

df = (
    df
    .sort_values("datetime")
    .drop_duplicates(
        subset=["datetime"]
    )
    .reset_index(drop=True)
)


# ==========================================================
# MODEL FEATURES
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
# VALIDATE COLUMNS
# ==========================================================

required_columns = (
    features
    +
    [
        "datetime",
        "target_aqi"
    ]
)


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        +
        "\n".join(
            missing_columns
        )
    )


# ==========================================================
# REMOVE ROWS WITH MISSING VALUES
#
# The latest row has no next-hour target.
# Therefore it is excluded from training.
# ==========================================================

training_df = (
    df
    .dropna(
        subset=features + ["target_aqi"]
    )
    .reset_index(drop=True)
)


# ==========================================================
# TARGET VALIDATION
# ==========================================================

if len(training_df) < 100:

    raise ValueError(
        "Not enough training data."
    )


target_min = training_df["target_aqi"].min()
target_max = training_df["target_aqi"].max()


if (
    target_min < 0
    or
    target_max > 500
):

    raise ValueError(
        "target_aqi must be between 0 and 500."
    )


# ==========================================================
# INPUT / TARGET
# ==========================================================

X = training_df[features]

y = training_df["target_aqi"]


print(
    "\nTraining dataset ready."
)

print(
    "Training rows:",
    len(training_df)
)

print(
    "Number of features:",
    len(features)
)

print(
    "Target:",
    "next-hour EPA-style AQI"
)

print(
    "Target range:",
    round(y.min(), 1),
    "->",
    round(y.max(), 1)
)


# ==========================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================================

split_index = int(
    len(training_df) * 0.8
)


X_train = X.iloc[
    :split_index
]

X_test = X.iloc[
    split_index:
]

y_train = y.iloc[
    :split_index
]

y_test = y.iloc[
    split_index:
]


print(
    "\nChronological split:"
)

print(
    "Training rows:",
    len(X_train)
)

print(
    "Testing rows:",
    len(X_test)
)


# ==========================================================
# MODELS
# ==========================================================

models = {

    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        max_depth=None,
        min_samples_leaf=1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ),

    "Ridge": Pipeline(
        steps=[

            (
                "scaler",
                StandardScaler()
            ),

            (
                "ridge",
                Ridge(
                    alpha=1.0
                )
            )

        ]
    )
}


# ==========================================================
# TRAIN AND EVALUATE
# ==========================================================

results = []

trained_models = {}


print(
    "\n" + "=" * 60
)

print(
    "MODEL TRAINING"
)

print(
    "=" * 60
)


for name, model in models.items():

    print(
        f"\nTraining {name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    predictions = pd.Series(
        predictions
    ).clip(
        lower=0,
        upper=500
    )

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

    results.append({

        "model": name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2

    })

    trained_models[name] = model

    print(
        f"{name} results:"
    )

    print(
        f"MAE  : {mae:.4f}"
    )

    print(
        f"RMSE : {rmse:.4f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )


# ==========================================================
# MODEL COMPARISON
# ==========================================================

comparison_df = pd.DataFrame(
    results
)

comparison_df = (
    comparison_df
    .sort_values(
        by="MAE"
    )
    .reset_index(drop=True)
)


print(
    "\n" + "=" * 60
)

print(
    "MODEL COMPARISON"
)

print(
    "=" * 60
)

print(
    comparison_df.to_string(
        index=False
    )
)


comparison_df.to_csv(
    COMPARISON_FILE,
    index=False
)


# ==========================================================
# SELECT BEST MODEL
# ==========================================================

best_model_name = (
    comparison_df
    .iloc[0]["model"]
)

best_model = (
    trained_models[
        best_model_name
    ]
)


best_mae = (
    comparison_df
    .iloc[0]["MAE"]
)

best_rmse = (
    comparison_df
    .iloc[0]["RMSE"]
)

best_r2 = (
    comparison_df
    .iloc[0]["R2"]
)


print(
    "\nBest model:",
    best_model_name
)

print(
    f"Best MAE: {best_mae:.4f}"
)

print(
    f"Best RMSE: {best_rmse:.4f}"
)

print(
    f"Best R²: {best_r2:.4f}"
)


# ==========================================================
# SAVE BEST MODEL
# ==========================================================

joblib.dump(
    best_model,
    MODEL_FILE
)


print(
    "\nSaved best model as:",
    MODEL_FILE
)


# ==========================================================
# SAVE MODEL METADATA
# ==========================================================

metadata = {

    "model_name":
        best_model_name,

    "model_version":
        4,

    "target":
        "next-hour EPA-style AQI",

    "forecast_horizon_hours":
        72,

    "features":
        features,

    "number_of_features":
        len(features),

    "training_rows":
        len(X_train),

    "testing_rows":
        len(X_test),

    "mae":
        float(best_mae),

    "rmse":
        float(best_rmse),

    "r2":
        float(best_r2),

    "target_min":
        float(y.min()),

    "target_max":
        float(y.max())

}


joblib.dump(
    metadata,
    METADATA_FILE
)


print(
    "Saved metadata as:",
    METADATA_FILE
)


# ==========================================================
# FINAL REPORT
# ==========================================================

print(
    "\n" + "=" * 60
)

print(
    "TRAINING COMPLETED SUCCESSFULLY"
)

print(
    "=" * 60
)

print(
    "\nModel:",
    best_model_name
)

print(
    "Features:",
    len(features)
)

print(
    "Forecast horizon:",
    "72 hours"
)

print(
    f"MAE: {best_mae:.2f} AQI points"
)

print(
    f"RMSE: {best_rmse:.2f} AQI points"
)

print(
    f"R²: {best_r2:.4f}"
)

print(
    "\nTarget is EPA-style AQI (0-500)."
)

print(
    "=" * 60
)
