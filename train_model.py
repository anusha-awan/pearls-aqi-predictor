import os
import pandas as pd
import joblib
import hopsworks

from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ==========================================================
# CONFIGURATION
# ==========================================================

FEATURE_GROUP_NAME = "aqi_features_v2"
FEATURE_GROUP_VERSION = 1

LOCAL_DATA_FILE = "features.csv"


# ==========================================================
# LOAD ENVIRONMENT
# ==========================================================

load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")


# ==========================================================
# LOAD DATA
# ==========================================================

print("=" * 60)
print("AQI TRAINING PIPELINE")
print("=" * 60)

df = None


# ==========================================================
# TRY HOPSWORKS FEATURE STORE
# ==========================================================

if api_key:

    try:

        print("\nConnecting to Hopsworks...")

        project = hopsworks.login(
            api_key_value=api_key
        )

        print("Hopsworks login successful.")

        fs = project.get_feature_store()

        print("Feature Store connected.")

        print("\nAttempting to load features from Hopsworks...")

        feature_group = fs.get_feature_group(
            name=FEATURE_GROUP_NAME,
            version=FEATURE_GROUP_VERSION
        )

        df = (
            feature_group
            .select_all()
            .read()
        )

        print(
            "Successfully loaded",
            len(df),
            "rows from Hopsworks Feature Store."
        )

    except Exception as e:

        print("\nHopsworks Feature Store read failed.")

        print(
            "Reason:",
            str(e)
        )

        print(
            "\nUsing local features.csv fallback."
        )


# ==========================================================
# LOCAL FALLBACK
# ==========================================================

if df is None:

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
# CLEAN DATA
# ==========================================================

df = (
    df
    .dropna(
        subset=features + ["target_aqi"]
    )
    .reset_index(drop=True)
)


print("\nTraining dataset ready.")

print(
    "Rows:",
    len(df)
)

print(
    "Features:",
    len(features)
)


# ==========================================================
# INPUT / TARGET
# ==========================================================

X = df[features]

y = df["target_aqi"]


# ==========================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================================

split_index = int(
    len(df) * 0.8
)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print(
    "\nTraining rows:",
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

    "Ridge Regression":

        Pipeline([
            (
                "scaler",
                StandardScaler()
            ),

            (
                "model",
                Ridge(
                    alpha=1.0
                )
            )
        ]),

    "Random Forest":

        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ),

    "Gradient Boosting":

        GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        )
}


# ==========================================================
# TRAIN AND EVALUATE
# ==========================================================

results = []

trained_models = {}


for name, model in models.items():

    print()
    print("=" * 60)
    print("Training:", name)
    print("=" * 60)

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = mean_squared_error(
        y_test,
        y_pred
    ) ** 0.5

    r2 = r2_score(
        y_test,
        y_pred
    )

    print(
        f"MAE : {mae:.6f}"
    )

    print(
        f"RMSE: {rmse:.6f}"
    )

    print(
        f"R2  : {r2:.6f}"
    )

    results.append({

        "model": name,

        "MAE": mae,

        "RMSE": rmse,

        "R2": r2
    })

    trained_models[name] = model


# ==========================================================
# MODEL COMPARISON
# ==========================================================

results_df = pd.DataFrame(
    results
)

print()
print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================================
# SELECT BEST MODEL
# ==========================================================

best_model_name = (
    results_df
    .sort_values(
        "RMSE"
    )
    .iloc[0]["model"]
)

best_model = trained_models[
    best_model_name
]


print()
print("=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Best model:",
    best_model_name
)


# ==========================================================
# SAVE BEST MODEL
# ==========================================================

joblib.dump(
    best_model,
    "aqi_model.pkl"
)

print(
    "\nBest model saved:"
)

print(
    "aqi_model.pkl"
)


# ==========================================================
# SAVE COMPARISON
# ==========================================================

results_df.to_csv(
    "model_comparison.csv",
    index=False
)


# ==========================================================
# SAVE METADATA
# ==========================================================

metadata = {

    "best_model":
        best_model_name,

    "features":
        features,

    "forecast_horizon":
        "72 hours",

    "selection_metric":
        "RMSE",

    "training_source":
        "Hopsworks Feature Store with local features.csv fallback",

    "training_rows":
        len(X_train),

    "testing_rows":
        len(X_test),

    "MAE":
        float(
            results_df
            .loc[
                results_df["model"] == best_model_name,
                "MAE"
            ]
            .iloc[0]
        ),

    "RMSE":
        float(
            results_df
            .loc[
                results_df["model"] == best_model_name,
                "RMSE"
            ]
            .iloc[0]
        ),

    "R2":
        float(
            results_df
            .loc[
                results_df["model"] == best_model_name,
                "R2"
            ]
            .iloc[0]
        )
}


joblib.dump(
    metadata,
    "model_metadata.pkl"
)


# ==========================================================
# FINAL
# ==========================================================

print()
print("=" * 60)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "Best model:",
    best_model_name
)

print(
    "Model saved: aqi_model.pkl"
)

print(
    "Comparison saved: model_comparison.csv"
)

print(
    "Metadata saved: model_metadata.pkl"
)

print("=" * 60)