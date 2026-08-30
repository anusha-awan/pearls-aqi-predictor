import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

print("Loading training data...")

df = pd.read_csv("training_data.csv")

df["datetime"] = pd.to_datetime(df["datetime"])

print("Total rows:", len(df))


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
# REMOVE MISSING VALUES
# ==========================================================

df = df.dropna(
    subset=features + ["target_aqi"]
).reset_index(drop=True)

print("Rows after removing missing values:", len(df))


# ==========================================================
# INPUT / TARGET
# ==========================================================

X = df[features]

y = df["target_aqi"]


# ==========================================================
# CHRONOLOGICAL TRAIN / TEST SPLIT
# ==========================================================

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]


print()
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))


# ==========================================================
# DEFINE MODELS
# ==========================================================

models = {

    "Ridge Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ]),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}


# ==========================================================
# TRAIN AND EVALUATE MODELS
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

    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R²  :", r2)

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

# Primary criterion:
# Lowest RMSE

best_model_name = (
    results_df
    .sort_values("RMSE")
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

print()
print("Best model saved as:")
print("aqi_model.pkl")


# ==========================================================
# SAVE MODEL COMPARISON
# ==========================================================

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

print()
print("Model comparison saved as:")
print("model_comparison.csv")


# ==========================================================
# SAVE MODEL METADATA
# ==========================================================

metadata = {

    "best_model": best_model_name,

    "features": features,

    "forecast_horizon": "72 hours",

    "selection_metric": "RMSE"

}

joblib.dump(
    metadata,
    "model_metadata.pkl"
)

print()
print("Model metadata saved as:")
print("model_metadata.pkl")


print()
print("=" * 60)
print("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)