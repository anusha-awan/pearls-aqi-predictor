import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================================
# SETTINGS
# ==========================================================

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

print("=" * 60)
print("TENSORFLOW AQI TRAINING")
print("=" * 60)

print()
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
# CLEAN DATA
# ==========================================================

df = df.dropna(
    subset=features + ["target_aqi"]
).reset_index(drop=True)

print("Rows after removing missing values:", len(df))


# ==========================================================
# INPUT / TARGET
# ==========================================================

X = df[features].astype(np.float32)

y = df["target_aqi"].astype(np.float32)


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
# SCALE FEATURES
# ==========================================================

print()
print("Scaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================================
# BUILD NEURAL NETWORK
# ==========================================================

print()
print("Building TensorFlow neural network...")

model = tf.keras.Sequential([
    
    tf.keras.layers.Input(
        shape=(len(features),)
    ),

    tf.keras.layers.Dense(
        128,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.2
    ),

    tf.keras.layers.Dense(
        64,
        activation="relu"
    ),

    tf.keras.layers.Dropout(
        0.2
    ),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        1
    )
])


# ==========================================================
# COMPILE
# ==========================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="mse",
    metrics=["mae"]
)


# ==========================================================
# TRAIN
# ==========================================================

print()
print("=" * 60)
print("TRAINING NEURAL NETWORK")
print("=" * 60)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

history = model.fit(
    X_train_scaled,
    y_train,
    validation_split=0.1,
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping],
    verbose=1
)


# ==========================================================
# PREDICTIONS
# ==========================================================

print()
print("Generating predictions...")

y_pred = model.predict(
    X_test_scaled,
    verbose=0
).flatten()


# ==========================================================
# EVALUATION
# ==========================================================

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)


print()
print("=" * 60)
print("TENSORFLOW MODEL RESULTS")
print("=" * 60)

print(f"MAE : {mae:.6f}")
print(f"RMSE: {rmse:.6f}")
print(f"R²  : {r2:.6f}")


# ==========================================================
# SAVE MODEL
# ==========================================================

model.save(
    "aqi_tensorflow_model.keras"
)

print()
print("TensorFlow model saved:")
print("aqi_tensorflow_model.keras")


# ==========================================================
# SAVE SCALER
# ==========================================================

import joblib

joblib.dump(
    scaler,
    "tensorflow_scaler.pkl"
)

print()
print("TensorFlow scaler saved:")
print("tensorflow_scaler.pkl")


# ==========================================================
# SAVE METADATA
# ==========================================================

metadata = {

    "model": "TensorFlow Neural Network",

    "framework": "TensorFlow",

    "tensorflow_version": tf.__version__,

    "features": features,

    "forecast_horizon": "72 hours",

    "MAE": float(mae),

    "RMSE": float(rmse),

    "R2": float(r2),

    "epochs_trained": len(history.history["loss"]),

    "architecture": [
        "Dense(128, relu)",
        "Dropout(0.2)",
        "Dense(64, relu)",
        "Dropout(0.2)",
        "Dense(32, relu)",
        "Dense(1)"
    ]

}

joblib.dump(
    metadata,
    "tensorflow_model_metadata.pkl"
)


print()
print("TensorFlow metadata saved:")
print("tensorflow_model_metadata.pkl")


# ==========================================================
# FINAL
# ==========================================================

print()
print("=" * 60)
print("TENSORFLOW TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)