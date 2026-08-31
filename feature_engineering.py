
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "historical_aqi.csv"
OUTPUT_FILE = "features.csv"


# =========================================================
# LOAD HISTORICAL DATA
# =========================================================

print("=" * 60)
print("AQI FEATURE ENGINEERING PIPELINE")
print("=" * 60)

print("\nLoading historical AQI dataset...")

df = pd.read_csv(INPUT_FILE)

print("Raw dataset shape:", df.shape)


# =========================================================
# DATETIME
# =========================================================

df["datetime"] = pd.to_datetime(
    df["datetime"],
    utc=True
)

# Sort chronologically and remove duplicate timestamps
df = (
    df.sort_values("datetime")
      .drop_duplicates(
          subset=["datetime"],
          keep="last"
      )
      .reset_index(drop=True)
)


# =========================================================
# TIME-BASED FEATURES
# =========================================================

df["hour"] = df["datetime"].dt.hour

df["day"] = df["datetime"].dt.day

df["month"] = df["datetime"].dt.month

df["day_of_week"] = (
    df["datetime"].dt.dayofweek
)


# =========================================================
# AQI LAG FEATURES
# =========================================================

df["aqi_lag_1"] = (
    df["aqi"].shift(1)
)

df["aqi_lag_3"] = (
    df["aqi"].shift(3)
)

df["aqi_lag_6"] = (
    df["aqi"].shift(6)
)

df["aqi_lag_12"] = (
    df["aqi"].shift(12)
)

df["aqi_lag_24"] = (
    df["aqi"].shift(24)
)

df["aqi_lag_48"] = (
    df["aqi"].shift(48)
)

df["aqi_lag_72"] = (
    df["aqi"].shift(72)
)


# =========================================================
# POLLUTANT LAG FEATURES
# =========================================================

df["pm2_5_lag_1"] = (
    df["pm2_5"].shift(1)
)

df["pm10_lag_1"] = (
    df["pm10"].shift(1)
)


# =========================================================
# ROLLING AQI FEATURES
#
# shift(1) prevents current AQI from being used
# to calculate its own historical rolling features.
# =========================================================

previous_aqi = (
    df["aqi"].shift(1)
)

df["aqi_rolling_6"] = (
    previous_aqi
    .rolling(window=6)
    .mean()
)

df["aqi_rolling_24"] = (
    previous_aqi
    .rolling(window=24)
    .mean()
)

df["aqi_rolling_72"] = (
    previous_aqi
    .rolling(window=72)
    .mean()
)


# =========================================================
# AQI CHANGE
# =========================================================

df["aqi_change"] = (
    df["aqi"]
    -
    df["aqi_lag_1"]
)


# =========================================================
# TARGET
#
# The model predicts the AQI of the next hour.
#
# For historical/training rows:
#
# current AQI -> next-hour AQI
#
# The final/latest row naturally has no next-hour
# observation yet, so its target_aqi remains NaN.
# =========================================================

df["target_aqi"] = (
    df["aqi"].shift(-1)
)


# =========================================================
# REMOVE ROWS WITHOUT SUFFICIENT HISTORY
#
# IMPORTANT:
# We remove rows missing historical FEATURES,
# but we DO NOT remove the latest row merely because
# target_aqi is unavailable.
# =========================================================

feature_columns = [

    "datetime",

    "aqi",

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


df = (
    df
    .dropna(
        subset=feature_columns
    )
    .reset_index(drop=True)
)


# =========================================================
# VALIDATION
# =========================================================

required_columns = [

    "datetime",

    "aqi",

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

    "aqi_change",

    "target_aqi"
]


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


# =========================================================
# CHECK MISSING VALUES
#
# target_aqi is allowed to be missing ONLY for the
# latest observation because there is no next-hour
# actual AQI available yet.
# =========================================================

feature_missing = (
    df[feature_columns]
    .isnull()
    .sum()
    .sum()
)


if feature_missing > 0:

    raise ValueError(
        f"Feature dataset contains "
        f"{feature_missing} missing values."
    )


# Count missing target values
missing_targets = (
    df["target_aqi"]
    .isnull()
    .sum()
)


# Only the final row may have a missing target
if missing_targets > 1:

    raise ValueError(
        "More than one target_aqi value is missing. "
        "This indicates an unexpected data gap."
    )


# =========================================================
# VALIDATE DUPLICATES
# =========================================================

duplicate_datetimes = (
    df["datetime"]
    .duplicated()
    .sum()
)


if duplicate_datetimes > 0:

    raise ValueError(
        f"Found {duplicate_datetimes} duplicate datetimes."
    )


# =========================================================
# SAVE ENGINEERED DATASET
# =========================================================

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# =========================================================
# FINAL REPORT
# =========================================================

print("\n" + "=" * 60)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 60)

print(
    "\nFinal dataset shape:",
    df.shape
)

print(
    "\nTotal rows:",
    len(df)
)

print(
    "Total columns:",
    len(df.columns)
)

print(
    "\nColumns:"
)

for column in df.columns:

    print(
        " -",
        column
    )


print(
    "\nFirst 5 rows:"
)

print(
    df.head()
)


print(
    "\nMissing feature values:"
)

print(
    df[feature_columns]
    .isnull()
    .sum()
)


print(
    "\nMissing target_aqi values:",
    missing_targets
)


print(
    "\nLatest engineered observation:"
)

print(
    df.tail(1).to_string(
        index=False
    )
)


print(
    "\nDate range:"
)

print(
    df["datetime"].min(),
    "->",
    df["datetime"].max()
)


print(
    "\nTarget AQI distribution:"
)

print(
    df["target_aqi"]
    .dropna()
    .value_counts()
    .sort_index()
)


print(
    "\nSaved as:",
    OUTPUT_FILE
)

print("=" * 60)

