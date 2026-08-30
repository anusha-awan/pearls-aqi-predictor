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

# Sort chronologically
df = (
    df.sort_values("datetime")
      .drop_duplicates(
          subset=["datetime"]
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
# shift(1) ensures that the current AQI is NOT used
# to calculate its own input features.
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
# Predict the AQI of the NEXT HOUR.
#
# Example:
#
# Current AQI  → 3
# Next hour AQI → 4
#
# target_aqi = 4
# =========================================================

df["target_aqi"] = (
    df["aqi"].shift(-1)
)


# =========================================================
# REMOVE ROWS WITHOUT SUFFICIENT HISTORY
# =========================================================

df = (
    df.dropna()
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
# =========================================================

total_missing = (
    df[required_columns]
    .isnull()
    .sum()
    .sum()
)


if total_missing > 0:

    raise ValueError(
        f"Dataset still contains "
        f"{total_missing} missing values."
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
    "\nMissing values:"
)

print(
    df[required_columns]
    .isnull()
    .sum()
)


print(
    "\nTarget AQI distribution:"
)

print(
    df["target_aqi"]
    .value_counts()
    .sort_index()
)


print(
    "\nSaved as:",
    OUTPUT_FILE
)

print("=" * 60)