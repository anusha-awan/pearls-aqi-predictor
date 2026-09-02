
import pandas as pd


# =========================================================
# CONFIGURATION
# =========================================================

INPUT_FILE = "historical_aqi.csv"
OUTPUT_FILE = "features.csv"


# =========================================================
# EPA AQI CALCULATION FROM PM2.5
# =========================================================
#
# EPA PM2.5 AQI:
#
# 0.0   - 9.0     -> AQI 0-50
# 9.1   - 35.4    -> AQI 51-100
# 35.5  - 55.4    -> AQI 101-150
# 55.5  - 125.4   -> AQI 151-200
# 125.5 - 225.4   -> AQI 201-300
# 225.5 - 325.4   -> AQI 301-500
#
# Values above 325.4 are capped at AQI 500.
#
# PM2.5 concentration is truncated to one decimal place
# before AQI calculation, following EPA methodology.
# =========================================================

def calculate_pm25_aqi(pm25):

    if pd.isna(pm25):
        return None

    try:
        pm25 = float(pm25)

    except (ValueError, TypeError):
        return None

    if pm25 < 0:
        return None

    # EPA truncates PM2.5 concentration to one decimal
    pm25 = int(pm25 * 10) / 10

    # PM2.5 above the highest EPA breakpoint
    # is capped at AQI 500.
    if pm25 > 325.4:
        return 500

    breakpoints = [

        (0.0, 9.0, 0, 50),

        (9.1, 35.4, 51, 100),

        (35.5, 55.4, 101, 150),

        (55.5, 125.4, 151, 200),

        (125.5, 225.4, 201, 300),

        (225.5, 325.4, 301, 500)

    ]

    for (
        concentration_low,
        concentration_high,
        aqi_low,
        aqi_high
    ) in breakpoints:

        if (
            concentration_low
            <= pm25
            <= concentration_high
        ):

            aqi = (

                (
                    aqi_high
                    - aqi_low
                )
                /
                (
                    concentration_high
                    - concentration_low
                )

            ) * (

                pm25
                - concentration_low

            ) + aqi_low

            # EPA AQI is rounded to nearest integer
            return int(round(aqi))

    return None


# =========================================================
# START
# =========================================================

print("=" * 60)
print("AQI FEATURE ENGINEERING PIPELINE")
print("=" * 60)

print("\nLoading historical AQI dataset...")

df = pd.read_csv(INPUT_FILE)

print("Raw dataset shape:", df.shape)


# =========================================================
# REQUIRED RAW COLUMNS
# =========================================================

required_raw_columns = [

    "datetime",

    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "pm2_5",
    "pm10",
    "nh3"

]


missing_raw_columns = [

    column

    for column in required_raw_columns

    if column not in df.columns

]


if missing_raw_columns:

    raise ValueError(

        "Missing required raw columns:\n"
        +
        "\n".join(
            missing_raw_columns
        )

    )


# =========================================================
# DATETIME
# =========================================================

df["datetime"] = pd.to_datetime(

    df["datetime"],

    utc=True,

    errors="coerce"

)


invalid_datetime_count = (
    df["datetime"].isna().sum()
)


if invalid_datetime_count > 0:

    print(

        f"\nRemoving {invalid_datetime_count} "
        "rows with invalid datetime..."

    )

    df = df.dropna(
        subset=["datetime"]
    )


# Sort chronologically
df = (

    df

    .sort_values("datetime")

    .drop_duplicates(

        subset=["datetime"],

        keep="last"

    )

    .reset_index(drop=True)

)


# =========================================================
# NUMERIC CONVERSION
# =========================================================

numeric_columns = [

    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "pm2_5",
    "pm10",
    "nh3"

]


for column in numeric_columns:

    df[column] = pd.to_numeric(

        df[column],

        errors="coerce"

    )


# =========================================================
# PM2.5 VALIDATION
# =========================================================

missing_pm25 = (
    df["pm2_5"].isna().sum()
)

negative_pm25 = (
    (df["pm2_5"] < 0).sum()
)


print(
    "\nPM2.5 missing values:",
    missing_pm25
)

print(
    "PM2.5 negative values:",
    negative_pm25
)


# =========================================================
# REMOVE INVALID PM2.5 ROWS
# =========================================================

if missing_pm25 > 0:

    print(
        f"\nRemoving {missing_pm25} rows "
        "with missing PM2.5..."
    )

    df = (

        df

        .dropna(
            subset=["pm2_5"]
        )

        .reset_index(drop=True)

    )


if negative_pm25 > 0:

    print(
        f"\nRemoving {negative_pm25} rows "
        "with negative PM2.5..."
    )

    df = (

        df[
            df["pm2_5"] >= 0
        ]

        .reset_index(drop=True)

    )


# =========================================================
# EPA AQI
# =========================================================

print(
    "\nCalculating EPA-style AQI from PM2.5..."
)


df["aqi"] = (

    df["pm2_5"]

    .apply(
        calculate_pm25_aqi
    )

)


# =========================================================
# REMOVE INVALID AQI ROWS
# =========================================================

invalid_aqi_rows = (
    df["aqi"].isna().sum()
)


if invalid_aqi_rows > 0:

    print(

        f"\nRemoving {invalid_aqi_rows} rows "
        "where EPA AQI could not be calculated..."

    )

    df = (

        df

        .dropna(
            subset=["aqi"]
        )

        .reset_index(drop=True)

    )


# Make sure AQI is numeric
df["aqi"] = pd.to_numeric(
    df["aqi"],
    errors="coerce"
)


# =========================================================
# TIME FEATURES
# =========================================================

df["hour"] = (
    df["datetime"].dt.hour
)

df["day"] = (
    df["datetime"].dt.day
)

df["month"] = (
    df["datetime"].dt.month
)

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
# shift(1) ensures the current AQI is NOT included
# in its own historical rolling calculation.
# =========================================================

previous_aqi = (
    df["aqi"].shift(1)
)


df["aqi_rolling_6"] = (

    previous_aqi

    .rolling(
        window=6,
        min_periods=6
    )

    .mean()

)


df["aqi_rolling_24"] = (

    previous_aqi

    .rolling(
        window=24,
        min_periods=24
    )

    .mean()

)


df["aqi_rolling_72"] = (

    previous_aqi

    .rolling(
        window=72,
        min_periods=72
    )

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
# Predict the EPA AQI of the next observation.
#
# target_aqi = next-row AQI
# =========================================================

df["target_aqi"] = (
    df["aqi"].shift(-1)
)


# =========================================================
# FEATURE COLUMNS
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


# =========================================================
# REMOVE ROWS WITHOUT SUFFICIENT FEATURE HISTORY
#
# IMPORTANT:
# target_aqi is intentionally NOT included here.
#
# The latest row is allowed to have a missing target
# because there is no next observation yet.
# =========================================================

before_feature_cleaning = len(df)


df = (

    df

    .dropna(
        subset=feature_columns
    )

    .reset_index(drop=True)

)


removed_feature_rows = (

    before_feature_cleaning
    -
    len(df)

)


print(

    "\nRows removed because of insufficient "
    "historical feature data:",

    removed_feature_rows

)


# =========================================================
# TARGET VALIDATION
# =========================================================

missing_targets = (

    df["target_aqi"]

    .isna()

    .sum()

)


print(
    "Missing target_aqi values:",
    missing_targets
)


# =========================================================
# HANDLE MISSING TARGETS
#
# Normally only the final row should have a missing target.
#
# If there are unexpected missing targets in the middle,
# remove those rows because they cannot be used for
# supervised training.
# =========================================================

if missing_targets > 1:

    print(
        "\nWARNING: Multiple missing target_aqi values detected."
    )

    print(
        "Removing rows with missing targets."
    )

    df = (

        df

        .dropna(
            subset=["target_aqi"]
        )

        .reset_index(drop=True)

    )

    missing_targets = (
        df["target_aqi"].isna().sum()
    )


# =========================================================
# FEATURE MISSING VALUE CHECK
# =========================================================

feature_missing = (

    df[feature_columns]

    .isnull()

    .sum()

    .sum()

)


if feature_missing > 0:

    raise ValueError(

        f"Feature dataset still contains "
        f"{feature_missing} missing values."

    )


# =========================================================
# AQI RANGE CHECK
# =========================================================

aqi_values = (
    df["aqi"]
    .dropna()
)


target_values = (
    df["target_aqi"]
    .dropna()
)


aqi_min = (
    aqi_values.min()
)

aqi_max = (
    aqi_values.max()
)


target_min = (
    target_values.min()
)

target_max = (
    target_values.max()
)


print(
    "\nCalculated AQI range:",
    f"{aqi_min:.1f} -> {aqi_max:.1f}"
)


print(
    "Target AQI range:",
    f"{target_min:.1f} -> {target_max:.1f}"
)


# =========================================================
# STRICT AQI VALIDATION
# =========================================================

if (

    aqi_min < 0
    or
    aqi_max > 500

):

    raise ValueError(

        f"Current EPA AQI is outside valid "
        f"range 0-500: {aqi_min} -> {aqi_max}"

    )


if (

    target_min < 0
    or
    target_max > 500

):

    raise ValueError(

        f"Target EPA AQI is outside valid "
        f"range 0-500: {target_min} -> {target_max}"

    )


# =========================================================
# DUPLICATE DATETIME CHECK
# =========================================================

duplicate_datetimes = (

    df["datetime"]

    .duplicated()

    .sum()

)


if duplicate_datetimes > 0:

    raise ValueError(

        f"Found {duplicate_datetimes} "
        "duplicate datetimes."

    )


# =========================================================
# DATASET SIZE CHECK
# =========================================================

if len(df) < 100:

    raise ValueError(

        "Not enough data after feature engineering."

    )


# =========================================================
# SAVE
# =========================================================

df.to_csv(

    OUTPUT_FILE,

    index=False

)


# =========================================================
# FINAL REPORT
# =========================================================

print(
    "\n" + "=" * 60
)

print(
    "FEATURE ENGINEERING COMPLETED"
)

print(
    "=" * 60
)


print(
    "\nFinal dataset shape:",
    df.shape
)


print(
    "Total rows:",
    len(df)
)


print(
    "Total columns:",
    len(df.columns)
)


print(
    "\nEPA AQI range:"
)


print(
    f"{aqi_min:.1f} -> {aqi_max:.1f}"
)


print(
    "\nTarget EPA AQI range:"
)


print(
    f"{target_min:.1f} -> {target_max:.1f}"
)


print(
    "\nMissing feature values:",
    feature_missing
)


print(
    "Missing target_aqi values:",
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
    "\nTarget EPA AQI statistics:"
)


print(

    df["target_aqi"]

    .dropna()

    .describe()

)


print(
    "\nSaved as:",
    OUTPUT_FILE
)


print(
    "=" * 60
)

