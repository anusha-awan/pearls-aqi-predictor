import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv


# =========================================================
# CONFIGURATION
# =========================================================

FEATURE_GROUP_NAME = "aqi_features_v2"
FEATURE_GROUP_VERSION = 1

DATA_FILE = "features.csv"

BATCH_SIZE = 500


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    raise ValueError(
        "HOPSWORKS_API_KEY not found in .env file."
    )


# =========================================================
# LOAD DATA
# =========================================================

print("=" * 60)
print("AQI FEATURE STORE UPLOAD")
print("=" * 60)

print("\nLoading engineered features...")

df = pd.read_csv(DATA_FILE)

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


print("Feature data loaded successfully.")
print("Rows:", len(df))
print("Columns:", len(df.columns))


# =========================================================
# REQUIRED COLUMNS
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


# =========================================================
# VALIDATE COLUMNS
# =========================================================

print("\nValidating feature columns...")

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing required columns:\n"
        + "\n".join(missing_columns)
    )

print("All required columns are present.")


# =========================================================
# VALIDATE MISSING VALUES
# =========================================================

missing_values = (
    df[required_columns]
    .isnull()
    .sum()
    .sum()
)

if missing_values > 0:

    raise ValueError(
        f"Dataset contains {missing_values} missing values."
    )

print("Missing values:", missing_values)


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

print("Duplicate datetimes:", duplicate_datetimes)


# =========================================================
# FINAL DATA CHECK
# =========================================================

print("\nDataset validation completed.")

print("Final rows:", len(df))

print("Final columns:", len(df.columns))

print(
    "Date range:",
    df["datetime"].min(),
    "->",
    df["datetime"].max()
)

print(
    "Target AQI values:",
    sorted(
        df["target_aqi"]
        .unique()
        .tolist()
    )
)


# =========================================================
# CONNECT TO HOPSWORKS
# =========================================================

print("\nConnecting to Hopsworks...")

project = hopsworks.login(
    api_key_value=api_key
)

print("Hopsworks login successful.")

print(
    "Project:",
    project.name
)


# =========================================================
# FEATURE STORE
# =========================================================

fs = project.get_feature_store()

print("Feature Store connected.")


# =========================================================
# GET EXISTING FEATURE GROUP
# =========================================================

print(
    f"\nLooking for Feature Group: "
    f"{FEATURE_GROUP_NAME} "
    f"v{FEATURE_GROUP_VERSION}"
)

feature_group = fs.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION
)

print("Existing Feature Group found.")

print(
    "Feature Group ID:",
    feature_group.id
)


# =========================================================
# UPLOAD IN BATCHES
# =========================================================

print("\nStarting batch upload...")

total_rows = len(df)

uploaded_rows = 0

batch_number = 0


for start in range(
    0,
    total_rows,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        total_rows
    )

    batch = df.iloc[
        start:end
    ].copy()

    batch_number += 1

    print(
        f"\nBatch {batch_number}: "
        f"rows {start + 1}-{end}"
    )

    try:

        job, report = feature_group.insert(
            batch,
            storage="offline",
            wait=True
        )

        uploaded_rows += len(batch)

        print(
            f"Batch {batch_number} uploaded successfully."
        )

        print(
            "Rows uploaded so far:",
            uploaded_rows,
            "/",
            total_rows
        )

    except Exception as e:

        print(
            f"\nBatch {batch_number} failed."
        )

        print(
            "Error:",
            str(e)
        )

        raise


# =========================================================
# SUCCESS
# =========================================================

print("\n" + "=" * 60)
print("FEATURE STORE UPLOAD SUCCESSFUL")
print("=" * 60)

print(
    "Feature Group:",
    FEATURE_GROUP_NAME
)

print(
    "Version:",
    FEATURE_GROUP_VERSION
)

print(
    "Total rows uploaded:",
    uploaded_rows
)

print(
    "Total columns:",
    len(df.columns)
)

print(
    "Date range:",
    df["datetime"].min(),
    "->",
    df["datetime"].max()
)

print("=" * 60)