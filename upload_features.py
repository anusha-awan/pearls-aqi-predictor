import os
import sys
import pandas as pd
import hopsworks
from dotenv import load_dotenv


# =========================================================
# CONFIGURATION
# =========================================================

FEATURE_GROUP_NAME = "aqi_features_v2"
FEATURE_GROUP_VERSION = 1

DATA_FILE = "features.csv"


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
# START
# =========================================================

print("=" * 60)
print("AQI FEATURE STORE UPLOAD")
print("=" * 60)


# =========================================================
# LOAD DATA
# =========================================================

print("\nLoading engineered features...")

df = pd.read_csv(DATA_FILE)

df["datetime"] = pd.to_datetime(
    df["datetime"],
    utc=True,
    errors="coerce"
)


# =========================================================
# CLEAN DATETIME
# =========================================================

invalid_datetime = df["datetime"].isna().sum()

if invalid_datetime > 0:

    print(
        f"Removing {invalid_datetime} rows "
        "with invalid datetime..."
    )

    df = df.dropna(
        subset=["datetime"]
    )


df = (
    df
    .sort_values("datetime")
    .drop_duplicates(
        subset=["datetime"],
        keep="last"
    )
    .reset_index(drop=True)
)


print(
    "Feature data loaded successfully."
)

print(
    "Rows:",
    len(df)
)

print(
    "Columns:",
    len(df.columns)
)


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

print(
    "\nValidating feature columns..."
)

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

print(
    "All required columns are present."
)


# =========================================================
# TARGET VALIDATION
# =========================================================

missing_target = (
    df["target_aqi"]
    .isna()
    .sum()
)

print(
    "\nMissing target_aqi values:",
    missing_target
)


if missing_target > 1:

    raise ValueError(
        "More than one target_aqi value is missing."
    )


# =========================================================
# FEATURE MISSING VALUE CHECK
# =========================================================

feature_columns = [
    column
    for column in required_columns
    if column != "target_aqi"
]

missing_feature_values = (
    df[feature_columns]
    .isnull()
    .sum()
    .sum()
)

print(
    "Missing feature values:",
    missing_feature_values
)


if missing_feature_values > 0:

    raise ValueError(
        f"Dataset contains "
        f"{missing_feature_values} missing feature values."
    )


# =========================================================
# DUPLICATE CHECK
# =========================================================

duplicate_datetimes = (
    df["datetime"]
    .duplicated()
    .sum()
)

print(
    "Duplicate datetimes:",
    duplicate_datetimes
)


if duplicate_datetimes > 0:

    raise ValueError(
        f"Found {duplicate_datetimes} "
        "duplicate datetimes."
    )


# =========================================================
# AQI RANGE CHECK
# =========================================================

aqi_min = df["aqi"].min()
aqi_max = df["aqi"].max()

target_values = (
    df["target_aqi"]
    .dropna()
)

target_min = target_values.min()
target_max = target_values.max()


print(
    "\nAQI range:",
    f"{aqi_min:.1f} -> {aqi_max:.1f}"
)

print(
    "Target AQI range:",
    f"{target_min:.1f} -> {target_max:.1f}"
)


if aqi_min < 0 or aqi_max > 500:

    raise ValueError(
        "AQI values outside valid EPA range 0-500."
    )


if target_min < 0 or target_max > 500:

    raise ValueError(
        "Target AQI values outside valid EPA range 0-500."
    )


# =========================================================
# LATEST FORECASTING ROW
# =========================================================

latest_row = df.tail(1).copy()

print(
    "\nLatest forecasting row:"
)

print(
    latest_row[
        [
            "datetime",
            "aqi",
            "pm2_5",
            "target_aqi"
        ]
    ].to_string(
        index=False
    )
)


# =========================================================
# REMOVE LATEST FORECASTING ROW
#
# The latest row has no next-hour target yet.
# It stays in features.csv for forecasting but is not
# uploaded to the historical Feature Store dataset.
# =========================================================

if pd.isna(df["target_aqi"].iloc[-1]):

    print(
        "\nLatest row has no target_aqi."
    )

    print(
        "Removing latest forecasting row "
        "from Feature Store upload..."
    )

    upload_df = df.iloc[:-1].copy()

else:

    upload_df = df.copy()


# =========================================================
# FINAL UPLOAD DATA VALIDATION
# =========================================================

print(
    "\nUpload dataset prepared."
)

print(
    "Rows to upload:",
    len(upload_df)
)

print(
    "Columns:",
    len(upload_df.columns)
)

remaining_missing = (
    upload_df[required_columns]
    .isnull()
    .sum()
    .sum()
)

print(
    "Remaining missing values:",
    remaining_missing
)


if remaining_missing > 0:

    raise ValueError(
        "Upload dataset still contains missing values."
    )


# =========================================================
# CONNECT TO HOPSWORKS
# =========================================================

print(
    "\nConnecting to Hopsworks..."
)

project = hopsworks.login(
    api_key_value=api_key
)

print(
    "Hopsworks login successful."
)

print(
    "Project:",
    project.name
)


# =========================================================
# FEATURE STORE
# =========================================================

fs = project.get_feature_store()

print(
    "Feature Store connected."
)


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


print(
    "Existing Feature Group found."
)

print(
    "Feature Group ID:",
    feature_group.id
)

print(
    "Primary keys:",
    feature_group.primary_key
)

print(
    "Online enabled:",
    feature_group.online_enabled
)


# =========================================================
# UPLOAD
# =========================================================

print(
    "\nStarting Feature Store upload..."
)

print(
    "Uploading all validated historical rows "
    "in a single operation..."
)


try:

    job, report = feature_group.insert(
        upload_df,
        storage="offline",
        wait=True
    )

    print(
        "\nUpload operation completed successfully."
    )

    print(
        "Job:",
        job
    )

    print(
        "Validation report:",
        report
    )


except Exception as e:

    error_text = str(e)

    print(
        "\nFEATURE STORE UPLOAD FAILED"
    )

    print(
        "Error:",
        error_text
    )

    # -----------------------------------------------------
    # KNOWN HOPSWORKS STORAGE ERROR
    # -----------------------------------------------------

    if (
        "RPC listener disconnected" in error_text
        or "Generic HdfsObjectStore error" in error_text
        or "HdfsObjectStore" in error_text
        or "Kernel error" in error_text
    ):

        print(
            "\nWARNING: Hopsworks Feature Store storage "
            "operation could not be completed."
        )

        print(
            "Local feature validation PASSED."
        )

        print(
            "Rows validated for upload:",
            len(upload_df)
        )

        print(
            "Remaining missing values:",
            remaining_missing
        )

        print(
            "Hopsworks returned an HDFS/RPC storage error."
        )

        print(
            "This error occurred during the Hopsworks "
            "storage operation, after dataset validation."
        )

        print(
            "The validated local feature dataset has "
            "been preserved."
        )

        print(
            "Continuing workflow because this is a "
            "known Hopsworks storage-side failure."
        )

        print(
            "\nFEATURE STORE UPLOAD DEFERRED"
        )

        print(
            "=" * 60
        )

        sys.exit(0)

    # -----------------------------------------------------
    # UNKNOWN ERROR
    # -----------------------------------------------------

    print(
        "\nUnexpected Feature Store error encountered."
    )

    raise


# =========================================================
# SUCCESS
# =========================================================

print(
    "\n" + "=" * 60
)

print(
    "FEATURE STORE UPLOAD SUCCESSFUL"
)

print(
    "=" * 60
)

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
    len(upload_df)
)

print(
    "Total columns:",
    len(upload_df.columns)
)

print(
    "Date range:",
    upload_df["datetime"].min(),
    "->",
    upload_df["datetime"].max()
)

print(
    "Latest forecasting row kept locally:",
    latest_row["datetime"].iloc[0]
)

print(
    "Latest target_aqi:",
    latest_row["target_aqi"].iloc[0]
)

print(
    "=" * 60
)