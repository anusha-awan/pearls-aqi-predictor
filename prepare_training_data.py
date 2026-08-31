import pandas as pd


# ==========================================================
# EPA AQI CALCULATION FROM PM2.5
# ==========================================================

def calculate_pm25_aqi(pm25):
    """
    Calculate US EPA-style AQI from PM2.5 concentration.

    PM2.5 breakpoints based on current EPA AQI breakpoints:
    0.0-9.0       -> AQI 0-50
    9.1-35.4      -> AQI 51-100
    35.5-55.4     -> AQI 101-150
    55.5-125.4    -> AQI 151-200
    125.5-225.4   -> AQI 201-300
    225.5-325.4   -> AQI 301-500
    """

    if pd.isna(pm25):
        return None

    pm25 = float(pm25)

    if pm25 < 0:
        return None

    breakpoints = [
        (0.0, 9.0, 0, 50),
        (9.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 125.4, 151, 200),
        (125.5, 225.4, 201, 300),
        (225.5, 325.4, 301, 500),
    ]

    # Values above 325.4 are capped at 500
    if pm25 > 325.4:
        pm25 = 325.4

    for c_low, c_high, i_low, i_high in breakpoints:

        if c_low <= pm25 <= c_high:

            aqi = (
                (i_high - i_low)
                / (c_high - c_low)
            ) * (pm25 - c_low) + i_low

            return round(aqi)

    return None


# ==========================================================
# LOAD ENGINEERED DATA
# ==========================================================

print("Loading features.csv...")

df = pd.read_csv("features.csv")

df["datetime"] = pd.to_datetime(df["datetime"])

df = (
    df
    .sort_values("datetime")
    .reset_index(drop=True)
)


print("Total rows:", len(df))


# ==========================================================
# CALCULATE EPA-STYLE AQI
# ==========================================================

print("\nCalculating EPA-style AQI from PM2.5...")

df["epa_aqi"] = df["pm2_5"].apply(
    calculate_pm25_aqi
)


# ==========================================================
# USE EPA AQI AS TARGET
# ==========================================================

df["target_aqi"] = df["epa_aqi"].shift(-1)


# Remove final row because it has no next-hour target
df = df.dropna(
    subset=["target_aqi"]
).reset_index(drop=True)


# ==========================================================
# SAVE
# ==========================================================

df.to_csv(
    "training_data.csv",
    index=False
)


# ==========================================================
# RESULTS
# ==========================================================

print("\nTraining data preparation completed!")

print("\nDataset shape:", df.shape)

print("\nAQI statistics:")
print(
    df["target_aqi"].describe()
)

print("\nSample AQI values:")
print(
    df[
        ["datetime", "pm2_5", "epa_aqi", "target_aqi"]
    ].head(10)
)

print("\nAQI range:")
print(
    "Minimum:",
    df["target_aqi"].min()
)

print(
    "Maximum:",
    df["target_aqi"].max()
)

print("\nSaved as: training_data.csv")