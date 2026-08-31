import os
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv


# =========================================================
# CONFIGURATION
# =========================================================

HISTORICAL_FILE = "historical_aqi.csv"
CURRENT_FILE = "aqi_data.csv"

LATITUDE = 31.5204
LONGITUDE = 74.3587

API_URL = "https://api.openweathermap.org/data/2.5/air_pollution"


# =========================================================
# LOAD API KEY
# =========================================================

load_dotenv()

api_key = os.getenv("OPENWEATHER_API_KEY")

if not api_key:
    raise ValueError(
        "OPENWEATHER_API_KEY not found in .env file."
    )


# =========================================================
# FETCH CURRENT AIR QUALITY
# =========================================================

print("=" * 60)
print("AQI DATA COLLECTION")
print("=" * 60)

print("\nFetching latest air-quality data...")

params = {
    "lat": LATITUDE,
    "lon": LONGITUDE,
    "appid": api_key
}

response = requests.get(
    API_URL,
    params=params,
    timeout=30
)

response.raise_for_status()

data = response.json()

pollution = data["list"][0]


# =========================================================
# CREATE RECORD
# =========================================================

timestamp = datetime.fromtimestamp(
    pollution["dt"],
    tz=timezone.utc
)

row = {
    "datetime": timestamp,
    "aqi": pollution["main"]["aqi"],
    "co": pollution["components"]["co"],
    "no": pollution["components"]["no"],
    "no2": pollution["components"]["no2"],
    "o3": pollution["components"]["o3"],
    "so2": pollution["components"]["so2"],
    "pm2_5": pollution["components"]["pm2_5"],
    "pm10": pollution["components"]["pm10"],
    "nh3": pollution["components"]["nh3"]
}

new_data = pd.DataFrame([row])


# =========================================================
# SAVE CURRENT DATA
# =========================================================

new_data.to_csv(
    CURRENT_FILE,
    index=False
)

print("\nLatest observation:")
print(new_data.to_string(index=False))

print(
    f"\nCurrent data saved to {CURRENT_FILE}"
)


# =========================================================
# UPDATE HISTORICAL DATASET
# =========================================================

print(
    f"\nUpdating {HISTORICAL_FILE}..."
)

if os.path.exists(HISTORICAL_FILE):

    historical = pd.read_csv(
        HISTORICAL_FILE
    )

else:

    historical = pd.DataFrame(
        columns=new_data.columns
    )


# Make sure datetime formats match
historical["datetime"] = pd.to_datetime(
    historical["datetime"],
    utc=True
)

new_data["datetime"] = pd.to_datetime(
    new_data["datetime"],
    utc=True
)


# Append latest observation
historical = pd.concat(
    [
        historical,
        new_data
    ],
    ignore_index=True
)


# Remove duplicate timestamps
historical = (
    historical
    .drop_duplicates(
        subset=["datetime"],
        keep="last"
    )
    .sort_values("datetime")
    .reset_index(drop=True)
)


# Save updated historical data
historical.to_csv(
    HISTORICAL_FILE,
    index=False
)


# =========================================================
# FINAL REPORT
# =========================================================

print("\n" + "=" * 60)
print("AQI DATA COLLECTION COMPLETED")
print("=" * 60)

print(
    "Latest timestamp:",
    historical["datetime"].max()
)

print(
    "Total historical records:",
    len(historical)
)

print(
    "Date range:",
    historical["datetime"].min(),
    "->",
    historical["datetime"].max()
)

print(
    f"\nUpdated file: {HISTORICAL_FILE}"
)

print("=" * 60)