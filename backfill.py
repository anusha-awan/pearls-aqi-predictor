import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

load_dotenv()

api_key = os.getenv("OPENWEATHER_API_KEY")

url = "https://api.openweathermap.org/data/2.5/air_pollution/history"

lat = 31.5204
lon = 74.3587

# Start and end dates
start_date = datetime(2025, 8, 1, tzinfo=timezone.utc)
end_date = datetime(2026, 8, 1, tzinfo=timezone.utc)
all_rows = []

current_date = start_date

while current_date < end_date:

    next_date = current_date + timedelta(days=1)

    start_timestamp = int(current_date.timestamp())
    end_timestamp = int(next_date.timestamp())

    params = {
        "lat": lat,
        "lon": lon,
        "start": start_timestamp,
        "end": end_timestamp,
        "appid": api_key
    }

    response = requests.get(url, params=params)

    print(
        "Fetching:",
        current_date.strftime("%Y-%m-%d"),
        "| Status:",
        response.status_code
    )

    if response.status_code == 200:

        data = response.json()

        for item in data["list"]:

            components = item["components"]

            row = {
                "datetime": datetime.fromtimestamp(
                    item["dt"],
                    tz=timezone.utc
                ),
                "aqi": item["main"]["aqi"],
                "co": components["co"],
                "no": components["no"],
                "no2": components["no2"],
                "o3": components["o3"],
                "so2": components["so2"],
                "pm2_5": components["pm2_5"],
                "pm10": components["pm10"],
                "nh3": components["nh3"]
            }

            all_rows.append(row)

    else:
        print("Error:", response.text)

    current_date = next_date

# Create dataframe
df = pd.DataFrame(all_rows)

# Remove duplicate timestamps
df = df.drop_duplicates(subset=["datetime"])

# Sort by time
df = df.sort_values("datetime")

print("\nFinal Dataset:")
print(df)

# Save dataset
df.to_csv("historical_aqi.csv", index=False)

print("\nHistorical data saved successfully!")
print("Number of rows:", len(df))