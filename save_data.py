import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENWEATHER_API_KEY")

url = "https://api.openweathermap.org/data/2.5/air_pollution"

params = {
    "lat": 31.5204,
    "lon": 74.3587,
    "appid": api_key
}

response = requests.get(url, params=params)

data = response.json()

pollution = data["list"][0]

row = {
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

df = pd.DataFrame([row])

print(df)

df.to_csv("aqi_data.csv", index=False)

print("\nData saved successfully!")