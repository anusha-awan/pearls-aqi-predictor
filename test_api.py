import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENWEATHER_API_KEY")
print("API key loaded:", bool(api_key))
url = "https://api.openweathermap.org/data/2.5/air_pollution"

params = {
    "lat": 31.5204,
    "lon": 74.3587,
    "appid": api_key
}

response = requests.get(url, params=params)

print("Status code:", response.status_code)

data = response.json()

print(data)