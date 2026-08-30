import os
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENWEATHER_API_KEY")

url = "https://api.openweathermap.org/data/2.5/air_pollution/history"

params = {
    "lat": 31.5204,
    "lon": 74.3587,
    "start": 1786387200,
    "end": 1786473600,
    "appid": api_key
}

response = requests.get(url, params=params)

print("Status code:", response.status_code)
print(response.json())