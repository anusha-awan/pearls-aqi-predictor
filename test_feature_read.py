import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")

print("Connecting to Hopsworks...")

project = hopsworks.login(
    api_key_value=api_key
)

print("Login successful!")

fs = project.get_feature_store()

print("Feature Store connected!")

fg = fs.get_feature_group(
    name="aqi_features_v2",
    version=1
)

print("Feature Group found!")
print("Feature Group ID:", fg.id)

print("\nReading data...")

df = fg.select_all().read()

print("\nREAD SUCCESSFUL!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nFirst 5 rows:")
print(df.head())