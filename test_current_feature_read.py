import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

print("=" * 60)
print("TESTING CURRENT HOPSWORKS FEATURE GROUP")
print("=" * 60)

api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    raise ValueError("HOPSWORKS_API_KEY not found.")

print("\nConnecting to Hopsworks...")

project = hopsworks.login(
    api_key_value=api_key
)

print("Login successful!")

fs = project.get_feature_store()

print("Feature Store connected!")

fg = fs.get_feature_group(
    name="aqi_features",
    version=1
)

print("Feature Group found!")
print("Name:", fg.name)
print("Version:", fg.version)
print("ID:", fg.id)

print("\nReading data...")

df = fg.select_all().read()

print("\nDATA READ SUCCESSFULLY!")
print("=" * 60)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())

print("=" * 60)
print("FEATURE GROUP TEST PASSED")
print("=" * 60)