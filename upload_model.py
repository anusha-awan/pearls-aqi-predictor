import hopsworks
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    raise ValueError(
        "HOPSWORKS_API_KEY not found in .env file."
    )

print("Connecting to Hopsworks...")

project = hopsworks.login(
    api_key_value=api_key
)

print("Connected successfully!")

# Get Model Registry
mr = project.get_model_registry()

print("Accessing Model Registry...")

# Create new model version
model = mr.python.create_model(
    name="aqi_random_forest",
    description="Random Forest model for next-hour AQI prediction"
)

print("Uploading aqi_model.pkl...")

model.save("aqi_model.pkl")

print("========================================")
print("MODEL UPLOAD SUCCESSFUL!")
print("========================================")
print("Model name:", model.name)
print("Model version:", model.version)