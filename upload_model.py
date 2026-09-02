import os
import hopsworks
from dotenv import load_dotenv


# ==========================================================
# LOAD ENVIRONMENT
# ==========================================================

load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    raise ValueError(
        "HOPSWORKS_API_KEY not found in .env file."
    )


# ==========================================================
# CONNECT TO HOPSWORKS
# ==========================================================

print("=" * 60)
print("AQI MODEL REGISTRY UPLOAD")
print("=" * 60)

print("\nConnecting to Hopsworks...")

project = hopsworks.login(
    api_key_value=api_key
)

print("Hopsworks login successful.")


# ==========================================================
# MODEL REGISTRY
# ==========================================================

print("\nAccessing Model Registry...")

mr = project.get_model_registry()


# ==========================================================
# CREATE MODEL VERSION
# ==========================================================

print("\nCreating model version...")

model = mr.python.create_model(
    name="aqi_gradient_boosting",
    description=(
        "Gradient Boosting regression model for "
        "72-hour AQI forecasting."
    )
)


# ==========================================================
# UPLOAD MODEL
# ==========================================================

print("\nUploading aqi_model.pkl...")

model.save(
    "aqi_model.pkl"
)


# ==========================================================
# SUCCESS
# ==========================================================

print("\n" + "=" * 60)
print("MODEL REGISTRY UPLOAD SUCCESSFUL")
print("=" * 60)

print(
    "Model name:",
    model.name
)

print(
    "Model version:",
    model.version
)

print("=" * 60)
