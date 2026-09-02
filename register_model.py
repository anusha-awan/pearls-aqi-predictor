import os
import joblib
import hopsworks
from dotenv import load_dotenv


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_FILE = "aqi_model.pkl"
MODEL_NAME = "pearls_aqi_predictor"

MODEL_VERSION = None


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

api_key = os.getenv("HOPSWORKS_API_KEY")

if not api_key:
    raise ValueError(
        "HOPSWORKS_API_KEY not found in .env file."
    )


# =========================================================
# START
# =========================================================

print("=" * 60)
print("PEARLS AQI MODEL REGISTRATION")
print("=" * 60)


# =========================================================
# VERIFY MODEL FILE
# =========================================================

print("\nLoading trained model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")
print("Model type:", type(model).__name__)


# =========================================================
# VERIFY MODEL PARAMETERS
# =========================================================

print("\nModel configuration:")

print(
    "n_estimators:",
    getattr(model, "n_estimators", "N/A")
)

print(
    "learning_rate:",
    getattr(model, "learning_rate", "N/A")
)

print(
    "max_depth:",
    getattr(model, "max_depth", "N/A")
)


# =========================================================
# CONNECT TO HOPSWORKS
# =========================================================

print("\nConnecting to Hopsworks...")

project = hopsworks.login(
    api_key_value=api_key
)

print("Hopsworks login successful.")
print("Project:", project.name)


# =========================================================
# MODEL REGISTRY
# =========================================================

print("\nConnecting to Model Registry...")

mr = project.get_model_registry()

print("Model Registry connected.")


# =========================================================
# CREATE MODEL
# =========================================================

print(
    "\nCreating model:",
    MODEL_NAME
)

model_metadata = mr.sklearn.create_model(
    name=MODEL_NAME,
    description=(
        "Pearls AQI Predictor using Gradient Boosting "
        "for next-hour EPA-style AQI prediction. "
        "The model is used to generate a 72-hour AQI forecast."
    ),
    metrics={
        "mae": 63.93,
        "rmse": 65.02,
        "r2": -4.0540,
        "mae_improvement_vs_persistence_percent": 46.14,
        "rmse_improvement_vs_persistence_percent": 46.74
    }
)


print("Model metadata created.")


# =========================================================
# SAVE MODEL TO REGISTRY
# =========================================================

print("\nUploading model to Hopsworks Model Registry...")

registered_model = model_metadata.save(
    MODEL_FILE
)


# =========================================================
# SUCCESS
# =========================================================

print("\n" + "=" * 60)
print("MODEL REGISTRATION SUCCESSFUL")
print("=" * 60)

print("Model name:", registered_model.name)
print("Model version:", registered_model.version)
print("Model ID:", registered_model.id)

print("=" * 60)