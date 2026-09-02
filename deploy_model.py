import os
import hopsworks
from dotenv import load_dotenv


# =========================================================
# CONFIGURATION
# =========================================================

MODEL_NAME = "pearls_aqi_predictor"
MODEL_VERSION = 1

DEPLOYMENT_NAME = "pearlsaqipredictordeployment"


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
print("PEARLS AQI MODEL DEPLOYMENT")
print("=" * 60)


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
# GET REGISTERED MODEL
# =========================================================

print(
    f"\nLooking for model: "
    f"{MODEL_NAME} v{MODEL_VERSION}"
)

model = mr.get_model(
    MODEL_NAME,
    version=MODEL_VERSION
)

print("Registered model found.")

print("Model name:", model.name)
print("Model version:", model.version)


# =========================================================
# MODEL SERVING
# =========================================================

print("\nConnecting to Model Serving...")

ms = project.get_model_serving()

print("Model Serving connected.")


# =========================================================
# CHECK EXISTING DEPLOYMENT
# =========================================================

print(
    f"\nChecking for existing deployment: "
    f"{DEPLOYMENT_NAME}"
)

existing = ms.get_deployment(
    DEPLOYMENT_NAME
)

if existing is not None:

    print(
        "Deployment already exists."
    )

    print(
        "Deployment:",
        DEPLOYMENT_NAME
    )

    print(
        "\nStopping here to avoid creating "
        "a duplicate deployment."
    )

else:

    # =====================================================
    # CREATE PREDICTOR
    # =====================================================

    print("\nCreating model predictor...")

    predictor = ms.create_predictor(
        model,
        name=DEPLOYMENT_NAME
    )

    print("Predictor created.")


    # =====================================================
    # DEPLOY
    # =====================================================

    print("\nDeploying model...")

    deployment = predictor.deploy()

    print("\n" + "=" * 60)
    print("MODEL DEPLOYMENT SUCCESSFUL")
    print("=" * 60)

    print(
        "Deployment name:",
        DEPLOYMENT_NAME
    )

    print(
        "Model:",
        MODEL_NAME
    )

    print(
        "Model version:",
        MODEL_VERSION
    )

    print(
        "Deployment URL:",
        deployment.get_endpoint_url()
    )

    print("=" * 60)


print("\nDeployment step completed.")