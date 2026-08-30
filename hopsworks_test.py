import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

fs = project.get_feature_store()

print("HOPSWORKS FEATURE STORE OK")
print("Project:", project.name)
print("Feature Store:", fs)