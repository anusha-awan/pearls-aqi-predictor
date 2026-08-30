import pandas as pd
import joblib
import shap

print("=" * 60)
print("SHAP FEATURE IMPORTANCE ANALYSIS")
print("=" * 60)

# Load training data
print("\nLoading training data...")
df = pd.read_csv("training_data.csv")

# Same features used by the trained model
features = [
    "co",
    "no",
    "no2",
    "o3",
    "so2",
    "pm2_5",
    "pm10",
    "nh3",
    "hour",
    "day",
    "month",
    "day_of_week",
    "aqi",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_lag_48",
    "aqi_lag_72",
    "pm2_5_lag_1",
    "pm10_lag_1",
    "aqi_rolling_6",
    "aqi_rolling_24",
    "aqi_rolling_72",
    "aqi_change"
]

# Remove missing values
df = df.dropna(
    subset=features + ["target_aqi"]
).reset_index(drop=True)

X = df[features]

print("Rows available:", len(X))
print("Features:", len(features))

# Load existing trained model
print("\nLoading existing AQI model...")
model = joblib.load("aqi_model.pkl")

print("Model loaded successfully.")

# Handle Pipeline models
if hasattr(model, "named_steps"):
    predictor = model.named_steps["model"]
else:
    predictor = model

# SHAP analysis
print("\nCalculating SHAP values...")

explainer = shap.TreeExplainer(predictor)

# Use a manageable sample
X_sample = X.sample(
    min(1000, len(X)),
    random_state=42
)

shap_values = explainer.shap_values(X_sample)

# Mean absolute SHAP importance
importance = pd.DataFrame({
    "feature": features,
    "mean_abs_shap": abs(shap_values).mean(axis=0)
})

importance = importance.sort_values(
    "mean_abs_shap",
    ascending=False
).reset_index(drop=True)

print("\n" + "=" * 60)
print("TOP FEATURE IMPORTANCE")
print("=" * 60)

print(
    importance.head(15).to_string(index=False)
)

# Save results
importance.to_csv(
    "shap_feature_importance.csv",
    index=False
)

print("\nSHAP results saved:")
print("shap_feature_importance.csv")

# Create SHAP bar plot
print("\nGenerating SHAP plot...")

shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    show=False
)

import matplotlib.pyplot as plt

plt.tight_layout()
plt.savefig(
    "shap_feature_importance.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

print("SHAP plot saved:")
print("shap_feature_importance.png")

print("\n" + "=" * 60)
print("SHAP ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 60)