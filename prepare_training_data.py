import pandas as pd

# Load engineered dataset
df = pd.read_csv("features.csv")

# Convert datetime
df["datetime"] = pd.to_datetime(df["datetime"])

# Create target: next hour AQI
df["target_aqi"] = df["aqi"].shift(-1)

# Remove the final row because it has no next-hour target
df = df.dropna(subset=["target_aqi"]).reset_index(drop=True)

# Save training dataset
df.to_csv("training_data.csv", index=False)

print("Training data preparation completed!")

print("\nDataset shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nTarget column:")
print(df["target_aqi"].head())

print("\nMissing values:")
print(df.isnull().sum().sum())

print("\nSaved as: training_data.csv")