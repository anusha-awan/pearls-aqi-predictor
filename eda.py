import pandas as pd
import matplotlib.pyplot as plt


# Load dataset
df = pd.read_csv("training_data.csv")

# Convert datetime
df["datetime"] = pd.to_datetime(df["datetime"])

print("Dataset Shape:", df.shape)

print("\nBasic Statistics:")
print(df.describe())

print("\nAQI Distribution:")
print(df["aqi"].value_counts().sort_index())


# -----------------------------------
# 1. AQI Over Time
# -----------------------------------

plt.figure(figsize=(14, 5))
plt.plot(df["datetime"], df["aqi"])
plt.title("AQI Over Time")
plt.xlabel("Date")
plt.ylabel("AQI")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("aqi_over_time.png")
plt.show()


# -----------------------------------
# 2. AQI Distribution
# -----------------------------------

plt.figure(figsize=(8, 5))
df["aqi"].value_counts().sort_index().plot(kind="bar")
plt.title("AQI Distribution")
plt.xlabel("AQI Level")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("aqi_distribution.png")
plt.show()


# -----------------------------------
# 3. AQI by Hour
# -----------------------------------

hourly_aqi = df.groupby("hour")["aqi"].mean()

plt.figure(figsize=(10, 5))
hourly_aqi.plot()
plt.title("Average AQI by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average AQI")
plt.xticks(range(24))
plt.tight_layout()
plt.savefig("aqi_by_hour.png")
plt.show()


# -----------------------------------
# 4. AQI by Day of Week
# -----------------------------------

daily_aqi = df.groupby("day_of_week")["aqi"].mean()

plt.figure(figsize=(10, 5))
daily_aqi.plot(kind="bar")
plt.title("Average AQI by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Average AQI")
plt.tight_layout()
plt.savefig("aqi_by_day_of_week.png")
plt.show()


print("\nEDA completed!")
print("Charts saved successfully.")