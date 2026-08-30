import pandas as pd

df = pd.read_csv("historical_aqi.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Datetimes:")
print(df["datetime"].duplicated().sum())

print("\nDate Range:")
print("Start:", df["datetime"].min())
print("End:", df["datetime"].max())

print("\nAQI Value Counts:")
print(df["aqi"].value_counts().sort_index())

print("\nData Types:")
print(df.dtypes)