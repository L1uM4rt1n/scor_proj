print ("========== EDA of dataset ==========")
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

calls_df = pd.read_csv("DataSets/calls.csv")
audio_df = pd.read_csv("DataSets/audio_recording.csv")

combined_df = pd.merge(calls_df, audio_df, on=["iot_lora_id", "port", "arrtime", "rssi", "district", "post_code", "unit"], how="outer")
combined_df.drop_duplicates(inplace=True)
numeric_df = combined_df.select_dtypes(include=[float, int])

# correlation matrix
corr_matrix = numeric_df.corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# histogram for numerical features
combined_df[["rssi"]].hist(bins=30, figsize=(10, 5))
plt.show()

# categorical features
plt.figure(figsize=(12, 6))
sns.countplot(x="district", data=combined_df, order=combined_df["district"].value_counts().index)
plt.title("Count of Calls per District")
plt.xlabel("District")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.show()

"""
assumption conditions for determining if it is an emergency
Condition 1: Multiple calls within 10-15 minutes from the same iot_lora_id
Condition 2: Calls made during late-night or early-morning hours (e.g., 12 AM to 6 AM)
"""

combined_df["arrtime"] = pd.to_datetime(combined_df["arrtime"])

combined_df["is_emergency"] = 0

# Condition 1
combined_df = combined_df.sort_values(by=["iot_lora_id", "arrtime"])
combined_df["time_diff"] = combined_df.groupby("iot_lora_id")["arrtime"].diff().dt.total_seconds() / 60
combined_df.loc[(combined_df["time_diff"] >= 10) & (combined_df["time_diff"] <= 15), "is_emergency"] = 1

# Condition 2
combined_df.loc[combined_df["arrtime"].dt.hour.isin(range(0, 6)), "is_emergency"] = 1

"""
original condition for determining if it is an emergency
Condition: if there is an audio recording
"""

# original condition
combined_df["is_emergency_2"] = combined_df["recording"].apply(lambda x: 1 if pd.notnull(x) else 0)

combined_df.drop(columns=["time_diff"], inplace=True)
combined_df.to_csv("DataSets/finalised_ifEmergency_dataset.csv", index=False)

print(combined_df.info())
print(combined_df.describe())
print(combined_df.isnull().sum())
