import pandas as pd

# Load the CSV file (replace with actual path if running locally)
input_path = 'finalised_ifEmergency_dataset.csv'
output_path = 'final_calls_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(input_path)

# Drop columns 'unit', 'is_emergency', and 'is_emergency_2'
df = df.drop(columns=['unit', 'is_emergency', 'is_emergency_2', 'port'])

# Convert 'recording' column: 1 if there is a value, 0 if blank
df['recording'] = df['recording'].apply(lambda x: 1 if pd.notna(x) and str(x).strip() else 0)

# Save the modified DataFrame to a new CSV file
df.to_csv(output_path, index=False)