import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load datasets
calls = pd.read_csv(r'C:\Users\User\calls.csv')
audio_calls = pd.read_csv(r'C:\Users\User\audio_recording.csv')

# Convert 'arrtime' column to datetime with specific format
# Format: "day/month/year hour:minute" (e.g., "1/3/2024 0:17")
calls['arrtime'] = pd.to_datetime(calls['arrtime'], format='%d/%m/%Y %H:%M', errors='coerce')

# Create target variable (1 if audio recording exists, 0 otherwise)
calls['is_emergency'] = calls['iot_lora_id'].isin(audio_calls['iot_lora_id']).astype(int)

# Assumption 1: Multiple calls within 10-15 minutes from the same iot_lora_id
calls = calls.sort_values(by=['iot_lora_id', 'arrtime'])
calls['time_diff'] = calls.groupby('iot_lora_id')['arrtime'].diff().dt.total_seconds() / 60  # in minutes
calls['multiple_calls'] = ((calls['time_diff'] <= 15) & (calls['time_diff'] > 0)).astype(int)

# Assumption 2: Calls made during late-night or early-morning hours (12 AM to 6 AM)
calls['late_night_call'] = calls['arrtime'].dt.hour.apply(lambda x: 1 if 0 <= x < 6 else 0)

# Drop columns that won't be used directly for features
calls = calls.drop(columns=['arrtime', 'time_diff'])

# Select features and target
X = calls[['rssi', 'district', 'post_code', 'unit', 'multiple_calls', 'late_night_call']]
y = calls['is_emergency']

# Convert categorical features (e.g., district) to dummy variables
X = pd.get_dummies(X, drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the logistic regression model
model = LogisticRegression(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print evaluation results
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:\n', conf_matrix)
print('Classification Report:\n', report)
