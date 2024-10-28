import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report

class EmergencyProbabilityModel:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
    
    def train(self, df):
        df['arrtime'] = pd.to_datetime(df['arrtime'])
        df['hour'] = df['arrtime'].dt.hour
        df['2_hour_interval'] = (df['hour'] // 2) * 2
        
        X = df[['rssi', 'hour', '2_hour_interval']]
        y = df['is_emergency']
        
        X_scaled = self.scaler.fit_transform(X)
        
        self.model.fit(X_scaled, y)
        
        emergency_probabilities = self.model.predict_proba(X_scaled)[:, 1]
        
        df['Pi'] = emergency_probabilities
        
        return df

def main():
    df = pd.read_csv("DataSets/finalised_dataset.csv")
    
    model = EmergencyProbabilityModel()
    df_with_probabilities = model.train(df)
    
    df_with_probabilities.to_csv("DataSets/emergency_probabilities.csv", index=False)
    
    y = df_with_probabilities['is_emergency']
    y_pred = (df_with_probabilities['Pi'] > 0.5).astype(int)
    
    print("\nModel Performance:")
    print("Confusion Matrix:")
    print(confusion_matrix(y, y_pred))
    print("\nClassification Report:")
    print(classification_report(y, y_pred))

if __name__ == "__main__":
    main()
