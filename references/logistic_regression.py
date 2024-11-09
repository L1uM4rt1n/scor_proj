import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, roc_curve, auc

class EmergencyProbabilityModel:
    def __init__(self):
        self.model = LogisticRegression()
        self.scaler = StandardScaler()
    
    def train(self, df, target_column):
        df["arrtime"] = pd.to_datetime(df["arrtime"])
        df["hour"] = df["arrtime"].dt.hour
        df["2_hour_interval"] = (df["hour"] // 2) * 2
        
        X = df[["rssi", "hour", "2_hour_interval"]]
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.model.fit(X_train_scaled, y_train)
        
        emergency_probabilities = self.model.predict_proba(X_test_scaled)[:, 1]

        roc_auc = roc_auc_score(y_test, emergency_probabilities)
        print(f"ROC AUC Score for {target_column}: {roc_auc}")
        
        df_test = X_test.copy()
        df_test["Pi"] = emergency_probabilities
        df_test["is_emergency"] = y_test
        
        return df_test, y_test

def main():
    df = pd.read_csv("DataSets/finalised_ifEmergency_dataset.csv")
    
    model = EmergencyProbabilityModel()
    print("========== new conditions ==========")
    df_with_probabilities, y_test = model.train(df, "is_emergency")
    df_with_probabilities["is_emergency_pred"] = (df_with_probabilities["Pi"] > 0.5).astype(int)
    
    print("\nModel Performance for is_emergency:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, df_with_probabilities["is_emergency_pred"]))
    print("\nClassification Report:")
    print(classification_report(y_test, df_with_probabilities["is_emergency_pred"]))
    
    fpr, tpr, _ = roc_curve(y_test, df_with_probabilities["Pi"])
    roc_auc = auc(fpr, tpr)
    
    plt.figure()
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression for new conditions")
    plt.legend(loc="lower right")
    plt.show()

    print("========== original conditions ==========")
    df_with_probabilities_2, y_test_2 = model.train(df, "is_emergency_2")
    df_with_probabilities_2["is_emergency_2_pred"] = (df_with_probabilities_2["Pi"] > 0.5).astype(int)
    
    print("\nModel Performance for is_emergency_2:")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_2, df_with_probabilities_2["is_emergency_2_pred"]))
    print("\nClassification Report:")
    print(classification_report(y_test_2, df_with_probabilities_2["is_emergency_2_pred"]))
    
    fpr_2, tpr_2, _ = roc_curve(y_test_2, df_with_probabilities_2["Pi"])
    roc_auc_2 = auc(fpr_2, tpr_2)
    
    plt.figure()
    plt.plot(fpr_2, tpr_2, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc_2)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Logistic Regression for original condition")
    plt.legend(loc="lower right")
    plt.show()

    combined_df = df_with_probabilities.merge(df_with_probabilities_2[['Pi', 'is_emergency_2_pred']], left_index=True, right_index=True, suffixes=('_is_emergency', '_is_emergency_2'))
    combined_df.to_csv("DataSets/combined_emergency_probabilities.csv", index=False)

if __name__ == "__main__":
    main()
