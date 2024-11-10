import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd
import os

class LogisticRegressionUtility:
    def __init__(self, model_path="models/logistic_model.joblib", scaler_path="models/scaler.joblib"):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.model = None
        self.scaler = None

    def train_and_save_model(self, df, target_column="is_emergency"):
        """Trains, evaluates, and saves the logistic regression model using new conditions."""
        # Prepare features based on new conditions
        df["arrtime"] = pd.to_datetime(df["arrtime"])
        df["hour"] = df["arrtime"].dt.hour
        df["two_hour"] = (df["hour"] // 2) * 2
        X = df[["rssi", "two_hour"]]
        y = df[target_column]
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Scaling features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train logistic regression model
        self.model = LogisticRegression()
        self.model.fit(X_train_scaled, y_train)
        
        # Save model and scaler
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        print("Model and scaler saved successfully.")

        # Model evaluation
        y_pred = self.model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        confusion = confusion_matrix(y_test, y_pred)
        classification = classification_report(y_test, y_pred)
        
        # ROC curve and AUC
        fpr, tpr, _ = roc_curve(y_test, self.model.predict_proba(X_test_scaled)[:, 1])
        roc_auc = auc(fpr, tpr)

        # Save ROC curve as JPEG
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve (area = %0.2f)" % roc_auc)
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Logistic Regression for New Conditions")
        plt.legend(loc="lower right")
        plt.savefig("results/logistic_regression_roc.jpeg", format="jpeg")
        plt.close()  # Close the plot to avoid display during automated runs
        
        # Save training log to a file
        with open("results/logistic_regression_log.txt", "w") as log_file:
            log_file.write("========== Training Log ==========\n")
            log_file.write(f"Accuracy: {accuracy:.4f}\n")
            log_file.write("Confusion Matrix:\n")
            log_file.write(f"{confusion}\n\n")
            log_file.write("Classification Report:\n")
            log_file.write(f"{classification}\n")
            log_file.write(f"ROC AUC: {roc_auc:.4f}\n")
        
        print("Training log saved to results/logistic_regression_log.txt")
        print("ROC curve saved as results/logistic_regression_roc.jpeg.")

    def load_model(self):
        """Loads a pre-trained model and scaler if they exist."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            print("Loaded pre-trained model and scaler.")
        else:
            raise FileNotFoundError("Pre-trained model or scaler not found. Train the model first.")

    def predict_probability(self, rssi, two_hour):
        """Predicts the probability of an emergency based on rssi and two_hour values."""
        if self.model is None or self.scaler is None:
            raise ValueError("Model and scaler must be loaded or trained before making predictions.")
        
        # Create a DataFrame from the individual inputs to match the expected input format
        input_df = pd.DataFrame([[rssi, two_hour]], columns=["rssi", "two_hour"])
        
        # Scale the input DataFrame
        X_scaled = self.scaler.transform(input_df)
        
        # Predict and return the probability of a genuine emergency for this single input
        return self.model.predict_proba(X_scaled)[0][1]

if __name__ == "__main__":
    df = pd.read_csv("data/finalised_ifEmergency_dataset.csv")
    logistic_util = LogisticRegressionUtility()
    logistic_util.train_and_save_model(df, target_column="is_emergency")
