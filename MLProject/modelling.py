import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sys
import os
import warnings

if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    # CLI parameters or default
    n_estimators = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    dataset_path = sys.argv[3] if len(sys.argv) > 3 else "fraud_preprocessing.csv"

    # Load preprocessed dataset
    df = pd.read_csv(dataset_path)

    # Get features and labels
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set MLflow tracking config
    # mlflow.set_tracking_uri(uri="http://127.0.0.1:5000/")
    mlflow.set_tracking_uri("file:../mlruns")
    
    # Run
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)

    # Explicit logging (in addition to autolog)
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=X_train[:5])
    accuracy = model.score(X_test, y_test)
    mlflow.log_metric("accuracy", accuracy)

    # print(f"Run ID: {run.info.run_id}")
