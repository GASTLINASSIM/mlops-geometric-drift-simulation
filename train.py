import mlflow
import mlflow.sklearn
import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# --- 1. MLOPS CONFIGURATION ---
# Connect to the Local MinIO and MLflow
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9090"
os.environ["AWS_ACCESS_KEY_ID"] = "minio_user"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio_password"

DATA_DIR = "data"
IMG_SIZE = 64

# --- 2. DATA LOADING ---
def load_data():
    X = []
    y = []
    labels = ["square", "circle"]
    print("📥 Loading images...")

    if not os.path.exists(DATA_DIR):
        raise Exception("Data folder not found! Run generate_data.py first.")

    for label in labels:
        folder = os.path.join(DATA_DIR, label)
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = Image.open(img_path).resize((IMG_SIZE, IMG_SIZE))
            img_vector = np.array(img).flatten()
            X.append(img_vector)
            y.append(0 if label == "square" else 1)

    return np.array(X), np.array(y)

# --- 3. TRAINING PIPELINE ---
if __name__ == "__main__":
    # Load Data
    X, y = load_data()

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Set MLflow Experiment
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Geometric_Shapes_Classification")

    print("🚀 Starting training.")
    with mlflow.start_run():
        # Hyperparameters
        n_estimators = 50

        # Train
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"🎯 Accuracy: {acc:.2f}")

        # --- LOGGING TO MLFLOW ---
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("dataset_size", len(X))
        mlflow.log_metric("accuracy", acc)

        # Log Model (Uploads to MinIO)
        mlflow.sklearn.log_model(clf, "model")
        print("✅ Model saved to MLflow Registry (MinIO)!")
