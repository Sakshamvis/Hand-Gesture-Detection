import os
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ---------------- Paths ----------------
DATA_DIR = "../data"
MODEL_DIR = "../models"
MODEL_PATH = os.path.join(MODEL_DIR, "gesture_model.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)

# ---------------- Gesture Labels ----------------
LABELS = {
    "hello": 0,
    "iloveyou": 1,
    "yes": 2,
    "no": 3,
    "thankyou": 4,
    "please": 5
}

X = []
y = []

# ---------------- Load Data ----------------
for gesture, label in LABELS.items():
    file_path = os.path.join(DATA_DIR, f"{gesture}.csv")

    if not os.path.exists(file_path):
        print(f"Missing file: {file_path}")
        continue

    data = pd.read_csv(file_path, header=None)

    X.append(data.values)
    y.extend([label] * len(data))

X = np.vstack(X)
y = np.array(y)

print("Total samples:", len(X))

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------------- Train Model ----------------
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# ---------------- Evaluate ----------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# ---------------- Save Model ----------------
joblib.dump(model, MODEL_PATH)
print(f"Model saved at: {MODEL_PATH}")
