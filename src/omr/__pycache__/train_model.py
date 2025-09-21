# train_model.py

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# ---------------- Paths ----------------
POS_PATH = "dataset/positive"
NEG_PATH = "dataset/negative"
MODEL_PATH = "models/bubble_classifier.pkl"

# ---------------- Load Dataset ----------------
X = []
y = []

print("[INFO] Loading dataset...")

# Positive samples
for file in os.listdir(POS_PATH):
    img_path = os.path.join(POS_PATH, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (28, 28))
    X.append(img.flatten())
    y.append(1)  # 1 = valid bubble

# Negative samples
for file in os.listdir(NEG_PATH):
    img_path = os.path.join(NEG_PATH, file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue
    img = cv2.resize(img, (28, 28))
    X.append(img.flatten())
    y.append(0)  # 0 = noise

X = np.array(X)
y = np.array(y)

print(f"[INFO] Dataset loaded: {len(X)} samples")

# ---------------- Split Dataset ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"[INFO] Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# ---------------- Train Model ----------------
print("[INFO] Training SVM classifier...")
clf = SVC(kernel='linear', probability=True)
clf.fit(X_train, y_train)

# ---------------- Evaluate Model ----------------
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"[INFO] Test Accuracy: {acc*100:.2f}%")

# ---------------- Save Model ----------------
os.makedirs("models", exist_ok=True)
joblib.dump(clf, MODEL_PATH)
print(f"[INFO] Model saved at {MODEL_PATH}")

# ---------------- Optional: Test on a single ROI ----------------
# roi_test = cv2.imread("dataset/positive/bubble_0.png", cv2.IMREAD_GRAYSCALE)
# roi_test = cv2.resize(roi_test, (28,28)).flatten()
# print("Prediction:", clf.predict([roi_test])[0])
