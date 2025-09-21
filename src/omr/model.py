import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib  # for saving the trained model

# ---------------- Bubble Detection Functions ----------------
def detect_bubble_blocks(sheet_img):
    h, w = sheet_img.shape[:2]
    gray = sheet_img if len(sheet_img.shape) == 2 else cv2.cvtColor(sheet_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    kernel = np.ones((2,2), np.uint8)
    clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    clean = cv2.dilate(clean, kernel, iterations=1)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    left_margin, right_margin = int(w*0.02), int(w*0.97)
    top_margin, bottom_margin = int(h*0.2), int(h*0.98)

    bubbles = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        aspect = bw / bh
        area = bw * bh
        if 12 < bw < 55 and 12 < bh < 55 and 0.55 < aspect < 1.5 and area > 200:
            if left_margin < x < right_margin and top_margin < y < bottom_margin:
                bubbles.append((x, y, bw, bh))
    return sorted(bubbles, key=lambda b: (b[1], b[0]))

# # ---------------- Dataset Labeling ----------------
# def create_dataset(img_path):
#     img = cv2.imread(img_path)
#     if img is None:
#         raise ValueError("Cannot read image at " + img_path)

#     # Detect bubbles
#     bubbles = detect_bubble_blocks(img)
#     os.makedirs("dataset/positive", exist_ok=True)
#     os.makedirs("dataset/negative", exist_ok=True)

#     count_pos = 0
#     count_neg = 0
#     for x, y, w, h in bubbles:
#         roi = img[y:y+h, x:x+w]
#         roi_resized = cv2.resize(roi, (28,28))
#         key = cv2.waitKey(0) & 0xFF
#         if key == ord('y'):
#             cv2.imwrite(f"dataset/positive/bubble_{count_pos}.png", roi_resized)
#             count_pos += 1
#         elif key == ord('n'):
#             cv2.imwrite(f"dataset/negative/noise_{count_neg}.png", roi_resized)
#             count_neg += 1
#     print("Dataset labeling complete!")

# ---------------- Prepare Data for ML ----------------
def load_dataset():
    X, y = [], []
    for label, folder in enumerate(["positive", "negative"]):
        folder_path = f"dataset/{folder}"
        for file in os.listdir(folder_path):
            img = cv2.imread(os.path.join(folder_path, file), cv2.IMREAD_GRAYSCALE)
            img_flat = img.flatten() / 255.0  # normalize
            X.append(img_flat)
            y.append(label)
    return np.array(X), np.array(y)

# ---------------- Train Model ----------------
def train_model():
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Test Accuracy:", accuracy_score(y_test, y_pred))
    joblib.dump(model, "bubble_classifier.pkl")
    print("Model saved as bubble_classifier.pkl")

# ---------------- Predict / Clean Bubbles ----------------
def clean_bubbles_with_ml(bubbles, sheet_img, model_path="bubble_classifier.pkl"):
    model = joblib.load(model_path)
    cleaned_bubbles = []
    for x, y, w, h in bubbles:
        roi = sheet_img[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_flat = cv2.resize(roi_gray, (28,28)).flatten() / 255.0
        pred = model.predict([roi_flat])[0]
        if pred == 1:  # 1 = valid bubble
            cleaned_bubbles.append((x, y, w, h))
    return cleaned_bubbles

# ---------------- Main ----------------
if __name__ == "__main__":
    img_path = "data/images/setA/Img8.jpeg"  # change as needed

    # Step 1: Label dataset
    create_dataset(img_path)

    # Step 2: Train ML model
    train_model()

    # Step 3: Test cleaning
    img = cv2.imread(img_path)
    bubbles = detect_bubble_blocks(img)
    cleaned = clean_bubbles_with_ml(bubbles, img)
    print(f"Original bubbles: {len(bubbles)}, Cleaned bubbles: {len(cleaned)}")
