import cv2
import numpy as np
import os
import joblib

# Base directory of the project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "bubble_classifier.pkl")

def clean_bubbles_with_ml(bubbles, sheet_img, model_path=MODEL_PATH):
    """
    Use trained ML model to remove noise bubbles.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Bubble classifier model not found at {model_path}")
    
    model = joblib.load(model_path)
    cleaned_bubbles = []

    for x, y, w, h in bubbles:
        roi = sheet_img[y:y+h, x:x+w]
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        roi_resized = cv2.resize(roi_gray, (28, 28)).flatten()  # same as training
        prediction = model.predict([roi_resized])
        if prediction[0] == 1:  # 1 = valid bubble
            cleaned_bubbles.append((x, y, w, h))
    
    return cleaned_bubbles


def is_marked(bubble_roi, threshold=0.5):
    """Check if a bubble ROI is filled."""
    if len(bubble_roi.shape) == 3:
        bubble_roi = cv2.cvtColor(bubble_roi, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(bubble_roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    fill_ratio = cv2.countNonZero(inv) / inv.size
    return fill_ratio > threshold


def extract_bubbles(sheet_image, grid_coordinates):
    detected = {}
    options = ['a', 'b', 'c', 'd']
    start_q = {"Python": 1, "EDA": 21, "SQL": 41, "POWER BI": 61, "Statistics": 81}

    for subject, bubbles in grid_coordinates.items():
        detected[subject] = {}
        q_start = start_q[subject]
        for q_index, start_idx in enumerate(range(0, len(bubbles), 4)):
            q_no = q_start + q_index
            answer = ' '
            for opt_idx in range(4):
                try:
                    x, y, w, h = bubbles[start_idx + opt_idx]
                    roi = sheet_image[y:y+h, x:x+w]
                    if is_marked(roi):
                        answer = options[opt_idx]
                        break
                except IndexError:
                    pass
            detected[subject][q_no] = answer
    return detected


def detect_bubble_blocks(sheet_img):
    h, w = sheet_img.shape[:2]
    gray = sheet_img if len(sheet_img.shape) == 2 else cv2.cvtColor(sheet_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Morphological cleaning
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


def generate_grid_coordinates(bubbles, questions_per_subject=20):
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Statistics"]
    start_q = [1, 21, 41, 61, 81]

    if len(bubbles) == 0:
        return {subj: [] for subj in subjects}

    y_sorted = sorted(bubbles, key=lambda b: b[1])
    row_groups, current_row = [], []
    row_thresh = 10

    for bubble in y_sorted:
        if not current_row:
            current_row.append(bubble)
        elif abs(bubble[1] - current_row[0][1]) <= row_thresh:
            current_row.append(bubble)
        else:
            row_groups.append(sorted(current_row, key=lambda b: b[0]))
            current_row = [bubble]
    if current_row:
        row_groups.append(sorted(current_row, key=lambda b: b[0]))

    grid_coordinates = {}
    idx = 0
    for subj_idx, subject in enumerate(subjects):
        subject_bubbles = []
        for q_index in range(questions_per_subject):
            if idx < len(row_groups):
                row = row_groups[idx]
                subject_bubbles.extend(row)
                idx += 1
        grid_coordinates[subject] = subject_bubbles

    return grid_coordinates
