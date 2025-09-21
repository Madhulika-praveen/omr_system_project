import cv2

def is_marked(bubble_roi, threshold=0.5):
    total_pixels = bubble_roi.size
    dark_pixels = cv2.countNonZero(bubble_roi)
    fill_ratio = dark_pixels / total_pixels
    return fill_ratio > threshold

def extract_bubbles(sheet_image, grid_coordinates):
    detected = {}
    for subject, bubbles in grid_coordinates.items():
        detected[subject] = {}
        for q_no, (x, y, w, h) in enumerate(bubbles, start=1):
            roi = sheet_image[y:y+h, x:x+w]
            marked = is_marked(roi)
            detected[subject][q_no] = 'a' if marked else ' '
    return detected

def detect_bubble_blocks(sheet_img):
    gray = sheet_img if len(sheet_img.shape) == 2 else cv2.cvtColor(sheet_img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bubbles = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if 15 < w < 35 and 15 < h < 35:
            bubbles.append((x,y,w,h))
    bubbles = sorted(bubbles, key=lambda b: (b[1], b[0]))  # top-to-bottom, left-to-right
    return bubbles

def generate_grid_coordinates(bubbles):
    subjects = ["Python", "EDA", "SQL", "POWER BI", "Statistics"]
    bubbles_per_subject = 20*4  # 20 questions, 4 options each
    grid_coordinates = {}
    for i, subj in enumerate(subjects):
        start = i*bubbles_per_subject
        end = start+bubbles_per_subject
        grid_coordinates[subj] = bubbles[start:end]
    return grid_coordinates
