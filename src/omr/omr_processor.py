import cv2
import numpy as np
from .bubble_detection import detect_bubble_blocks, clean_bubbles_with_ml, generate_grid_coordinates, extract_bubbles
from .preprocessing import preprocess_sheet
from .scoring import load_answer_key, score_sheet

ANSWER_KEY_PATH = "E:/omr_system_project/data/answers/ground_truth_setA.csv"

try:
    ANSWER_KEY = load_answer_key(ANSWER_KEY_PATH)
except Exception as e:
    ANSWER_KEY = {}
    print("Warning: Could not load answer key:", e)

def process_sheet(file_stream, set_id="A"):
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    aligned = preprocess_sheet(img)
    bubbles = detect_bubble_blocks(aligned)
    bubbles_clean = clean_bubbles_with_ml(bubbles, aligned)
    grid = generate_grid_coordinates(bubbles_clean)
    student_answers = extract_bubbles(aligned, grid)

    if not ANSWER_KEY:
        return {"error": "Answer key missing or failed to load."}

    scores, total = score_sheet(student_answers, ANSWER_KEY)
    return {"scores": scores, "total": total, "set": set_id}
