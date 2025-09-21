# src/api/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional
import numpy as np
import cv2
import os
import joblib

from omr.preprocessing import preprocess_sheet
from omr.bubble_detection import (
    extract_bubbles, 
    detect_bubble_blocks, 
    generate_grid_coordinates, 
    clean_bubbles_with_ml
)
from omr.scoring import load_answer_key, score_sheet

# --- FastAPI app ---
app = FastAPI(title="OMR FastAPI Service")

# --- Paths (relative) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANSWER_KEY_PATH = os.path.join(BASE_DIR, "data", "answers", "ground_truth_setA.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "bubble_classifier.pkl")

# --- Load answer key safely ---
ANSWER_KEY = {}
try:
    ANSWER_KEY = load_answer_key(ANSWER_KEY_PATH)
    print("Answer key loaded successfully")
except Exception as e:
    print("WARNING: Could not load answer key:", e)

# --- Test route ---
@app.get("/")
def read_root():
    return {"message": "OMR API is running"}

# --- Core sheet processing ---
def process_sheet(file_stream, set_id: str = "A"):
    """
    Process uploaded OMR sheet file and return scores.
    """
    # Read image from file stream
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if img is None:
        return {"error": "Failed to read image. Please upload a valid OMR sheet."}

    # Preprocessing and bubble detection
    aligned = preprocess_sheet(img)
    bubbles = detect_bubble_blocks(aligned)
    bubbles_clean = clean_bubbles_with_ml(bubbles, aligned, model_path=MODEL_PATH)
    grid = generate_grid_coordinates(bubbles_clean)
    student_answers = extract_bubbles(aligned, grid)

    # DEBUG: check keys
    print("Student answers keys:", student_answers.keys())
    print("Answer key keys:", ANSWER_KEY.keys())

    if not ANSWER_KEY:
        return {"error": "Answer key missing or failed to load."}

    scores, total = score_sheet(student_answers, ANSWER_KEY)
    return {"scores": scores, "total": total, "set": set_id}

# --- Upload endpoint ---
@app.post("/upload/")
async def upload_sheet(file: UploadFile = File(...), set_id: Optional[str] = "A"):
    try:
        result = process_sheet(file.file, set_id.upper())
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# --- Allow calling process_sheet directly (for Streamlit) ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
