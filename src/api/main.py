from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional
import numpy as np
import cv2

from omr.preprocessing import preprocess_sheet
from omr.bubble_detection import (
    extract_bubbles, 
    detect_bubble_blocks, 
    generate_grid_coordinates, 
    clean_bubbles_with_ml
)
from omr.scoring import load_answer_key, score_sheet

app = FastAPI(title="OMR FastAPI Service")

# Safe answer key loading
ANSWER_KEY = {}
try:
    ANSWER_KEY = load_answer_key("E:/omr_system_project/data/answers/ground_truth_setA.csv")
    print("Answer key loaded successfully")
except Exception as e:
    print("WARNING: Could not load answer key:", e)

# --- Test route ---
@app.get("/")
def read_root():
    return {"message": "OMR API is running"}

# --- Core sheet processing ---
def process_sheet(file_stream, set_id: str = "A"):
    file_bytes = np.frombuffer(file_stream.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    aligned = preprocess_sheet(img)  # your deskewed/aligned image
    bubbles = detect_bubble_blocks(aligned)
    bubbles_clean = clean_bubbles_with_ml(bubbles, aligned)  # remove noise
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
    
