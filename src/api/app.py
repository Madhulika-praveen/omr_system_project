# app.py
from fastapi import FastAPI
from omr.preprocessing import preprocess_sheet
from omr.bubble_detection import (
    extract_bubbles, 
    detect_bubble_blocks, 
    generate_grid_coordinates, 
    clean_bubbles_with_ml
)
from omr.scoring import load_answer_key, score_sheet
from fastapi.responses import JSONResponse
from typing import Optional
import numpy as np
import cv2

app = FastAPI(title="OMR FastAPI Service")

# Load answer key
ANSWER_KEY = {}
try:
    ANSWER_KEY = load_answer_key("data/answers/ground_truth_setA.csv")
except Exception as e:
    print("WARNING: Could not load answer key:", e)

@app.get("/")
def read_root():
    return {"message": "OMR API is running"}

def process_sheet(file_stream, set_id: str = "A"):
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

@app.post("/upload/")
async def upload_sheet(file: UploadFile = File(...), set_id: Optional[str] = "A"):
    try:
        result = process_sheet(file.file, set_id.upper())
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
