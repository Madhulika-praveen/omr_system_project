from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from typing import Optional
from omr.omr_processor import process_sheet

app = FastAPI(title="OMR FastAPI Service")

@app.get("/")
def read_root():
    return {"message": "OMR API is running"}

@app.post("/upload/")
async def upload_sheet(file: UploadFile = File(...), set_id: Optional[str] = "A"):
    try:
        result = process_sheet(file.file, set_id.upper())
        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
