from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from inference import PneumoniaModel

app = FastAPI(title="AeroScan API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../ml/pneumonia_model.pth')
model = PneumoniaModel(MODEL_PATH)

@app.get("/")
def read_root():
    return {"message": "AeroScan Pneumonia Detection API is running."}

@app.post("/predict")
async def predict_pneumonia(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")
    
    try:
        contents = await file.read()
        result = model.predict(contents)
        return {"filename": file.filename, "result": result}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
