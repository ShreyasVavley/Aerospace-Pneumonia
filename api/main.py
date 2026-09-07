from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
import urllib.request
from contextlib import asynccontextmanager
from inference import PneumoniaModel

import torch
torch.set_num_threads(1)

def ping_self():
    external_url = os.getenv("RENDER_EXTERNAL_URL")
    if not external_url:
        return
    try:
        urllib.request.urlopen(external_url)
        print(f"Pinged {external_url} to keep backend active.")
    except Exception as e:
        print(f"Keep-alive ping failed: {e}")

async def keep_alive_loop():
    while True:
        await asyncio.sleep(10 * 60) # Wait 10 minutes
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, ping_self)
        except Exception:
            pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the keep-alive task
    task = asyncio.create_task(keep_alive_loop())
    yield
    # Cancel the task on shutdown
    task.cancel()

app = FastAPI(title="AeroScan API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=False,
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
