from fastapi import FastAPI, HTTPException
import pickle
import numpy as np
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
from .schemas import PredictRequest, PredictResponse

app = FastAPI(title="Criteo Uplift Inference API")

# Enable CORS for the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the student model at startup
MODEL_PATH = Path("results/production_uplift_model.pkl")
model = None

@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    else:
        print(f"Warning: Model not found at {MODEL_PATH}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
        
    # Extract features in order
    features = np.array([[
        request.f0, request.f1, request.f2, request.f3, 
        request.f4, request.f5, request.f6, request.f7, 
        request.f8, request.f9, request.f10, request.f11
    ]])
    
    try:
        score = float(model.predict(features)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    # Business Logic: Bid if (Uplift * LTV) > Cost
    LTV = 10.0
    COST = 0.10
    
    bid_eligible = (score * LTV) > COST
    decision = "BID" if bid_eligible else "DO NOT BID"
    
    return PredictResponse(
        uplift_score=score,
        decision=decision,
        bid_eligible=bid_eligible
    )
