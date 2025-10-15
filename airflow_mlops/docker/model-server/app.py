import os, pickle
from fastapi import FastAPI
from pydantic import BaseModel

MODELS = os.environ.get("MODELS_DIR","/models")
SLOT = os.environ.get("MODEL_SLOT","blue")
MODEL_PATHS = [f"{MODELS}/model_{SLOT}.pkl", f"{MODELS}/model_green.pkl", f"{MODELS}/candidate.pkl"]

def load_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            with open(p,"rb") as f:
                return pickle.load(f)
    return {"threshold": 0.5, "p1": 0.5}

app = FastAPI()
model = load_model()

class Item(BaseModel):
    feat_mean: float

@app.get("/health")
def health():
    return {"status":"ok","slot":SLOT}

@app.post("/predict")
def predict(item: Item):
    """"
    Simple prediction logic based on a threshold."""
    x = item.feat_mean
    p1 = model["p1"] if x >= model["threshold"] else 1 - model["p1"]
    return {"score": round(p1,3)}
