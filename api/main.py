from pathlib import Path
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

API_DIR = Path(__file__).resolve().parent
BASE_DIR = API_DIR.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.joblib"
INDEX_PATH = API_DIR / "static" / "index.html"
MODEL_VERSION = "1.0.0"

app = FastAPI(title="Netflix Customer Churn API", version=MODEL_VERSION)

model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None


class ChurnRequest(BaseModel):
    watch_hours: float = Field(ge=0)
    last_login_days: float = Field(ge=0)
    monthly_fee: float = Field(ge=0)
    number_of_profiles: float = Field(ge=0)
    avg_watch_time_per_day: float = Field(ge=0)
    subscription_type: str
    region: str
    device: str
    payment_method: str
    favorite_genre: str


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(INDEX_PATH)


@app.get("/health")
def health():
    return {"status": "ok", "model_available": model is not None, "model_version": MODEL_VERSION}


@app.post("/predict")
def predict(request: ChurnRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not found. Run training first.")
    row = pd.DataFrame([request.model_dump()])
    pred = int(model.predict(row)[0])
    prob = float(model.predict_proba(row)[0, 1])
    return {
        "prediction": pred,
        "prediction_label": "churned" if pred else "retained",
        "churn_probability": round(prob, 6),
        "model_version": MODEL_VERSION,
    }