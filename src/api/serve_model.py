import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pathlib import Path
import uvicorn
import sys
import numpy as np

# --- CONFIGURATION ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"

# --- APP INITIALIZATION ---
app = FastAPI(
    title="Adey Innovations Fraud Detection API",
    description="A Real-time API to detect fraudulent transactions.",
    version="1.0.0"
)

# --- LOAD MODEL ---
print(f"📂 Project Root detected as: {BASE_DIR}")
print(f"⏳ Attempting to load model from: {MODEL_PATH}")

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model Loaded Successfully.")

    # CRITICAL: Verify model expects features
    if hasattr(model, "feature_names_in_"):
        print(f"ℹ️ Model expects {len(model.feature_names_in_)} features.")
    else:
        print("⚠️ Warning: Model does not store feature names. Input order must be perfect manually.")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    sys.exit(1)


# --- INPUT SCHEMA ---
class TransactionData(BaseModel):
    features: dict


# --- ENDPOINTS ---
@app.get("/")
def home():
    return {"message": "Fraud Detection API is Running."}


@app.post("/predict")
def predict(data: TransactionData):
    try:
        # 1. Convert input JSON to DataFrame
        input_df = pd.DataFrame([data.features])

        # 2. FORCE COLUMN ALIGNMENT (The Fix)
        # We assume the model HAS feature_names_in_ based on your previous logs.
        # We list() it to ensure it's compatible with pandas reindex.
        expected_cols = list(model.feature_names_in_)

        # This reorders columns to match training and fills missing ones with 0
        input_df = input_df.reindex(columns=expected_cols, fill_value=0)

        # 3. Make Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": int(prediction),
            "fraud_probability": float(probability),
            "is_fraud": bool(prediction == 1),
            "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.3 else "Low"
        }

    except Exception as e:
        # Return the specific error to the user
        raise HTTPException(status_code=400, detail=str(e))


# --- RUN SERVER ---
if __name__ == "__main__":
    uvicorn.run("serve_model:app", host="0.0.0.0", port=8000, reload=True)