import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import HTTPException

__version__ = "0.1.0"

BASE_DIR = Path(__file__).resolve(strict=True).parent

def load_model():
    with open(BASE_DIR / f"trained_pipeline-{__version__}.pkl", "rb") as f:
        model = joblib.load(f)
    return model

model = load_model()

def predict_pipeline(input_data):
    try:
        input_df = pd.DataFrame([input_data])  # Ensure input is in the correct format
        prediction = model.predict(input_df)
        return prediction.tolist()
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))