from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.model.model import predict_pipeline, __version__ as model_version

app = FastAPI()

class PredictionRequest(BaseModel):
    input_data: dict

class PredictionOut(BaseModel):
    prediction: list

@app.get("/healthcheck")
async def healthcheck():
    return {"status": "healthy", "model_version": model_version}

@app.post("/predict", response_model=PredictionOut)
async def predict(prediction_request: PredictionRequest):
    try:
        prediction = predict_pipeline(prediction_request.input_data)
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))