from typing import Literal

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from Components.predict import PhishingPredictor
import pandas as pd
from io import StringIO


predictor = PhishingPredictor(loadBert=False)

ModelName = Literal["lr", "nb", "xgb", "minilm"]

class PhishingResponse(BaseModel):
    model_name: str
    prediction: int
    confidence: float

app = FastAPI(version="1.0", description="Phishing Email Classifier API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict-text", response_model=list[PhishingResponse])
async def predict_text(text: str = Form(...), model_names: list[ModelName] = Form(...)):

    if not text.strip():
        raise HTTPException(status_code=422, detail="Text must not be empty.")

    if "minilm" in model_names and predictor.trainer is None:
        predictor.load_minilm()

    preds_list: list[PhishingResponse] = []

    for model_name in model_names:
        results = predictor.predict_text(text, model_name)
        preds_list.append(
            PhishingResponse(
                model_name=model_name,
                prediction=results["prediction"],
                confidence=results["confidence"],
            )
        )

    return preds_list


@app.post("/predict-csv", response_model=list[dict])
async def predict_csv(
    file: UploadFile = File(...),
    model_name: ModelName = Form(...),
    text_column: str = Form(...),
):

    if model_name == "minilm":
        predictor.load_minilm()

    try:
        contents = await file.read()
        csv_content = contents.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="CSV must be UTF-8 encoded.")

    try:
        df = pd.read_csv(StringIO(csv_content))
    except pd.errors.EmptyDataError:
        raise HTTPException(status_code=400, detail="The uploaded CSV is empty.")
    except pd.errors.ParserError as exc:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded CSV has no rows.")

    df, _, _ = predictor.predict_dataframe(df, text_column, model_name)
    return df.to_dict(orient="records")