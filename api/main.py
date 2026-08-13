from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from pydantic import BaseModel
import re
from Components.predict import PhishingPredictor
import pandas as pd
from io import StringIO


predictor = PhishingPredictor(loadBert=False)

class PhishingResponse(BaseModel):
    model_name: str
    prediction: int
    confidence: float

def clean_text(text: str) -> str:
    
    text = str(text).lower()

    text = re.sub(r'<.*?>', ' ', text)              # remove HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)      # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)             # remove email addresses
    text = re.sub(r'[^a-z\s]', ' ', text)            # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()         # collapse extra spaces

    return text

app = FastAPI(version="1.0",description="Phishing Email Classifier API")

@app.post("/predict-text", response_model=list[PhishingResponse])
async def predict_text(text: str = Form(...), model_names: list[str] = Form(...)):

    if "minilm" in model_names and predictor.trainer is None:
        predictor.load_minilm()

    cleaned_text = clean_text(text)
    preds_list : list[PhishingResponse] = []

    for model_name in model_names:
        results = predictor.predict_text(cleaned_text, model_name)
        preds_list.append(PhishingResponse(model_name=model_name, prediction=results['prediction'], confidence=results['confidence']))

    return preds_list

@app.post("/predict-csv", response_model=list[dict])
async def predict_csv(file: UploadFile = File(...), model_name: str = Form(...),text_column: str = Form(...)):

    if model_name == "minilm":
        predictor.load_minilm()

    try:
        contents = await file.read()
        csv_content = contents.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(
            status_code=400,
            detail="CSV must be UTF-8 encoded."
        )

    df = pd.read_csv(StringIO(csv_content)) 
    df,_,_ = predictor.predict_dataframe(df, text_column, model_name)    
    return df.to_dict(orient="records")






