from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer
)

def clean_data(df: pd.DataFrame, text_column: str) -> pd.DataFrame:
    """
    Cleans the text data in the specified column of the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing text data.
        text_column (str): The name of the column containing text to be cleaned.
    
    Returns:
        pd.DataFrame: A new DataFrame with cleaned text data.
    """
    df = df.copy()
    df[text_column] = df[text_column].astype(str).str.lower()
    df[text_column] = df[text_column].str.replace(r'<.*?>', ' ', regex=True)  # remove HTML tags
    df[text_column] = df[text_column].str.replace(r'http\S+|www\S+', ' ', regex=True)  # remove URLs
    df[text_column] = df[text_column].str.replace(r'\S+@\S+', ' ', regex=True)  # remove email addresses
    df[text_column] = df[text_column].str.replace(r'[^a-z\s]', ' ', regex=True)  # keep only letters
    df[text_column] = df[text_column].str.replace(r'\s+', ' ', regex=True).str.strip()  # collapse extra spaces
    
    return df

class PhishingPredictor:

    def __init__(self,loadBert=True):

        path = Path(__file__).resolve().parents[1]

        device = torch.device('cuda')

        self.vectorizer = joblib.load(path / "models" / "vectorizer.joblib")

        self.models = {
            "lr": joblib.load(path / "models" / "lr_model.joblib"),
            "nb": joblib.load(path / "models" / "nb_model.joblib"),
            "xgb": joblib.load(path / "models" / "xgb_model.joblib")
        }

        self.minilm_path = path/ "models"/ "minilm_phishing"

        self.tokenizer = None
        self.minilm = None
        self.trainer = None

        if loadBert:
            self.load_minilm()

    # MINILM

    def load_minilm(self):
        # Skip reloading from disk if already loaded — this used to
        # reload on every single prediction call, which was slow.
        if self.trainer is not None:
            return

        self.tokenizer = AutoTokenizer.from_pretrained(self.minilm_path, use_fast=True)
        self.minilm = AutoModelForSequenceClassification.from_pretrained(self.minilm_path, num_labels=2)
        self.trainer = Trainer(model=self.minilm)

    def _predict_minilm(self, texts):

        self.load_minilm()  # no-op if already loaded
        dataset = Dataset.from_dict({"text": texts})

        def tokenize(batch):

            return self.tokenizer(
                batch["text"],
                truncation=True,
                padding="max_length",
                max_length=512
            )

        dataset = dataset.map(tokenize,batched=True)

        output = self.trainer.predict(dataset)
        logits = output.predictions
        probs = torch.softmax(torch.tensor(logits),dim=1).numpy()
        predictions = np.argmax(logits,axis=1)
        confidence = probs.max(axis=1)

        return (predictions,confidence)

    # CLASSICAL ML

    def _predict_ml(self,model_name,texts):

        X = self.vectorizer.transform(texts)

        model = self.models[model_name]

        predictions = model.predict(X)  
        probs = model.predict_proba(X)  # All models below (LogR, NB, XGBoost) support predict_proba, so _predict_ml always gets a confidence score.

        confidence = np.max(probs,axis=1)
        
        return (predictions,confidence)

    
    # SINGLE TEXT

    def predict_text(self,text,model_name):

        if model_name == "minilm":

            preds, confs = self._predict_minilm([text])
        
            return {
                "prediction":int(preds[0]),
                "confidence":float(confs[0]),
            }

        preds, confs = (self._predict_ml(model_name,[text]))

        return {
            "prediction":int(preds[0]),
            "confidence":float(confs[0])
        }

    # DATAFRAME

    def predict_dataframe(self,df: pd.DataFrame,text_column,model_name):

        df = clean_data(df,text_column)
        texts = df[text_column].astype(str).tolist()

        if model_name == "minilm":
            preds, confs = (self._predict_minilm(texts))
            df["prediction"] = preds
            df["confidence"] = confs
            return df, preds, confs

        preds, confs = (self._predict_ml(model_name,texts))
        df["prediction"] = preds
        df["confidence"] = confs

        return df, preds,confs



if __name__ == "__main__":
    predictor = PhishingPredictor(loadBert=True)
    test_df = pd.read_csv("synthetic_test_dataset.csv")
    
    result_df, preds, confs = predictor.predict_dataframe(
        test_df,
        text_column="text",
        model_name="minilm"
    )