import pandas as pd
from Components.predict.predict import PhishingPredictor
from Components.project_paths import PROJECT_ROOT
from Components.train.train_models import calculate_metrics
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
path = PROJECT_ROOT
predictor = PhishingPredictor()

test_df = pd.read_csv(path / "Data" / "test_dataset_cleaned.csv")

test_metrics_df = pd.DataFrame(columns=["model", "accuracy", "precision", "recall", "f1"])

for model_name in ["lr", "nb", "xgb", "minilm"]:
    _, preds, _ = predictor.predict_dataframe(
        test_df,
        text_column="text",
        model_name=model_name
    )

    test_df[f"prediction_{model_name}"] = preds


for model_name in ["lr", "nb", "xgb", "minilm"]:
    metrics = calculate_metrics(test_df["label"], test_df[f"prediction_{model_name}"], model_name=model_name)   
    test_metrics_df.loc[len(test_metrics_df)] = metrics

test_metrics_df.to_csv(path / "Evaluation_data" / "test_metrics.csv", index=False)