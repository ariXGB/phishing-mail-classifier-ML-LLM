import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from xgboost import XGBClassifier
import json
from pathlib import Path
import torch
import joblib
import pandas as pd
import random

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

from preprocess_data import preprocess_data

path = Path(__file__).resolve().parents[1]

with open(f"{path}/Evaluation_data/best_params.json", "r") as f:
    best_params = json.load(f)

models = {
    'lr':LogisticRegression(**best_params['lr']),
    'nb':MultinomialNB(**best_params['nb']),
    'xgb':XGBClassifier(**best_params['xgb']),
    'minilm': AutoModelForSequenceClassification.from_pretrained(best_params['minilm']['model_name'],num_labels=best_params['minilm']['num_labels']
),
}

eval_metrics_df = pd.DataFrame(columns=['model','accuracy','precision','recall','f1'])

def calculate_metrics(y_true, y_pred, model_name=None):
    metrics = {    
        "accuracy": round(accuracy_score(y_true, y_pred) * 100, 3),
        "precision": round(precision_score(y_true, y_pred) * 100, 3),
        "recall": round(recall_score(y_true, y_pred) * 100, 3),
        "f1": round(f1_score(y_true, y_pred) * 100, 3),
    }
    

    if model_name:
        metrics["model"] = model_name

    return metrics

def train_minilm(train_dataset, test_dataset, tokenizer):

    def metrics(eval_pred):
        logits,labels = eval_pred
        predictions = np.argmax(logits, axis=1)

        metric = calculate_metrics(labels, predictions)
        return metric

    model = models['minilm']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(device)

    training_args = TrainingArguments(
        seed=SEED,
        output_dir=f"{path}/models/minilm_checkpoints",
        num_train_epochs=3,              
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none"                 # disables wandb/etc logging prompts
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=metrics
    )

    trainer.train()
    final_metrics = trainer.evaluate()

    eval_metrics_df.loc[len(eval_metrics_df)] = {
    'model': 'minilm',
    'accuracy': final_metrics['eval_accuracy'],
    'precision': final_metrics['eval_precision'],
    'recall': final_metrics['eval_recall'],
    'f1': final_metrics['eval_f1']
}

    # Save the fine-tuned model + tokenizer for inference later
    trainer.save_model(f"{path}/models/minilm_phishing")
    tokenizer.save_pretrained(f"{path}/models/minilm_phishing")

    return final_metrics

def eval_ml_metrics(model, name, X_test, y_test):

    predictions = model.predict(X_test)
 
    metric = calculate_metrics(y_test, predictions, model_name=name)
    eval_metrics_df.loc[len(eval_metrics_df)] = metric
    print(metric)

def train():

    train_dataset, test_dataset, X_train, y_train, X_test, y_test, tokenizer = preprocess_data("final_train_dataset.csv")

    for name,model in models.items():

        print(f'Training model {name}')

        if name == 'minilm':
            minilm_metrics = train_minilm(train_dataset, test_dataset, tokenizer)
            print(minilm_metrics)
            continue

        model.fit(X_train,y_train)
        joblib.dump(model, path / "models" / f"{name}_model.joblib")
        eval_ml_metrics(model,name, X_test, y_test)

    eval_metrics_df.to_csv(f'{path}/Evaluation_data/eval_metrics.csv', index=False)
    print('Training Completed & metrics recorded for the models.')

if __name__ == "__main__":
    train()