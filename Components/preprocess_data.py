import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from datasets import Dataset
from pathlib import Path
from load_split_data import load_and_split_data

path = Path(__file__).resolve().parents[1]

vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words='english'
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased", use_fast=True)

tokenizer_params = {
    'padding' : 'max_length',
    'truncation' : True,
    'max_length' : 512,
}

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        **tokenizer_params
    )

def preprocess_data(filename:str):

    train_df,test_df = load_and_split_data(filename, 0.2,random_state=42)

    X_train, y_train = train_df['text'],train_df['label']
    X_test, y_test = test_df['text'],test_df['label']

    X_train = vectorizer.fit_transform(X_train)
    joblib.dump(vectorizer, f'{path}/models/vectorizer.joblib')
    X_test = vectorizer.transform(X_test)

    train_dataset = Dataset.from_pandas(
        train_df[["text", "label"]]
    )

    test_dataset = Dataset.from_pandas(
        test_df[["text", "label"]]
    )

    train_dataset = train_dataset.rename_column("label", "labels")
    test_dataset = test_dataset.rename_column("label", "labels")

    train_dataset = train_dataset.map(
        tokenize_fn,
        batched=True
    )

    test_dataset = test_dataset.map(
        tokenize_fn,
        batched=True
    )
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    test_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"]
    )

    return train_dataset, test_dataset, X_train, y_train, X_test, y_test,tokenizer