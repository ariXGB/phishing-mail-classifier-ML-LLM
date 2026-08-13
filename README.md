# Phishing Message Detector 

Okay so here's the thing about phishing detectors — everyone builds one, trains a single logistic regression on some Kaggle dataset, gets a suspiciously clean 96% accuracy, and calls it a day. I didn't want to do that. I wanted to know *which kind of model actually gets it*, and whether throwing a real transformer at the problem is worth the extra weight compared to something dumb and fast like Naive Bayes.

So this project trains four different models — Logistic Regression, Naive Bayes, XGBoost, and a fine-tuned MiniLM transformer — on the same phishing/legit text data, scores all of them with the exact same metrics, and then lets you actually run a message through any (or all) of them via a live API and a little Streamlit app.

Fair warning: this is not a production email security tool. Don't wire this into your inbox and trust it with your life. It's a proper ML comparison project — the kind of thing where you actually care about the pipeline being right, not just the demo looking good.

## Meet the models

* **Logistic Regression** — boring, dependable, does its job in milliseconds
* **Naive Bayes** — technically wrong about word independence and somehow fine anyway
* **XGBoost** — squeezes out a bit more accuracy than the other two just because it can
* **MiniLM** — the transformer. Actually reads the sentence instead of bag-of-words-ing its way through it. Slower, hungrier, but genuinely smarter about context

You can run one model on a message, or all four side by side, and watch where they agree and where they don't — which honestly is the more interesting part.

## What's actually happening under the hood

Text gets cleaned before it goes anywhere near a model — HTML tags stripped, URLs gone, email addresses scrubbed, only letters left standing. Phishing text is messy by nature (obfuscation is basically the whole game), so this step isn't optional, it's load-bearing.

The classical models (LR, NB, XGB) run on TF-IDF vectors. Nothing fancy, just solid word-frequency features that these algorithms are genuinely good at working with.

MiniLM gets the full treatment — tokenized, batched, fine-tuned with Hugging Face's `Trainer`, checkpointed every epoch, and the best version (by F1, not just accuracy, because accuracy alone lies to you on imbalanced data) gets kept and saved for later.

Every model, no matter how it was trained, gets evaluated through the exact same metrics function. Accuracy, precision, recall, F1 — computed identically across the board, so when you compare numbers you're actually comparing apples to apples and not four different definitions of "good."

Then a FastAPI backend serves predictions — one endpoint for a single message, one for a whole CSV of them — and a Streamlit frontend sits on top so you don't have to hit the API with curl like some kind of animal.

## Project layout

```
project/
│
├── main.py                    # FastAPI backend
├── app.py                     # Streamlit frontend
│
├── Components/
│   ├── load\_split\_data.py     # loads the csv, splits train/test
│   ├── preprocess\_data.py     # TF-IDF + tokenization
│   ├── predict.py             # PhishingPredictor - does the actual inference
│   ├── train\_models.py        # trains all four models
│   └── test\_models.py         # evaluates trained models on held-out data
│
├── Data/                      # datasets, raw and cleaned
├── models/                    # saved model files (.joblib + the MiniLM checkpoint)
└── Evaluation\_data/           # best\_params.json, eval metrics, test metrics
```

Training and serving are kept pretty separate on purpose. `train\_models.py` spits out model files, and `predict.py` just picks up whatever's sitting in `models/` and runs with it. So retraining a model doesn't mean touching a single line of API code — you just drop the new file in and it's live.

## Stack

Python, FastAPI, Pydantic on the backend. Streamlit on the frontend. scikit-learn and XGBoost for the classical models, TF-IDF for the vectorizing. PyTorch and Hugging Face Transformers for MiniLM. Pandas and joblib gluing it all together.

## Running it

Clone it, install the requirements:

```bash
git clone <repository-url>
cd project
pip install -r requirements.txt
```

**Train the models** (skip this if you're just using the models already sitting in `models/`)

```bash
python Components/train\_models.py
```

This trains all four and logs the numbers to `Evaluation\_data/eval\_metrics.csv`. Heads up — the MiniLM fine-tune wants a GPU. It'll technically run on CPU too, it'll just take a while, so maybe go make a coffee.

**Check performance on the test set**

```bash
python Components/test\_models.py
```

Runs everything against the held-out data and writes `Evaluation\_data/test\_metrics.csv`.

**Start the backend**

```bash
uvicorn main:app --reload
```

`http://localhost:8000`, docs at `/docs`.

**Start the frontend**

```bash
streamlit run app.py
```

`http://localhost:8501` — paste a message in, pick your model(s), hit Analyze. Or go to the batch tab and throw a CSV at it if you've got a whole pile of messages to check.

## A few things I was careful about

MiniLM doesn't load until someone actually asks for it. It sits there as `None` until the first request needs it, then stays loaded for anything after that — because there's no reason to eat a transformer's memory footprint if all anyone wanted was Naive Bayes.

Cleaning happens the same way every single time, whether it's one message or a thousand rows in a CSV, so the model always sees the kind of input it was trained on instead of raw, messy internet text.

And the metrics function is the single source of truth everywhere — training, evaluation, testing all call the same function, so there's no version drift where "accuracy" quietly means something slightly different in two different files.

## Where I'd take it next

* Actual input validation on the API — right now a typo'd model name or a missing CSV column just throws an ugly 500 instead of telling you what went wrong
* Real tests. `test\_models.py` runs real evaluations but doesn't assert anything — it's a script, not a test suite
* Ensembling the four models instead of just comparing them side by side
* Some kind of explainability layer — SHAP for the classical models, attention weights for MiniLM — so a verdict comes with an actual reason attached
* Docker, so "works on my machine" stops being the asterisk on every demo

## Why I built it this way

I got tired of the version of this project where you train one model, get a suspiciously good number, and never actually interrogate whether it's a good approach or just a lucky dataset. Comparing four genuinely different modeling philosophies — sparse vectors vs. gradient boosting vs. an actual transformer — on identical data with identical scoring felt like a much more honest way to learn what's actually going on, instead of pretending one model is "the answer."

