"""
Streamlit frontend for the Phishing Email Detector project.

Run with:
    streamlit run app.py

This file only handles UI. All ML logic lives in Components/predict.py
(the PhishingPredictor class you already built).
"""

import streamlit as st
import pandas as pd
import requests
import os
from dotenv import load_dotenv
import re

# Load environment variables from .env file
load_dotenv()

def clean_text(text: str) -> str:

    text = str(text).lower()

    text = re.sub(r'<.*?>', ' ', text)              # remove HTML tags
    text = re.sub(r'http\S+|www\S+', ' ', text)      # remove URLs
    text = re.sub(r'\S+@\S+', ' ', text)             # remove email addresses
    text = re.sub(r'[^a-z\s]', ' ', text)            # keep only letters
    text = re.sub(r'\s+', ' ', text).strip()         # collapse extra spaces

    return text

st.set_page_config(
    page_title="Phishing Mail Detector",
    page_icon="🛡️",
    layout="centered",
)

st.markdown(
    """
    <style>
    /* Overall font + spacing */
    .main {
        padding-top: 1.5rem;
    }
    /* Result card */
    .result-card {
        padding: 1rem 1.2rem;
        border-radius: 10px;
        margin-bottom: 0.8rem;
        border: 1px solid #2b2b2b22;
    }
    .phishing {
        background-color: #FF3600;
        border-left: 5px solid #e63946;
    }
    .legit {
        background-color: #00FF77;
        border-left: 5px solid #2a9d3f;
        color: black;
    }
    .model-name {
        font-weight: 600;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.7;
    }
    .verdict {
        font-size: 1.3rem;
        font-weight: 700;
        margin: 0.2rem 0;
    }
    .confidence {
        font-size: 0.85rem;
        opacity: 0.75;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

FASTAPI_URL = os.getenv("FASTAPI_URL")

MODEL_LABELS = {
    "lr": "Logistic Regression",
    "nb": "Naive Bayes",
    "xgb": "XGBoost",
    "minilm": "MiniLM",
}



st.sidebar.header("⚙️ Settings")
st.sidebar.write("Choose one or more models to run the prediction on the input text.")

selected_models = []
for key, label in MODEL_LABELS.items():
    # model checked by default so a new user sees a result immediately
    default_checked = key == "minilm"
    if st.sidebar.checkbox(label, value=default_checked, key=f"cb_{key}"):
        selected_models.append(key)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Tip: Select all 4 models to compare how classical ML vs. "
    "a transformer model handle the same message."
)


st.title("🛡️ Phishing Message Detector")
st.write(
    "Paste an email or message below. The selected model(s) will "
    "tell you whether it looks like **phishing** or **legitimate** text."
)

tab_single, tab_batch = st.tabs(["✉️ Single Message", "📄 Batch (CSV)"])


with tab_single:
    text_input = st.text_area(
        "Message text",
        height=180,
        placeholder="e.g. Your account has been suspended. Click here to verify...",
    )
    text_input = clean_text(text_input)
    predict_clicked = st.button("Analyze", type="primary", use_container_width=True)

    if predict_clicked:
        if not text_input.strip():
            st.warning("The entered text does not contain any usable content.")        
        elif not selected_models:
            st.warning("Please select at least one model from the sidebar.")
        else:
            st.subheader("Results")

            for model_key in selected_models:
                with st.spinner(f"Running {MODEL_LABELS[model_key]}..."):

                    try:
                        response = requests.post(
                            f"{FASTAPI_URL}/predict-text",
                            data={"text": text_input, "model_names": [model_key]},
                        ).json()[0]

                    except requests.RequestException as e:
                        st.error(f"Error occurred while fetching prediction for {MODEL_LABELS[model_key]}: {e}")
                        continue

                is_phishing = int(response["prediction"]) == 1
                verdict_text = "⚠️ Model predicts Likely Phishing" if is_phishing else "✅ Model predicts Likely Legitimate"
                card_class = "phishing" if is_phishing else "legit"

                confidence = response["confidence"]
                confidence_text = (
                    f"Confidence: {confidence * 100:.1f}%"
                    if confidence is not None
                    else "Confidence: N/A"
                )

                st.markdown(
                    f"""
                    <div class="result-card {card_class}">
                        <div class="model-name">{MODEL_LABELS[model_key]}</div>
                        <div class="verdict">{verdict_text}</div>
                        <div class="confidence">{confidence_text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


# ------------------------------------------------------------------
# TAB 2 — BATCH PREDICTION VIA CSV UPLOAD
# ------------------------------------------------------------------
with tab_batch:
    st.write("Upload a CSV file with a text column to score many messages at once.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head(), use_container_width=True)

        text_column = st.selectbox("Which column contains the message text?", df.columns)

        # Batch mode only makes sense with one model at a time,
        # since predict_dataframe() is designed per-model.
        batch_model = st.selectbox(
            "Model to use for batch prediction",
            options=list(MODEL_LABELS.keys()),
            format_func=lambda k: MODEL_LABELS[k],
        )

        if st.button("Run Batch Prediction", type="primary"):

            with st.spinner("Scoring all rows..."):
                try:
                    response = requests.post(
                        f"{FASTAPI_URL}/predict-csv",
                        files={
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "text/csv"
                            )
                        },
                        data={
                            "model_name": batch_model,
                            "text_column": text_column
                        }
                    )

                    response.raise_for_status()

                    # Convert the result back to a DataFrame for display and download
                    result_df = pd.DataFrame(response.json())
                    progress=True

                except requests.RequestException as e:
                    progress=False
                    st.error(f"Error occurred while fetching batch prediction: {e}")
                    
            if progress:
                st.success(f"Done! Scored {len(result_df)} rows.")
                st.dataframe(result_df, use_container_width=True)

                csv_bytes = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_bytes,
                    file_name="phishing_predictions.csv",
                    mime="text/csv",
                )