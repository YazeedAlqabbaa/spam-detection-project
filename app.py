import os
import re
import time
import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "rf_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
    vectorizer = pickle.load(f)


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'\+?\d[\d -]{8,}\d', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = ' '.join(text.split())
    return text


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    email_text = data.get("email", "")

    if not email_text or not email_text.strip():
        return jsonify({"error": "No email text provided"}), 400

    start = time.perf_counter()

    cleaned = preprocess(email_text)
    features = vectorizer.transform([cleaned])
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0]

    elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

    # Model returns string labels: 'spam' or 'ham'
    is_spam = bool(str(prediction) == "1" or str(prediction).lower() == "spam")
    confidence = float(np.max(proba)) * 100

    return jsonify({
        "result": "spam" if is_spam else "legitimate",
        "confidence": round(confidence, 2),
        "word_count": len(email_text.split()),
        "processing_time_ms": elapsed_ms,
    })


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
